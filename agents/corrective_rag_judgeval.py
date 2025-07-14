import os
from typing import Dict, TypedDict, List
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langgraph.graph import END, StateGraph
import tempfile
import json
import re

# Judgeval imports
from judgeval.tracer import Tracer
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer, 
    AnswerRelevancyScorer,
    HallucinationScorer,
    AnswerCorrectnessScorer
)

# Initialize Judgeval
judgment = Tracer(project_name="corrective_rag_agents")
judgment_client = JudgmentClient()

# Initialize session state
def initialize_session_state():
    """Initialize session state variables for API keys and model selection."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        st.session_state.doc_url = "https://arxiv.org/pdf/2307.09288.pdf"
        st.session_state.model_provider = "google"  # groq or google
        st.session_state.model_name = "llama-3.1-70b-versatile"  # default Groq model

def setup_sidebar():
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model selection
        st.subheader("Model Provider")
        st.session_state.model_provider = st.selectbox(
            "Select LLM Provider",
            ["groq", "google"],
            help="Choose between Groq (Mixtral/Llama) or Google (Gemini)"
        )
        
        if st.session_state.model_provider == "groq":
            st.session_state.groq_api_key = st.text_input(
                "Groq API Key", 
                value=st.session_state.groq_api_key, 
                type="password",
                help="Get free API key from console.groq.com"
            )
            st.session_state.model_name = st.selectbox(
    "Groq Model",
    ["llama-3.2-90b-text-preview", "llama-3.2-70b-versatile", "llama-3.2-11b-text-preview", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
)
        else:
            st.session_state.google_api_key = st.text_input(
                "Google API Key", 
                value=st.session_state.google_api_key, 
                type="password",
                help="Get free API key from makersuite.google.com"
            )
            st.session_state.model_name = "gemini-1.5-flash"
        
        st.session_state.tavily_api_key = st.text_input(
            "Tavily API Key (Optional)", 
            value=st.session_state.tavily_api_key, 
            type="password",
            help="For web search functionality"
        )
        
        st.session_state.doc_url = st.text_input(
            "Default Document URL", 
            value=st.session_state.doc_url
        )
        
        # Validate required keys
        if st.session_state.model_provider == "groq" and not st.session_state.groq_api_key:
            st.error("Please provide Groq API key")
            st.stop()
        elif st.session_state.model_provider == "google" and not st.session_state.google_api_key:
            st.error("Please provide Google API key")
            st.stop()
        
        st.session_state.initialized = True

# Initialize states
initialize_session_state()
setup_sidebar()

# Create LLM based on provider selection
@judgment.observe(span_type="llm_init")
def get_llm():
    """Initialize LLM based on selected provider."""
    if st.session_state.model_provider == "groq":
        return ChatGroq(
            model_name=st.session_state.model_name,
            api_key=st.session_state.groq_api_key,
            temperature=0,
            max_tokens=1000
        )
    else:
        return ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Updated to currently available model
    google_api_key=st.session_state.google_api_key,
    temperature=0,
    max_tokens=1000
)

# Initialize free embeddings (HuggingFace)
@st.cache_resource
def get_embeddings():
    """Get free HuggingFace embeddings."""
    try:
        # Show progress
        progress_text = st.empty()
        progress_text.text("Initializing embeddings model...")
        
        # Try to create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Test the embeddings
        test_text = "This is a test"
        test_embedding = embeddings.embed_query(test_text)
        progress_text.text(f"Embeddings initialized! Dimension: {len(test_embedding)}")
        
        return embeddings
    
        
    except Exception as e:
        st.error(f"Failed to load embeddings model: {str(e)}")
        st.error("Try: pip install sentence-transformers")
        st.stop()

# Initialize embeddings
embeddings = get_embeddings()

retriever = None

# Document loading functions
@judgment.observe(span_type="tool", name="load_documents")
def load_documents(file_or_url: str, is_url: bool = True) -> list:
    """Load documents from URL or file."""
    try:
        if is_url:
            loader = WebBaseLoader(file_or_url)
            loader.requests_per_second = 1
        else:
            file_extension = os.path.splitext(file_or_url)[1].lower()
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_or_url)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_or_url)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        
        docs = loader.load()
        return docs
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return []

# GraphState definition
class GraphState(TypedDict):
    keys: Dict[str, any]

# Retrieve function with Judgeval
@judgment.observe(span_type="function", name="retrieve")
def retrieve(state):
    """Retrieve relevant documents."""
    print("~-retrieve-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    
    if retriever is None:
        return {"keys": {"documents": [], "question": question}}
    
    documents = retriever.invoke(question)
    
    # Run async evaluation on retrieval quality
    if documents:
        judgment.async_evaluate(
            scorers=[AnswerRelevancyScorer(threshold=0.5)],
            input=question,
            actual_output="\n".join([doc.page_content[:200] for doc in documents[:3]]),
            model="gpt-4o"  # Using Judgment's judge model
        )
    
    return {"keys": {"documents": documents, "question": question}}

# Generate function with Judgeval
@judgment.observe(span_type="function", name="generate")
def generate(state):
    """Generate answer using selected LLM."""
    print("~-generate-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    
    try:
        llm = get_llm()
        
        prompt = PromptTemplate(
            template="""Based on the following context, please answer the question.
            
Context: {context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        context = "\n\n".join(doc.page_content for doc in documents) if documents else "No context available."
        
        # Create and run chain
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question} 
            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        generation = rag_chain.invoke({})
        
        # Run async evaluation on generation
        if documents:
            judgment.async_evaluate(
                scorers=[
                    FaithfulnessScorer(threshold=0.7),
                    AnswerRelevancyScorer(threshold=0.6)
                ],
                input=question,
                actual_output=generation,
                retrieval_context=[doc.page_content for doc in documents[:3]],
                model="gpt-4o"
            )
        
        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation
            }
        }
        
    except Exception as e:
        error_msg = f"Error in generate function: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return {"keys": {"documents": documents, "question": question, 
                "generation": "Sorry, I encountered an error while generating the response."}}

# Grade documents function with Judgeval
@judgment.observe(span_type="function", name="grade_documents")
def grade_documents(state):
    """Grade relevance of retrieved documents."""
    print("~-check relevance-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    
    llm = get_llm()
    
    prompt = PromptTemplate(
        # Continuing from grade_documents function...
        template="""You are grading the relevance of a retrieved document to a user question.
Return ONLY a JSON object with a "score" field that is either "yes" or "no".
Do not include any other text or explanation.

Document: {context}
Question: {question}

Rules:
- Check for related keywords or semantic meaning
- Use lenient grading to only filter clear mismatches
- Return exactly like this example: {{"score": "yes"}} or {{"score": "no"}}""",
        input_variables=["context", "question"]
    )
    
    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )
    
    filtered_docs = []
    search = "No"
    
    for d in documents:
        try:
            response = chain.invoke({"question": question, "context": d.page_content})
            json_match = re.search(r'\{.*\}', response)
            if json_match:
                response = json_match.group()
            
            score = json.loads(response)
            
            if score.get("score") == "yes":
                print("~-grade: document relevant-~")
                filtered_docs.append(d)
            else:
                print("~-grade: document not relevant-~")
                search = "Yes"
                
        except Exception as e:
            print(f"Error grading document: {str(e)}")
            filtered_docs.append(d)
            continue
    
    return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}

# Transform query function with Judgeval
@judgment.observe(span_type="function", name="transform_query")
def transform_query(state):
    """Transform the query to produce a better question."""
    print("~-transform query-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    
    llm = get_llm()
    
    prompt = PromptTemplate(
        template="""Generate a search-optimized version of this question by 
analyzing its core semantic meaning and intent.
\n ------- \n
{question}
\n ------- \n
Return only the improved question with no additional text:""",
        input_variables=["question"],
    )
    
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})
    
    return {
        "keys": {"documents": documents, "question": better_question}
    }

# Mock web search for free alternative
@judgment.observe(span_type="tool", name="web_search")
def web_search(state):
    """Web search simulation (or use Tavily if API key provided)."""
    print("~-web search-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    
    if st.session_state.tavily_api_key:
        # Use actual Tavily search if API key is provided
        from langchain_community.tools import TavilySearchResults
        tool = TavilySearchResults(
            api_key=st.session_state.tavily_api_key,
            max_results=3
        )
        try:
            results = tool.invoke({"query": question})
            web_doc = Document(
                page_content=str(results),
                metadata={"source": "tavily_search", "query": question}
            )
            documents.append(web_doc)
        except Exception as e:
            st.warning(f"Web search failed: {e}")
    else:
        # Mock web search result
        mock_result = f"Web search results for '{question}': This is a simulated result demonstrating web search capability."
        web_doc = Document(
            page_content=mock_result,
            metadata={"source": "mock_web_search", "query": question}
        )
        documents.append(web_doc)
    
    return {"keys": {"documents": documents, "question": question}}

# Decision function
def decide_to_generate(state):
    """Decide whether to generate or search web."""
    print("~-decide to generate-~")
    state_dict = state["keys"]
    search = state_dict["run_web_search"]
    
    if search == "Yes":
        print("~-decision: transform query and run web search-~")
        return "transform_query"
    else:
        print("~-decision: generate-~")
        return "generate"

# Build the workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

# Build graph edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# Streamlit UI
st.title("🔄 Corrective RAG Agent with Judgeval")
st.markdown("**Enhanced with Judgment Labs tracing and evaluation**")

# Document input section
# Document input section
st.subheader("📄 Document Input")
input_option = st.radio("Choose input method:", ["URL", "File Upload", "Use Sample Text"])

docs = None

if input_option == "URL":
    url = st.text_input("Enter document URL:", value=st.session_state.doc_url)
    if url and st.button("Load URL"):
        with st.spinner("Loading document..."):
            docs = load_documents(url, is_url=True)
            if docs:
                st.success(f"Loaded {len(docs)} pages from URL")
                
elif input_option == "File Upload":
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'txt', 'md'])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            docs = load_documents(tmp_file.name, is_url=False)
            if docs:
                st.success(f"Loaded {len(docs)} pages from file")
        os.unlink(tmp_file.name)
        
else:  # Use Sample Text
    # Create a sample document for testing
    sample_text = """
    Artificial Intelligence and Machine Learning Overview
    
    Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn. The field of AI research has been highly successful in developing effective techniques for solving a wide range of problems.
    
    Machine Learning (ML) is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. ML focuses on the development of computer programs that can access data and use it to learn for themselves.
    
    Deep Learning is a subset of ML that uses neural networks with multiple layers. These neural networks attempt to simulate the behavior of the human brain to "learn" from large amounts of data.
    
    Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.
    
    Computer Vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.
    """
    docs = [Document(page_content=sample_text, metadata={"source": "sample"})]
    st.info("Using sample text document for testing")

# Process documents and create retriever
if docs:
    with st.spinner("Processing documents..."):
        try:
            # Filter out empty documents
            docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
            
            if not docs:
                st.error("All documents are empty. Please try a different file.")
                st.stop()
            
            # Show document preview
            st.write(f"Document preview: {docs[0].page_content[:200]}...")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=50,  # Reduced overlap
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            all_splits = text_splitter.split_documents(docs)
            
            # Filter empty chunks
            all_splits = [split for split in all_splits if split.page_content.strip()]
            
            if not all_splits:
                st.error("No valid text chunks created. Document might be too short or improperly formatted.")
                st.stop()
            
            st.write(f"Created {len(all_splits)} text chunks")
            
            # Try to create vector store with better error handling
            try:
                # First, test if embeddings work
                test_embedding = embeddings.embed_query("test")
                st.write(f"Embedding dimension: {len(test_embedding)}")
                
                # Create vector store
                vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.success(f"✅ Created vector store with {len(all_splits)} chunks")
                
            except Exception as e:
                st.error(f"Error creating vector store: {str(e)}")
                st.error("Trying alternative approach...")
                
                # Alternative: Create vectorstore manually
                try:
                    # Create Chroma client
                    import chromadb
                    chroma_client = chromadb.PersistentClient(path="./chroma_db")
                    
                    # Delete collection if exists
                    try:
                        chroma_client.delete_collection("langchain")
                    except:
                        pass
                    
                    # Create new collection
                    collection = chroma_client.create_collection("langchain")
                    
                    # Create vectorstore with explicit collection
                    vectorstore = Chroma(
                        client=chroma_client,
                        collection_name="langchain",
                        embedding_function=embeddings
                    )
                    
                    # Add documents one by one
                    for i, doc in enumerate(all_splits):
                        try:
                            vectorstore.add_documents([doc])
                        except Exception as doc_error:
                            st.warning(f"Skipped chunk {i}: {str(doc_error)}")
                    
                    retriever = vectorstore.as_retriever()
                    st.success("✅ Created vector store using alternative method")
                    
                except Exception as alt_error:
                    st.error(f"Alternative method also failed: {str(alt_error)}")
                    st.error("Please try restarting the app or using the sample text option.")
                    st.stop()
                    
        except Exception as e:
            st.error(f"Document processing error: {str(e)}")
            st.error("Please try using the 'Use Sample Text' option to test the system.")
            st.stop()

# Query section
st.subheader("🤔 Ask a Question")
st.text("Example: What are the experiment results and ablation studies in this research paper?")

user_question = st.text_input("Please enter your question:")

if user_question and retriever:
    # Run the agent with Judgeval tracing
    @judgment.observe(span_type="workflow", name="corrective_rag_workflow")
    def run_corrective_rag(question: str):
        inputs = {
            "keys": {
                "question": question,
            }
        }
        
        # Track workflow steps
        steps = []
        for output in app.stream(inputs):
            for key, value in output.items():
                steps.append((key, value))
                with st.expander(f"Step: {key}", expanded=False):
                    if key == "retrieve":
                        st.write(f"Retrieved {len(value['keys']['documents'])} documents")
                    elif key == "grade_documents":
                        st.write(f"Filtered to {len(value['keys']['documents'])} relevant documents")
                        st.write(f"Web search needed: {value['keys'].get('run_web_search', 'No')}")
                    elif key == "generate":
                        st.write("Generated final answer")
                    else:
                        st.json(value['keys'])
        
        return value['keys'].get('generation', 'No generation produced'), steps
    
    # Run with progress indicator
    with st.spinner("Processing your question..."):
        final_answer, workflow_steps = run_corrective_rag(user_question)
    
    # Display results
    st.subheader("💡 Answer:")
    st.write(final_answer)
    
    # Run comprehensive evaluation
    if st.button("🔍 Run Judgeval Evaluation"):
        with st.spinner("Running comprehensive evaluation..."):
            # Get retrieval context from workflow
            retrieval_docs = []
            for step_name, step_data in workflow_steps:
                if step_name == "generate" and "documents" in step_data["keys"]:
                    retrieval_docs = step_data["keys"]["documents"]
                    break
            
            retrieval_context = [doc.page_content for doc in retrieval_docs[:3]]
            
            # Create evaluation example
            example = Example(
                input=user_question,
                actual_output=final_answer,
                retrieval_context=retrieval_context
            )
            
            # Run multiple scorers
            scorers = [
                FaithfulnessScorer(threshold=0.7),
                AnswerRelevancyScorer(threshold=0.6),
            ]
            
            try:
                results = judgment_client.run_evaluation(
                    examples=[example],
                    scorers=scorers,
                    model="gpt-4o",
                    project_name="corrective_rag_evaluation"
                )
                
                # Display evaluation results
                st.subheader("📊 Evaluation Results:")
                
                for scorer_name, score_data in results.items():
                    if isinstance(score_data, dict) and 'score' in score_data:
                        score = score_data['score']
                        passed = score_data.get('passed', score > 0.5)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label=scorer_name.replace('_', ' ').title(),
                                value=f"{score:.2%}",
                                delta="Pass" if passed else "Fail"
                            )
                        with col2:
                            st.progress(score)
                
                st.success("Evaluation complete! Check Judgment dashboard for detailed traces.")
                
            except Exception as e:
                st.error(f"Evaluation error: {str(e)}")

# Add information about Judgeval integration
with st.sidebar:
    st.divider()
    st.subheader("🔬 Judgeval Integration")
    st.info("""
    This agent is fully integrated with Judgment Labs:
    
    ✅ **Tracing**: All LLM calls and tool usage
    ✅ **Evaluation**: Real-time quality metrics
    ✅ **Multi-Model**: Supports Groq & Google Gemini
    ✅ **Free APIs**: No paid subscriptions required
    
    View traces at: platform.judgment.dev
    """)

if user_question and not retriever:
    st.warning("Please load a document first before asking questions.")