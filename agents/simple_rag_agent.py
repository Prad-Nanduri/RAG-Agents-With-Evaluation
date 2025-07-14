import os
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
import tempfile
import requests
from judgeval.tracer import Tracer
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer
import shutil
import requests
import tempfile
import PyPDF2
import pdfplumber
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document

def load_documents(path_or_url: str, is_url: bool = True) -> list:
    """
    1) If is_url & .pdf → download via requests → recurse local
    2) If local .pdf → PyPDF2; if empty → pdfplumber
    3) If local .txt/.md → direct read
    4) Else if is_url non‐PDF → WebBaseLoader
    ALWAYS returns a list (possibly empty).
    """
    try:
        # 1) PDF URL → download
        if is_url and path_or_url.lower().endswith(".pdf"):
            st.info("⬇️ Downloading PDF…")
            resp = requests.get(path_or_url,
                                headers={"User-Agent": "Mozilla/5.0"},
                                timeout=15)
            resp.raise_for_status()
            st.info(f"✅ Downloaded {len(resp.content)//1024} KB")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            return load_documents(tmp_path, is_url=False)

        # 4) Non‐PDF URL → fallback
        if is_url:
            st.info("🌐 Non-PDF URL → WebBaseLoader")
            loader = WebBaseLoader(path_or_url)
            loader.requests_per_second = 1
            docs = loader.load() or []
            st.success(f"✅ WebBaseLoader returned {len(docs)} docs")
            return docs

        # 2 & 3) Local file on disk
        ext = os.path.splitext(path_or_url)[1].lower()

        # 2a) Local PDF
        if ext == ".pdf":
            # PyPDF2 pass
            with open(path_or_url, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = [pg.extract_text() or "" for pg in reader.pages]
            combined = "\n\n".join(pages).strip()

            # fallback to pdfplumber
            if not combined:
                st.info("⚙️ PyPDF2 empty → trying pdfplumber…")
                with pdfplumber.open(path_or_url) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                combined = "\n\n".join(pages).strip()

            if not combined:
                st.error("❌ No text extracted from PDF")
                return []
            st.success(f"✅ Extracted ~{len(combined)//1000} K chars")
            return [Document(page_content=combined,
                             metadata={"source": path_or_url})]

        # 2b) Local txt/md
        if ext in [".txt", ".md"]:
            txt = open(path_or_url, "r", encoding="utf-8", errors="ignore").read().strip()
            if not txt:
                st.error("❌ Text file is empty")
                return []
            return [Document(page_content=txt,
                             metadata={"source": path_or_url})]

        st.error(f"❌ Unsupported file type: {ext}")
        return []

    except Exception as e:
        st.error(f"❌ load_documents error: {e}")
        return []

# Initialize Judgeval with error handling
try:
    judgment = Tracer(project_name="simple_rag_agent")
    judgment_client = JudgmentClient()
    JUDGEVAL_ENABLED = True
except Exception as e:
    JUDGEVAL_ENABLED = False

st.set_page_config(page_title="Simple RAG Agent", page_icon="📚", layout="wide")

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        st.session_state.doc_url = "https://arxiv.org/pdf/2307.09288.pdf"
        st.session_state.model_provider = "groq"
        st.session_state.model_name = "llama-3.1-8b-instant"
        st.session_state.docs = None
        st.session_state.retriever = None
        st.session_state.current_docs_source = None
        st.session_state.vectorstore = None

def setup_sidebar():
    """Setup sidebar for configuration."""
    with st.sidebar:
        st.header("🔧 Configuration")
        
        st.subheader("Model Provider")
        st.session_state.model_provider = st.selectbox(
            "Select LLM Provider",
            ["groq", "google"],
            help="Choose between Groq or Google"
        )
        
        if st.session_state.model_provider == "groq":
            st.session_state.groq_api_key = st.text_input(
                "Groq API Key", 
                value=st.session_state.groq_api_key, 
                type="password"
            )
            st.session_state.model_name = st.selectbox(
                "Groq Model",
                ["llama-3.1-8b-instant", "llama-3.2-1b-preview", "llama-3.2-3b-preview", "mixtral-8x7b-32768"]
            )
        else:
            st.session_state.google_api_key = st.text_input(
                "Google API Key", 
                value=st.session_state.google_api_key, 
                type="password"
            )
            st.session_state.model_name = "gemini-1.5-flash"
        
        st.session_state.doc_url = st.text_input(
            "Default Document URL", 
            value=st.session_state.doc_url
        )
        
        # Display API key status
        if st.session_state.model_provider == "groq":
            if st.session_state.groq_api_key:
                st.success("✅ Groq API key provided")
            else:
                st.error("❌ Please provide Groq API key")
        else:
            if st.session_state.google_api_key:
                st.success("✅ Google API key provided")
            else:
                st.error("❌ Please provide Google API key")
        
        st.session_state.initialized = True
        
        st.divider()
        st.info("""
        **Simple RAG Agent**
        
        This is a baseline RAG implementation without:
        - Document grading
        - Query transformation
        - Web search fallback
        
        Direct retrieval → Generation pipeline.
        """)

initialize_session_state()
setup_sidebar()

@st.cache_resource
def get_embeddings():
    """Get free HuggingFace embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = get_embeddings()

def trace_function(func):
    """Decorator to optionally trace functions."""
    if JUDGEVAL_ENABLED:
        try:
            return judgment.observe(span_type="function", name=func.__name__)(func)
        except:
            return func
    return func

def get_llm():
    """Initialize LLM based on selected provider with detailed error handling."""
    try:
        if st.session_state.model_provider == "groq":
            if not st.session_state.groq_api_key:
                return None
            
            llm = ChatGroq(
                model=st.session_state.model_name,
                api_key=st.session_state.groq_api_key,
                temperature=0,
                max_tokens=1000
            )
            
        else:  # Google
            if not st.session_state.google_api_key:
                return None
                
            llm = ChatGoogleGenerativeAI(
                model=st.session_state.model_name,
                google_api_key=st.session_state.google_api_key,
                temperature=0,
                max_tokens=1000
            )
            
        return llm
            
    except Exception as e:
        st.error(f"❌ Error initializing LLM ({st.session_state.model_provider}): {str(e)}")
        return None
    
#####################################################
    

def create_vector_store(docs):
    """Create vector store from documents."""
    try:
        if not docs:
            return None, 0
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(docs)
        
        if not splits:
            st.error("No chunks created from documents")
            return None, 0
        
        # Create vector store without persistence (simpler)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Update session state
        st.session_state.retriever = retriever
        st.session_state.vectorstore = vectorstore
        st.session_state.docs = docs
        
        return retriever, len(splits)
        
    except Exception as e:
        st.error(f"Vector store error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, 0
    
def simple_rag_query(question: str, retriever):
    """Simple RAG without correction or grading."""
    try:
        # Retrieve documents
        docs = retriever.invoke(question)
        
        if not docs:
            return "No relevant documents found for your question.", []
        
        # Generate answer directly
        llm = get_llm()
        
        if llm is None:
            return "Error: Could not initialize language model. Please check your API key.", docs
        
        prompt = PromptTemplate(
            template="""Based on the following context, please answer the question.
            
Context: {context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        context = "\n\n".join([doc.page_content for doc in docs[:3]])
        
        # Create and execute chain
        chain = prompt | llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        return answer, docs
        
    except Exception as e:
        st.error(f"Error in RAG query: {str(e)}")
        return f"Error processing your question: {str(e)}", []

# Main UI
st.title("📚 Simple RAG Agent")
st.markdown("**Baseline RAG without correction mechanisms**")

# Test LLM availability
if st.session_state.model_provider == "groq" and not st.session_state.groq_api_key:
    st.warning("⚠️ Please provide your Groq API key in the sidebar to use the application.")
elif st.session_state.model_provider == "google" and not st.session_state.google_api_key:
    st.warning("⚠️ Please provide your Google API key in the sidebar to use the application.")

# Document input
st.subheader("📄 Document Input")
input_option = st.radio("Choose input method:", ["URL", "File Upload", "Use Sample Text"])

if input_option == "URL":
    url = st.text_input("Enter document URL:", value=st.session_state.doc_url)
    if url and st.button("Load URL"):
        with st.spinner("Downloading & parsing…"):
            docs = load_documents(url, is_url=True) or []
            st.write(f"🐞 DEBUG: loaded {len(docs)} docs")
            if docs:
                retriever, num_chunks = create_vector_store(docs)
                if retriever:
                    st.session_state.current_docs_source = f"URL: {url}"
                    st.success(f"✅ Loaded 1 doc & created {num_chunks} chunks")
                else:
                    st.error("❌ Failed to build vector store")
            else:
                st.error("❌ Could not load any text from that URL.")

elif input_option == "File Upload":
    uploaded_file = st.file_uploader("Upload a document", type=['pdf','txt','md'])
    if uploaded_file:
        st.info(f"📁 File uploaded: {uploaded_file.name} ({uploaded_file.size//1024:.1f} KB)")
        if st.button("Process File"):
            with st.spinner("Parsing uploaded file…"):
                # save to temp
                suffix = os.path.splitext(uploaded_file.name)[1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded_file.getbuffer()); tmp.close()
                path = tmp.name

                docs = load_documents(path, is_url=False) or []
                os.unlink(path)

                st.write(f"🐞 DEBUG: loaded {len(docs)} docs")
                if docs:
                    retriever, num_chunks = create_vector_store(docs)
                    if retriever:
                        st.session_state.current_docs_source = f"File: {uploaded_file.name}"
                        st.success(f"✅ Parsed upload into {num_chunks} chunks")
                    else:
                        st.error("❌ Failed to build vector store")
                else:
                    st.error("❌ Could not extract any text from the upload")

else:  # Sample text
    sample_text = """
    Artificial Intelligence and Machine Learning Overview
    
    Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn. The field of AI research has been highly successful in developing effective techniques for solving a wide range of problems.
    
    Machine Learning (ML) is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. ML focuses on the development of computer programs that can access data and use it to learn for themselves.
    
    Deep Learning is a subset of ML that uses neural networks with multiple layers. These neural networks attempt to simulate the behavior of the human brain to "learn" from large amounts of data.
    
    Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.
    
    Computer Vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.
    """
    st.info("**📖 Sample Text Content:**")
    st.text_area("Sample Document", sample_text, height=200, disabled=True)
    if st.button("Load Sample Text"):
        docs = [Document(page_content=sample_text, metadata={"source":"sample"})]
        retriever, num_chunks = create_vector_store(docs)
        if retriever:
            st.session_state.current_docs_source = "Sample Text"
            st.success(f"✅ Created {num_chunks} chunks from sample text")
    

# Show current document status
if st.session_state.current_docs_source:
    st.info(f"📄 **Current document:** {st.session_state.current_docs_source}")

# Query section
st.subheader("🤔 Ask a Question")
user_question = st.text_input("Enter your question:")

if user_question:
    if st.session_state.retriever:
        # Check if API key is available before processing
        api_key_available = (
            (st.session_state.model_provider == "groq" and st.session_state.groq_api_key) or
            (st.session_state.model_provider == "google" and st.session_state.google_api_key)
        )
        
        if not api_key_available:
            st.error("❌ Please provide your API key in the sidebar before asking questions.")
        else:
            with st.spinner("Generating answer..."):
                answer, retrieved_docs = simple_rag_query(user_question, st.session_state.retriever)
            
            # Display results
            if not answer.startswith("Error:"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("💡 Answer:")
                    st.write(answer)
                
                with col2:
                    st.subheader("📄 Retrieved Chunks:")
                    for i, doc in enumerate(retrieved_docs[:3]):
                        with st.expander(f"Chunk {i+1}"):
                            st.text(doc.page_content[:200] + "...")
                
                # Single evaluation button with proper logic
                if JUDGEVAL_ENABLED:
                    if st.button("🔍 Run Evaluation", key="run_eval_btn"):
                        with st.spinner("Running evaluation..."):
                            example = Example(
                                input=user_question,
                                actual_output=answer,
                                retrieval_context=[doc.page_content for doc in retrieved_docs[:3]]
                            )
                            
                            try:
                                results = judgment_client.run_evaluation(
                                    examples=[example],
                                    scorers=[
                                        FaithfulnessScorer(threshold=0.7),
                                        AnswerRelevancyScorer(threshold=0.6)
                                    ],
                                    model="gpt-4o",
                                    project_name="simple_rag_evaluation"
                                )
                                
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
                                            
                            except Exception as e:
                                st.error(f"Evaluation error: {str(e)}")
                else:
                    if st.button("🔍 Run Evaluation", key="eval_disabled_btn"):
                        st.warning("Evaluation is disabled due to Judgeval initialization issues.")
            else:
                st.error(answer)
    else:
        st.warning("❌ Please load a document first!")

# Debug information
if st.checkbox("🔧 Show Debug Info"):
    st.subheader("Debug Information")
    st.write(f"**Model Provider:** {st.session_state.model_provider}")
    st.write(f"**Model Name:** {st.session_state.model_name}")
    
    # Show API key status more clearly
    if st.session_state.model_provider == "groq":
        api_key_status = "✅" if st.session_state.groq_api_key else "❌"
        st.write(f"**Groq API Key Available:** {api_key_status}")
    else:
        api_key_status = "✅" if st.session_state.google_api_key else "❌"
        st.write(f"**Google API Key Available:** {api_key_status}")
    
    st.write(f"**Retriever Available:** {'✅' if st.session_state.retriever else '❌'}")
    st.write(f"**Current Document Source:** {st.session_state.current_docs_source or 'None'}")
    st.write(f"**Judgeval Enabled:** {'✅' if JUDGEVAL_ENABLED else '❌'}")
    
    # Test LLM button
    if st.button("🧪 Test LLM Connection", key="test_llm_btn"):
        with st.spinner("Testing LLM..."):
            llm = get_llm()
            if llm:
                try:
                    test_response = llm.invoke("Say 'Hello, I am working!'")
                    st.success(f"✅ LLM Test Successful: {test_response.content}")
                except Exception as e:
                    st.error(f"❌ LLM Test Failed: {str(e)}")
            else:
                st.error("❌ Could not initialize LLM")
    
    # Test PDF loading capability
    if st.button("🧪 Check PDF Dependencies", key="test_pdf_btn"):
        try:
            import PyPDF2
            st.success("✅ PyPDF2 is available")
        except ImportError:
            st.error("❌ PyPDF2 is not installed. Run: pip install PyPDF2")
        
        try:
            from langchain_community.document_loaders import PyPDFLoader
            st.success("✅ PyPDFLoader is available")
        except ImportError:
            st.error("❌ PyPDFLoader is not available")

            # Add this test function to your code
def test_pdf_loading():
    """Test PDF loading capabilities"""
    st.subheader("PDF Loading Test")
    
    uploaded_file = st.file_uploader("Test PDF", type=['pdf'], key="test_pdf")
    
    if uploaded_file and st.button("Test Load", key="test_load"):
        try:
            # Save file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name
            
            st.write(f"File saved to: {temp_path}")
            
            # Test PyPDF2
            try:
                import PyPDF2
                with open(temp_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    st.write(f"PyPDF2: Found {len(reader.pages)} pages")
                    
                    # Try to read first page
                    if reader.pages:
                        text = reader.pages[0].extract_text()
                        st.write(f"First page has {len(text)} characters")
                        st.text(text[:500] + "..." if len(text) > 500 else text)
            except Exception as e:
                st.error(f"PyPDF2 failed: {e}")
            
            # Clean up
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"Test failed: {e}")

# Add this to your main UI (temporarily)
if st.checkbox("Run PDF Debug Test"):
    test_pdf_loading()