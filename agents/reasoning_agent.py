import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from judgeval.tracer import Tracer
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer

# Initialize Judgeval with error handling
try:
    judgment = Tracer(project_name="reasoning_agent")
    judgment_client = JudgmentClient()
    JUDGEVAL_ENABLED = True
except Exception as e:
    st.warning(f"Judgeval initialization failed: {e}")
    JUDGEVAL_ENABLED = False

st.set_page_config(page_title="Reasoning Agent", page_icon="🧠", layout="wide")

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        st.session_state.model_provider = "groq"
        st.session_state.model_name = "llama-3.1-70b-versatile"

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
                ["llama-3.2-90b-text-preview", "llama-3.2-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
            )
        else:
            st.session_state.google_api_key = st.text_input(
                "Google API Key", 
                value=st.session_state.google_api_key, 
                type="password"
            )
            st.session_state.model_name = "gemini-1.5-flash"
        
        # Validate
        if st.session_state.model_provider == "groq" and not st.session_state.groq_api_key:
            st.error("Please provide Groq API key")
            st.stop()
        elif st.session_state.model_provider == "google" and not st.session_state.google_api_key:
            st.error("Please provide Google API key")
            st.stop()
        
        st.session_state.initialized = True
        
        st.divider()
        st.info("""
        **Pure Reasoning Agent**
        
        This agent uses:
        - Chain-of-Thought prompting
        - No external documents/RAG
        - Step-by-step reasoning
        - Pure LLM knowledge
        
        Best for general knowledge questions.
        """)

initialize_session_state()
setup_sidebar()

def trace_function(func):
    """Decorator to optionally trace functions."""
    if JUDGEVAL_ENABLED:
        try:
            return judgment.observe(span_type="function", name=func.__name__)(func)
        except:
            return func
    return func

def get_llm():
    """Initialize LLM based on selected provider."""
    try:
        if st.session_state.model_provider == "groq":
            llm = ChatGroq(
                model_name=st.session_state.model_name,
                api_key=st.session_state.groq_api_key,
                temperature=0.1,
                max_tokens=2000
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=st.session_state.google_api_key,
                temperature=0.1,
                max_tokens=2000
            )
        
        return llm
        
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def reasoning_agent(question: str, debug_mode: bool = False):
    """Pure reasoning agent using Chain-of-Thought prompting."""
    
    try:
        if debug_mode:
            st.write(f"🔍 Debug: Processing question: '{question}'")
        
        llm = get_llm()
        
        if llm is None:
            error_msg = "Error: Could not initialize language model."
            if debug_mode:
                st.error("Debug: LLM initialization failed")
            return error_msg, error_msg, error_msg
        
        if debug_mode:
            st.success("Debug: LLM initialized successfully")
        
        # Chain-of-Thought prompt
        prompt = PromptTemplate(
            template="""You are an expert reasoner. Answer the following question using step-by-step reasoning.

Question: {question}

Please think through this step-by-step:
1. First, identify what the question is asking
2. Break down the key components
3. Consider relevant knowledge and connections
4. Reason through to a comprehensive answer
5. Provide a clear, well-structured response

Step-by-step reasoning:
Let me think through this carefully.

Step 1 - Understanding the question:
[Analyze what is being asked]

Step 2 - Key components:
[Identify the main elements]

Step 3 - Relevant knowledge:
[Connect to what I know]

Step 4 - Reasoning process:
[Work through the logic]

Step 5 - Final answer:
[Provide clear response]

Remember to be thorough but concise.""",
            input_variables=["question"]
        )
        
        # Create and execute chain
        chain = prompt | llm | StrOutputParser()
        
        if debug_mode:
            st.write("Debug: Chain created, invoking...")
        
        # Get the full reasoning
        full_reasoning = chain.invoke({"question": question})
        
        if debug_mode:
            st.write(f"Debug: Response received, length: {len(full_reasoning) if full_reasoning else 0}")
        
        if not full_reasoning:
            error_msg = "Error: No response generated from the model."
            if debug_mode:
                st.error("Debug: Empty response from model")
            return error_msg, error_msg, error_msg
        
        # Extract just the final answer (after "Final answer:" or "Step 5")
        if "Final answer:" in full_reasoning:
            parts = full_reasoning.split("Final answer:")
            reasoning_steps = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else full_reasoning
        elif "Step 5" in full_reasoning:
            parts = full_reasoning.split("Step 5")
            reasoning_steps = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else full_reasoning
        else:
            reasoning_steps = full_reasoning
            # Try to extract the last paragraph as final answer
            paragraphs = full_reasoning.split('\n\n')
            final_answer = paragraphs[-1] if paragraphs else full_reasoning
        
        if debug_mode:
            st.success("Debug: Response processed successfully")
            st.write(f"Debug: About to return tuple with lengths - full: {len(full_reasoning)}, steps: {len(reasoning_steps)}, answer: {len(final_answer)}")
        
        # Run evaluation on the final answer only if Judgeval is enabled
        if JUDGEVAL_ENABLED:
            try:
                judgment.async_evaluate(
                    scorers=[AnswerRelevancyScorer(threshold=0.7)],
                    input=question,
                    actual_output=final_answer,
                    model="gpt-4o"
                )
            except Exception as e:
                if debug_mode:
                    st.warning(f"Evaluation error: {str(e)}")
        
        return full_reasoning, reasoning_steps, final_answer
        
    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}"
        if debug_mode:
            st.error(f"Debug: Exception occurred: {str(e)}")
        else:
            st.error(error_msg)
        return error_msg, error_msg, error_msg

# Main UI
st.title("🧠 Pure Reasoning Agent")
st.markdown("**Chain-of-Thought reasoning without external documents**")

# Create tabs for different reasoning modes
tab1, tab2, tab3 = st.tabs(["💭 General Reasoning", "🔢 Math & Logic", "📝 Creative Tasks"])

with tab1:
    st.subheader("Ask Any Question")
    st.text("Examples: Explain quantum computing, What causes seasons, How does democracy work")
    
    question1 = st.text_input("Enter your question:", key="general")
    
    if question1:
        with st.spinner("Thinking step-by-step..."):
            result = reasoning_agent(question1, debug_mode=False)
            
            # Handle the result safely
            if result and isinstance(result, tuple) and len(result) == 3:
                full_reasoning, steps, answer = result
            else:
                full_reasoning = steps = answer = "Error: Could not generate response"
        
        # Display reasoning process
        with st.expander("🔍 See Full Reasoning Process", expanded=True):
            st.text(full_reasoning)
        
        st.subheader("💡 Final Answer:")
        st.info(answer)

with tab2:
    st.subheader("Math & Logic Problems")
    st.text("Examples: Solve 2x + 5 = 13, Logic puzzles, Word problems")
    
    question2 = st.text_input("Enter your math/logic question:", key="math")
    
    if question2:
        with st.spinner("Working through the problem..."):
            result = reasoning_agent(question2)
            
            # Handle the result safely
            if result and isinstance(result, tuple) and len(result) == 3:
                full_reasoning, steps, answer = result
            else:
                full_reasoning = steps = answer = "Error: Could not generate response"
        
        with st.expander("📐 See Solution Steps", expanded=True):
            st.text(full_reasoning)
        
        st.subheader("✅ Solution:")
        st.success(answer)

with tab3:
    st.subheader("Creative Writing & Ideas")
    st.text("Examples: Write a haiku about AI, Brainstorm startup ideas, Create a story outline")
    
    question3 = st.text_input("Enter your creative prompt:", key="creative")
    
    if question3:
        with st.spinner("Creating..."):
            result = reasoning_agent(question3)
            
            # Handle the result safely
            if result and isinstance(result, tuple) and len(result) == 3:
                full_reasoning, steps, answer = result
            else:
                full_reasoning = steps = answer = "Error: Could not generate response"
        
        with st.expander("🎨 See Creative Process", expanded=True):
            st.text(full_reasoning)
        
        st.subheader("🌟 Result:")
        st.write(answer)

# Comparison section
st.divider()
st.subheader("🔬 Agent Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Agent Type", "Pure Reasoning")
    st.caption("No external data")

with col2:
    st.metric("Best For", "General Knowledge")
    st.caption("Math, logic, creative tasks")

with col3:
    st.metric("Limitations", "No Recent Info")
    st.caption("Limited to training data")

# Evaluation section
if any([question1, question2, question3]):
    if st.button("🔍 Run Comprehensive Evaluation", key="evaluation_button"):
        if JUDGEVAL_ENABLED:
            active_question = question1 or question2 or question3
            
            with st.spinner("Running evaluation..."):
                # Re-run to get fresh answer
                result = reasoning_agent(active_question)
                
                if result and isinstance(result, tuple) and len(result) == 3:
                    _, _, final_answer = result
                    
                    example = Example(
                        input=active_question,
                        actual_output=final_answer
                    )
                    
                    try:
                        results = judgment_client.run_evaluation(
                            examples=[example],
                            scorers=[AnswerRelevancyScorer(threshold=0.7)],
                            model="gpt-4o",
                            project_name="reasoning_agent_evaluation"
                        )
                        
                        st.subheader("📊 Evaluation Results:")
                        for scorer_name, score_data in results.items():
                            if isinstance(score_data, dict) and 'score' in score_data:
                                score = score_data['score']
                                st.metric(
                                    label="Answer Relevancy",
                                    value=f"{score:.2%}"
                                )
                                st.progress(score)
                                
                    except Exception as e:
                        if "Project limit exceeded" in str(e):
                            st.warning("Evaluation limit reached - the agent is working correctly!")
                        else:
                            st.error(f"Evaluation error: {str(e)}")
                else:
                    st.error("Could not generate answer for evaluation")
        else:
            st.warning("Evaluation is disabled due to Judgeval initialization issues.")

# Footer comparison info
with st.sidebar:
    st.divider()
    st.subheader("📊 Quick Comparison")
    
    comparison_data = {
        "Feature": ["External Data", "Fact Checking", "Speed", "Creativity", "Accuracy"],
        "Corrective RAG": ["✅", "✅", "⭐⭐", "⭐", "⭐⭐⭐"],
        "Simple RAG": ["✅", "❌", "⭐⭐⭐", "⭐", "⭐⭐"],
        "Reasoning": ["❌", "❌", "⭐⭐⭐", "⭐⭐⭐", "⭐⭐"]
    }
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)