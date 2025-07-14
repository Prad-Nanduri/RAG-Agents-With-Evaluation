"""Test suite for RAG agents with Judgeval integration."""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.corrective_rag_judgeval import retrieve, grade_documents, generate
from agents.simple_rag_agent import simple_rag_query
from agents.reasoning_agent import reasoning_agent
# Set test environment
os.environ["JUDGMENT_API_KEY"] = "test_key"
os.environ["JUDGMENT_ORG_ID"] = "test_org"
os.environ["GROQ_API_KEY"] = "test_groq_key"

from agents.corrective_rag_judgeval import run_corrective_rag
from agents.simple_rag_agent import simple_rag_query  
from agents.reasoning_agent import reasoning_agent

# Update the agents dictionary in run_full_evaluation():
agents = {
    "corrective_rag": run_corrective_rag,
    "simple_rag": simple_rag_query,
    "reasoning": reasoning_agent
}


class TestCorrectiveRAG:
    """Test cases for corrective RAG agent."""
    
    @pytest.fixture
    def mock_state(self):
        """Create mock state for testing."""
        return {
            "keys": {
                "question": "What is the capital of France?",
                "documents": [
                    MagicMock(page_content="Paris is the capital of France."),
                    MagicMock(page_content="London is the capital of UK.")
                ]
            }
        }
    
    def test_retrieve_with_no_retriever(self):
        """Test retrieve function when retriever is None."""
        state = {"keys": {"question": "test question"}}
        result = retrieve(state)
        
        assert result["keys"]["documents"] == []
        assert result["keys"]["question"] == "test question"
    
    @patch('corrective_rag_judgeval.get_llm')
    def test_grade_documents(self, mock_llm, mock_state):
        """Test document grading functionality."""
        # Mock LLM response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = '{"score": "yes"}'
        mock_llm.return_value = mock_chain
        
        result = grade_documents(mock_state)
        
        assert "documents" in result["keys"]
        assert "run_web_search" in result["keys"]
    
    @patch('corrective_rag_judgeval.get_llm')
    @patch('corrective_rag_judgeval.judgment.async_evaluate')
    def test_generate_with_evaluation(self, mock_eval, mock_llm, mock_state):
        """Test generation with Judgeval evaluation."""
        # Mock LLM response
        mock_llm.return_value.invoke.return_value = "Paris is the capital of France."
        
        result = generate(mock_state)
        
        assert "generation" in result["keys"]
        assert mock_eval.called
        
        # Check evaluation parameters
        eval_call = mock_eval.call_args
        assert eval_call.kwargs["input"] == mock_state["keys"]["question"]


class TestSimpleRAG:
    """Test cases for simple RAG agent."""
    
    @patch('simple_rag_agent.wrap')
    def test_simple_rag_query(self, mock_wrap):
        """Test simple RAG query execution."""
        # Mock retriever and LLM
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = [
            MagicMock(page_content="Test content")
        ]
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Test answer"
        mock_wrap.return_value = mock_llm
        
        result = simple_rag_query(
            "Test question",
            mock_retriever,
            "test_api_key"
        )
        
        assert result == "Test answer"
        assert mock_retriever.get_relevant_documents.called


class TestReasoningAgent:
    """Test cases for reasoning agent."""
    
    @patch('reasoning_agent.wrap')
    @patch('reasoning_agent.judgment.async_evaluate')
    def test_chain_of_thought_reasoning(self, mock_eval, mock_wrap):
        """Test chain-of-thought reasoning."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Step 1: Analyze\nStep 2: Reason\nAnswer: Test"
        mock_wrap.return_value = mock_llm
        
        result = reasoning_agent(
            "Test question",
            "groq",
            "test_api_key"
        )
        
        assert "Step 1:" in result
        assert mock_eval.called


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual services."""
    
    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="Requires actual GROQ API key"
    )
    def test_end_to_end_corrective_rag(self):
        """Test full corrective RAG pipeline."""
        # This would test with actual services
        pass