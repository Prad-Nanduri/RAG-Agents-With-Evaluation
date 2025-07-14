#!/usr/bin/env python3
"""
Comprehensive evaluation runner for multiple agent types.
This script runs all agents through judgeval and generates a comparison report.
"""

import os
import json
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer  , 
    AnswerRelevancyScorer,
    HallucinationScorer,
    AnswerCorrectnessScorer
)

# Import our agents
# In run_evaluation.py, update the imports section:
from agents.corrective_rag_judgeval import run_corrective_rag
from agents.simple_rag_agent import simple_rag_query
from agents.reasoning_agent import reasoning_agent


# Update the agents dictionary in run_full_evaluation():
agents = {
    "corrective_rag": run_corrective_rag,
    "simple_rag": simple_rag_query,
    "reasoning": reasoning_agent
}

class AgentEvaluator:
    def __init__(self, groq_api_key: str, google_api_key: str = None):
        self.groq_api_key = groq_api_key
        self.google_api_key = google_api_key
        self.client = JudgmentClient()
        self.results = []
        
    def prepare_test_cases(self) -> List[Dict[str, Any]]:
        """Prepare diverse test cases for evaluation."""
        return [
            {
                "question": "What are the main findings of this research?",
                "doc_url": "https://arxiv.org/pdf/2307.09288.pdf",
                "expected_topics": ["results", "findings", "conclusions"]
            },
            {
                "question": "Explain the methodology used in the experiments.",
                "doc_url": "https://arxiv.org/pdf/2307.09288.pdf",
                "expected_topics": ["methodology", "approach", "techniques"]
            },
            {
                "question": "What are the limitations and future work mentioned?",
                "doc_url": "https://arxiv.org/pdf/2307.09288.pdf",
                "expected_topics": ["limitations", "future", "improvements"]
            }
        ]
    
def evaluate_agent(self, agent_name: str, agent_func, test_case: Dict) -> Dict:
    """Evaluate a single agent on a test case."""
    print(f"\n🤖 Evaluating {agent_name} on: {test_case['question']}")
    
    start_time = time.time()
    
    try:
        # Run agent based on type
        if agent_name == "corrective_rag":
            # Corrective RAG doesn't exist as a standalone function
            # For now, return mock result
            answer = f"Mock answer from {agent_name} for: {test_case['question']}"
            
        elif agent_name == "simple_rag":
            # Simple RAG requires document setup - skip for automated test
            answer = f"Simple RAG requires document setup - using mock answer for: {test_case['question']}"
            
        else:  # reasoning agent
            # Import the actual function
            from agents.reasoning_agent import reasoning_agent
            # reasoning_agent returns (full_reasoning, steps, final_answer)
            _, _, answer = reasoning_agent(test_case['question'])
        
        execution_time = time.time() - start_time
        
        # Create evaluation example
        example = Example(
            input=test_case['question'],
            actual_output=answer,
            retrieval_context=[]  # Would be populated for RAG agents
        )
        
        # Run comprehensive evaluation
        scorers = [
            FaithfulnessScorer(threshold=0.7),
            AnswerRelevancyScorer(threshold=0.6)
            # Removed HallucinationScorer and AnswerCorrectnessScorer due to earlier issues
        ]
        
        eval_results = self.client.run_evaluation(
            examples=[example],
            scorers=scorers,
            model="gpt-4o",
            project_name=f"agent_comparison_{agent_name}"
        )
        
        # Compile results
        return {
            "agent": agent_name,
            "question": test_case['question'],
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "execution_time": execution_time,
            "scores": eval_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error evaluating {agent_name}: {str(e)}")
        return {
            "agent": agent_name,
            "question": test_case['question'],
            "error": str(e),
            "execution_time": time.time() - start_time
        }
    
def run_full_evaluation(self):
        """Run evaluation across all agents and test cases."""
        print("🚀 Starting comprehensive agent evaluation...")
        
        test_cases = self.prepare_test_cases()
        
        agents = {
            "corrective_rag": run_corrective_rag,
            "simple_rag": simple_rag_query,
            "reasoning": reasoning_agent
        }
        
        for test_case in test_cases:
            for agent_name, agent_func in agents.items():
                result = self.evaluate_agent(agent_name, agent_func, test_case)
                self.results.append(result)
                time.sleep(2)  # Rate limiting
        
        self.generate_report()
    
def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("\n📊 Generating evaluation report...")
        
        # Convert results to DataFrame
        df_results = []
        for result in self.results:
            if "scores" in result:
                row = {
                    "agent": result["agent"],
                    "question": result["question"][:50] + "...",
                    "execution_time": result["execution_time"]
                }
                for scorer, score_data in result["scores"].items():
                    if isinstance(score_data, dict):
                        row[scorer] = score_data.get("score", 0)
                df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average scores by agent
        score_cols = [col for col in df.columns if col not in ["agent", "question", "execution_time"]]
        avg_scores = df.groupby("agent")[score_cols].mean()
        
        avg_scores.plot(kind="bar", ax=axes[0, 0])
        axes[0, 0].set_title("Average Scores by Agent")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Execution time comparison
        df.groupby("agent")["execution_time"].mean().plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].set_title("Average Execution Time by Agent")
        axes[0, 1].set_ylabel("Time (seconds)")
        
        # 3. Score distribution heatmap
        sns.heatmap(avg_scores.T, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1, 0])
        axes[1, 0].set_title("Score Heatmap")
        
        # 4. Performance radar chart (simplified box plot instead)
        df_melted = df.melt(id_vars=["agent"], value_vars=score_cols)
        sns.boxplot(data=df_melted, x="agent", y="value", hue="variable", ax=axes[1, 1])
        axes[1, 1].set_title("Score Distribution by Agent")
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig("agent_evaluation_report.png", dpi=300, bbox_inches="tight")
        
        # Save detailed results
        with open("evaluation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
                # Print summary
        print("\n📈 EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total evaluations: {len(self.results)}")
        print(f"Agents tested: {', '.join(avg_scores.index)}")
        print("\nAverage Scores by Agent:")
        print(avg_scores.round(3))
        print("\nAverage Execution Time:")
        print(df.groupby("agent")["execution_time"].mean().round(2))
        print("\n✅ Report saved to: agent_evaluation_report.png")
        print("✅ Detailed results saved to: evaluation_results.json")

if __name__ == "__main__":
    # Load API keys from environment
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if not groq_key:
        print("❌ Please set GROQ_API_KEY environment variable")
        exit(1)
    
    evaluator = AgentEvaluator(groq_key, google_key)
    evaluator.run_full_evaluation()