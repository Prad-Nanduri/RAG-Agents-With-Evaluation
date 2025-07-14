#!/usr/bin/env python3
"""
Compare all three agent types on the same questions.
"""

import os
import sys
import time
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.corrective_rag_judgeval import run_corrective_rag
from agents.simple_rag_agent import simple_rag_query
from agents.reasoning_agent import reasoning_agent

from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer

class AgentComparison:
    def __init__(self):
        self.client = JudgmentClient()
        self.results = []
        
    def prepare_test_questions(self):
        """Test questions for comparison."""
        return [
            {
                "question": "What is machine learning?",
                "type": "definition",
                "requires_context": False
            },
            {
                "question": "Explain the difference between supervised and unsupervised learning.",
                "type": "comparison",
                "requires_context": False
            },
            {
                "question": "What are the main findings in this document?",
                "type": "document_specific",
                "requires_context": True
            }
        ]
    
    def run_comparison(self):
        """Run all agents on test questions."""
        questions = self.prepare_test_questions()
        
        print("🚀 Starting agent comparison...")
        print("=" * 50)
        
        for q in questions:
            print(f"\n📝 Question: {q['question']}")
            print(f"   Type: {q['type']}")
            
            # Skip document-specific questions for reasoning agent
            if q['requires_context']:
                print("   ⚠️  Skipping reasoning agent (requires document)")
                
            # Run each agent
            results = self.evaluate_question(q)
            self.results.extend(results)
            
            # Brief pause between questions
            time.sleep(2)
        
        self.generate_comparison_report()
    
    def evaluate_question(self, question_data):
        """Evaluate a single question across all applicable agents."""
        results = []
        question = question_data['question']
        
        # Test each agent
        agents = [
            ("Corrective RAG", "corrective_rag"),
            ("Simple RAG", "simple_rag"),
            ("Pure Reasoning", "reasoning")
        ]
        
        for agent_name, agent_type in agents:
            if question_data['requires_context'] and agent_type == "reasoning":
                continue
                
            print(f"\n   Testing {agent_name}...")
            
            try:
                start_time = time.time()
                
                # Get answer based on agent type
                if agent_type == "corrective_rag":
                    # Would need to set up retriever first
                    answer = "Sample answer from corrective RAG"
                elif agent_type == "simple_rag":
                    # Would need to set up retriever first
                    answer = "Sample answer from simple RAG"
                else:  # reasoning
                    _, _, answer = reasoning_agent(question)
                
                execution_time = time.time() - start_time
                
                # Evaluate
                example = Example(
                    input=question,
                    actual_output=answer
                )
                
                eval_results = self.client.run_evaluation(
                    examples=[example],
                    scorers=[AnswerRelevancyScorer(threshold=0.7)],
                    model="gpt-4o",
                    project_name=f"agent_comparison_{agent_type}"
                )
                
                results.append({
                    "agent": agent_name,
                    "question": question,
                    "question_type": question_data['type'],
                    "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "execution_time": execution_time,
                    "relevancy_score": eval_results.get("AnswerRelevancyScorer", {}).get("score", 0),
                    "timestamp": datetime.now().isoformat()
                })
                
                print(f"      ✅ Score: {results[-1]['relevancy_score']:.2%}")
                print(f"      ⏱️  Time: {execution_time:.2f}s")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                results.append({
                    "agent": agent_name,
                    "question": question,
                    "question_type": question_data['type'],
                    "error": str(e),
                    "execution_time": 0,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def generate_comparison_report(self):
        """Generate visual comparison report."""
        print("\n📊 Generating comparison report...")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save raw results
        with open("evaluation/agent_comparison_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Average scores by agent
        if 'relevancy_score' in df.columns:
            avg_scores = df.groupby('agent')['relevancy_score'].mean()
            avg_scores.plot(kind='bar', ax=axes[0, 0], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[0, 0].set_title('Average Relevancy Score by Agent')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_ylim(0, 1)
        
        # 2. Execution time comparison
        avg_time = df.groupby('agent')['execution_time'].mean()
        avg_time.plot(kind='bar', ax=axes[0, 1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('Average Execution Time by Agent')
        axes[0, 1].set_ylabel('Time (seconds)')
        
        # 3. Score by question type
        if 'relevancy_score' in df.columns:
            pivot_scores = df.pivot_table(
                values='relevancy_score', 
                index='question_type', 
                columns='agent', 
                aggfunc='mean'
            )
            pivot_scores.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Performance by Question Type')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend(title='Agent')
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary_text = "Summary Statistics\n" + "="*30 + "\n"
        
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent]
            summary_text += f"\n{agent}:\n"
            summary_text += f"  Avg Score: {agent_df['relevancy_score'].mean():.2%}\n" if 'relevancy_score' in agent_df else ""
            summary_text += f"  Avg Time: {agent_df['execution_time'].mean():.2f}s\n"
            summary_text += f"  Success Rate: {(~agent_df['execution_time'].isna()).mean():.2%}\n"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('evaluation/agent_comparison_report.png', dpi=300, bbox_inches='tight')
        
        print("\n✅ Report saved to: evaluation/agent_comparison_report.png")
        print("✅ Raw data saved to: evaluation/agent_comparison_results.json")
        
        # Print summary to console
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        print(summary_text)

if __name__ == "__main__":
    comparison = AgentComparison()
    comparison.run_comparison()