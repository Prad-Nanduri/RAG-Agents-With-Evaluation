# run_all.py

import subprocess

def run(script_path, label):
    print(f"\n🔹 Running: {label}")
    result = subprocess.run(["python", script_path])
    if result.returncode != 0:
        print(f"❌ {label} failed with exit code {result.returncode}")
        exit(result.returncode)
    print(f"✅ {label} completed successfully")

if __name__ == "__main__":
    run("agents/simple_rag_agent.py", "Simple RAG Agent")
    run("agents/reasoning_agent.py", "Reasoning Agent")
    run("agents/corrective_rag_judgeval.py", "Corrective RAG Agent")
    run("evaluation/run_comparison.py", "Run Comparison Evaluation")
    run("evaluation/run_evaluation.py", "Run Evaluation")
    print("\n🎉 All agents and evaluations completed successfully!")
