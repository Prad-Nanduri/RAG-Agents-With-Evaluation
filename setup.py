from setuptools import setup, find_packages

setup(
    name="judgment-corrective-rag",
    version="0.1.0",
    author="Your Name",
    description="Advanced RAG agents with Judgment Labs integration",
    packages=find_packages(),
    install_requires=[
        "judgeval>=0.1.0",
        "langchain>=0.1.0",
        "langchain-groq>=0.0.1",
        "langchain-google-genai>=0.0.5",
        "langgraph>=0.0.20",
        "streamlit>=1.28.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "run-evaluation=run_evaluation:main",
            "corrective-rag=corrective_rag_judgeval:main",
        ],
    },
)