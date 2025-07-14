# Makefile for Judgment Labs RAG Agents

.PHONY: help setup test run-eval clean docker lint format

help:
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies and setup environment"
	@echo "  make test       - Run test suite"
	@echo "  make run-eval   - Run full evaluation suite"
	@echo "  make dashboard  - Launch evaluation dashboard"
	@echo "  make clean      - Clean temporary files"
	@echo "  make docker     - Build Docker image"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code"

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .
	cp .env.example .env
	@echo "✅ Setup complete! Edit .env with your API keys"

test:
	. venv/bin/activate && pytest tests/ -v --cov=.

run-eval:
	. venv/bin/activate && python run_evaluation.py

dashboard:
	. venv/bin/activate && streamlit run dashboard.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf chroma_db

docker:
	docker build -t judgment-rag-agents .

lint:
	. venv/bin/activate && ruff check .
	. venv/bin/activate && black --check .

format:
	. venv/bin/activate && ruff check --fix .
	. venv/bin/activate && black .