SHELL := /bin/bash
VENV := .venv311

.PHONY: install check run1 run2 run3 clean

install:
	python3.11 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

check:
	. $(VENV)/bin/activate && python -c "from langchain_openai import ChatOpenAI; from langgraph.graph import StateGraph; print('✅ LangChain + LangGraph present')"

run1:
	. $(VENV)/bin/activate && python -m exercises.ex1_hello_langchain

run2:
	. $(VENV)/bin/activate && python -m exercises.ex2_router_with_tool

run3:
	. $(VENV)/bin/activate && python -m exercises.ex3_langgraph_router

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .ruff_cache
