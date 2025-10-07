# 🧠 LangChain + LangGraph Sandbox

A hands-on playground for learning how [LangChain](https://python.langchain.com) and [LangGraph](https://www.langchain.com/langgraph) work together to build intelligent, tool-using pipelines and graph-based agent workflows.

---

## 📂 Project layout

```text
lcg-sandbox/
│
├── exercises/
│   ├── __init__.py
│   ├── llm_provider.py             ← LLM switcher (Groq / OpenAI)
│   ├── ex1_hello_langchain.py      ← minimal LangChain prompt
│   ├── ex2_router_with_tool.py     ← branching logic + calculator tool
│   └── ex3_langgraph_router.py     ← LangGraph state machine router
│
├── requirements.txt
├── Makefile                        ← reproducible commands
└── README.md
```

---

## ⚙️ Setup

### 1️⃣ Clone & enter the repo
```bash
git clone https://github.com/yourname/lcg-sandbox.git
cd lcg-sandbox
```

### 2️⃣ Create a Python 3.11 virtual environment
```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

### 4️⃣ Create your `.env` file

#### ➤ Using **Groq** (recommended for free usage)
```bash
PROVIDER=groq
GROQ_API_KEY=sk_your_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Get a free key at [https://console.groq.com/keys](https://console.groq.com/keys).

#### ➤ Using **OpenAI**
```bash
PROVIDER=openai
OPENAI_API_KEY=sk_your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 5️⃣ Load environment variables
```bash
export $(cat .env | xargs)
```

---

## 🚀 Run the exercises

| Command | Description |
|----------|-------------|
| `make check` | Verify LangChain + LangGraph installation |
| `make run1` | Basic prompt with the selected LLM |
| `make run2` | Router that switches between a calculator tool and the LLM |
| `make run3` | Same logic implemented as a LangGraph state graph |
| `make clean` | Remove venv and caches |

Example:
```bash
make run1
```

Output:
```
An API ... is a set of rules that allows software systems to communicate...
```

---

## 🧩 How it works

- **`llm_provider.py`** — abstracts the choice of model (Groq or OpenAI)  
- **`ex1_hello_langchain.py`** — builds a simple `ChatPromptTemplate → ChatModel` chain  
- **`ex2_router_with_tool.py`** — introduces branching logic using `RunnableBranch` and a custom calculator tool  
- **`ex3_langgraph_router.py`** — implements the same routing via a **LangGraph**, showing how to design LLM “agents” as graphs of state transitions

---

## 💡 Next ideas

- Add structured outputs (Pydantic models)  
- Add memory or retrieval (PDF or website ingestion)  
- Visualize your LangGraph (`graph.get_graph().draw()`)  
- Deploy a lightweight FastAPI endpoint wrapping these chains


