from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from exercises.llm_provider import get_chat_llm


# --- small calculator tool ---
def tiny_calc(expr: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expr) <= allowed:
        return "Only + - * / and parentheses are allowed."
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


# --- define the graph state ---
class State(TypedDict):
    question: str
    route: Literal["math", "chat"] | None
    answer: str | None


# --- classify node ---
def classify_node(state: State) -> State:
    q = state["question"].strip().lower()
    is_math = (
        all(ch in "0123456789+-*/(). " for ch in q)
        and any(op in q for op in "+-*/")
    )
    return {**state, "route": "math" if is_math else "chat"}


# --- calculator node ---
def calc_node(state: State) -> State:
    result = tiny_calc(state["question"])
    return {**state, "answer": result}


# --- chat node ---
llm = get_chat_llm()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise tutor."),
    ("human", "{question}")
])

def chat_node(state: State) -> State:
    msg = prompt.invoke({"question": state["question"]})
    out = llm.invoke(msg)
    return {**state, "answer": out.content}


# --- routing logic ---
def route_decider(state: State) -> str:
    return state["route"] or "chat"


def main():
    graph = StateGraph(State)

    # add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("calc", calc_node)
    graph.add_node("chat", chat_node)

    # define flow
    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_decider,
        {"math": "calc", "chat": "chat"},
    )
    graph.add_edge("calc", END)
    graph.add_edge("chat", END)

    # compile and run
    app = graph.compile()

    for q in ["12/(3+1)", "Explain vector databases simply."]:
        final_state = app.invoke({"question": q, "route": None, "answer": None})
        print(f"Q: {q}")
        print(f"Route: {final_state['route']}")
        print(f"A: {final_state['answer']}")
        print("-" * 50)


if __name__ == "__main__":
    main()

