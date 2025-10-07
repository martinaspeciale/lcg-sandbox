from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from exercises.llm_provider import get_chat_llm


# a simple calculator tool
def tiny_calc(expr: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expr) <= allowed:
        return "Only + - * / and parentheses are allowed."
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


# simple classifier to decide whether to use calculator or LLM
def is_simple_math(input: Dict[str, Any]) -> bool:
    q = input["question"].strip().lower()
    return all(ch in "0123456789+-*/(). " for ch in q) and any(op in q for op in "+-*/")


def main():
    classifier = RunnableLambda(is_simple_math)

    llm = get_chat_llm()
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, concise assistant."),
        ("human", "{question}")
    ])
    chat_chain = chat_prompt | llm | StrOutputParser()

    calc_chain = (
        RunnableLambda(lambda x: {"result": tiny_calc(x["question"])})
        | RunnableLambda(lambda x: x["result"])
    )

    router = RunnableBranch(
        (classifier, calc_chain),
        chat_chain
    )

    for q in ["2*(3+4)", "What is LangChain in plain words?"]:
        print(f"Q: {q}")
        print("A:", router.invoke({"question": q}))
        print("-" * 50)


if __name__ == "__main__":
    main()

