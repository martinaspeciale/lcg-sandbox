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
    # 1) wrap the classifier, so it can be plugged into RunnableBranch
    classifier = RunnableLambda(is_simple_math)

    # 2) prepare the LLM chain
    llm = get_chat_llm()
    #  - ChatPromptTemplate.from_messages([...]) builds a two-message prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, concise assistant."),
        ("human", "{question}")
    ])
    # - the pipe | composes a chain: input dict --> formats prompt --> calls LLM --> parses to string
    # - chat_chain.invoke({"question": "What is LangChain?"}) --> returns a string response
    chat_chain = chat_prompt | llm | StrOutputParser()

    # 3) prepare the calculator chain
    # - first lambda: expects input dict with question, calls tiny_calc, then wraps the result as {"result:": "..."}
    # - second lambda:unwraps the dict and returns just the string result 
    # - end result: calc_chain.invoke({"question": "2*(3+4)"}) --> "14"
    calc_chain = (
        RunnableLambda(lambda x: {"result": tiny_calc(x["question"])})
        | RunnableLambda(lambda x: x["result"])
    )

    # 4) the conditional router 
    # - evaluate classifier with the input 
    # - if True --> run calc_chain 
    # - Else    --> run chat_chain (fallback brach, because it comes last without a predicate)
    # n.b. we can add more pairs (predicate, branch) before the default fallback if we want a multi-layer router
    router = RunnableBranch(
        (classifier, calc_chain),
        chat_chain
    )

    # 5) try on two questions 
    # - router.invoke(..) handles the end-to-end decision and execution 
    # - outputs a string in both cases (calculator or LLM), so the calling code is simple
    for q in ["2*(3+4)", "What is LangChain in plain words?"]:
        print(f"Q: {q}")
        print("A:", router.invoke({"question": q}))
        print("-" * 50)


if __name__ == "__main__":
    main()

