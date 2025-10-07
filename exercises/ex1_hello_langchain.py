from langchain_core.prompts import ChatPromptTemplate
from exercises.llm_provider import get_chat_llm

def main():
    llm = get_chat_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise tutor."),
        ("human", "Explain {topic} in one paragraph for a beginner.")
    ])
    chain = prompt | llm
    print(chain.invoke({"topic": "what an API is"}).content)

if __name__ == "__main__":
    main()

