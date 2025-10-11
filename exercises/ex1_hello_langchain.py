from langchain_core.prompts import ChatPromptTemplate
from exercises.llm_provider import get_chat_llm

def main():
    topic = input("\n\n Input a topic to be explained: ")
    llm = get_chat_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise tutor."),
        ("human", "Explain {topic} in one paragraph for a beginner.")
    ])
    chain = prompt | llm
    response = chain.invoke({"topic":topic})
    print("\n\n 🧠 Answer from the model: ", response.content, "\n\n")

if __name__ == "__main__":
    main()

