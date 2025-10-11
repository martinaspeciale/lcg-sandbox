import os
from dotenv import load_dotenv

load_dotenv()

def get_chat_llm():
    provider = os.getenv("PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to your .env")
        return ChatGroq(model=model, api_key=api_key)
    else:
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # OPENAI_API_KEY is read from env by the client
        return ChatOpenAI(model=model)

