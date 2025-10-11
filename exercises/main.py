from dotenv import load_dotenv 
from pydantic import BaseModel 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os

load_dotenv()
model = os.getenv("GROQ_MODEL")
api_key = os.getenv("GROQ_API_KEY")

# llm = ChatGroq(model=model, api_key=api_key)
# print(llm.invoke("what is the meaning of the universe?").content)

# 1. define a simple Python class which will specify the type of content we want our LLM to generate
# 2. give the LLM a promp telling it to answer the user's question
#.   and as a part of your response, generate it using this model
# --> we'll get output in a format that we can then know and use predictably 

class ResearchResponse(BaseModel):

    # here we specify all of the fields we want as output from the LLM call 
    # (as complicated as we want, we can also have nested objects) so long as all of 
    # our classes (e.g. ResearchResponse) inherit from the BaseModel from Pydantic
    topic: str 
    summary: str 
    source: list[str]
    tools_used : list[str]

llm = ChatGroq(model=model, api_key=api_key)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# parser will now allow us to take the output of the LM and parse it into this model
# so that we can use it like a normal python object inside our code

prompt = ChatPromptTemplate.from_messages(
    [
        # system message: information to the LLM so it knows what it's supposed to be doing
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")

        # chat_history and agent_scratchpad will be automatically filled in by our agent_executor
    ]
    # we are partially going to fill in this prompt by passing the format instructions: 
    # we use our parser to take the pydantic model ResearchResponse, 
    # turn it into a string so that we can give it to the prompt 
    # --> the llm will know when it generates a response, it's got to do it in this format
).partial(format_instructions=parser.get_format_instructions())


agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[],
    verbose=True
)

raw_response = agent_executor.invoke({"query": "What is the capital of France?"})
print(raw_response)