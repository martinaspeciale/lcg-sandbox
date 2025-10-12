from dotenv import load_dotenv 
from pydantic import BaseModel 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from exercises.tools import search_tool, wiki_tool, save_tool, save_to_txt, print_agent_log
import json 

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
            You are a research assistant helping generate a research summary.
            You can use ONLY these tools when needed: Wikipedia and search.

            Your goal is to collect enough information to write a concise and accurate summary.
            Once you have enough information to answer, STOP using tools and produce your final answer.

            The final answer must be a valid JSON object matching this format:
            {format_instructions}

            Respond with ONLY the JSON object — no explanations, no markdown, and no extra text.
            If you already have sufficient info, do not call any more tools.
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


tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate"
)

query = input("What can I help you research?")

print_agent_log("Starting agent: ", input_data=query)

raw_response = agent_executor.invoke({"query": query})
# print(raw_response)

print_agent_log("\nAgent finished: ", output_data=str(raw_response))

try:
    output_text = raw_response.get("output", "")

    if '"properties":' in output_text:
        try:
            data = json.loads(output_text)
            if isinstance(data, dict) and "properties" in data:
                data = data["properties"]
            structured_response = ResearchResponse(**data)
        except Exception as e:
            raise ValueError(f"Failed to fix nested 'properties' JSON: {e}")
    else:
        structured_response = parser.parse(output_text)

    for key, value in structured_response:
        print(f"* {key} : {value}")

    save_to_txt(
        data=json.dumps(structured_response.dict(), indent=2, ensure_ascii=False),
        filename="research/full_research_output.txt"
    )

except Exception as e:
    print("Error parsing response:", e, "\nRaw response:", raw_response)

# we want to add the ability to add various tools 
# Tools are things that the LLM/agent can use that we can either write ourself or we can bring in from things like the Langchain Community Hub

