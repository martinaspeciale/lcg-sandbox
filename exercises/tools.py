from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun 
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool 
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from datetime import datetime
import os

# --- Define input schema ---
class SaveToTxtInput(BaseModel):
    data: str = Field(..., description="The research content to be saved into the research folder research/")
    filename: str = Field(
        default="research/research_output.txt",
        description="Name of the text file to save data into"
    )

# --- Define the actual function ---
def save_to_txt(data: str, filename: str = "research_output.txt"):
    # Ensure the folder exists
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


def print_agent_log(step_name: str, tool_name: str = None, input_data: str = None, output_data: str = None):
    """
    Pretty-print a single agent step (instead of using verbose=True).
    """
    print("\n" + "=" * 60)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP: {step_name}")
    if tool_name:
        print(f"🔧 Tool used: {tool_name}")
    if input_data:
        print(f"➡️  Input: {input_data[:250]}{'...' if len(input_data) > 250 else ''}")
    if output_data:
        print(f"⬅️  Output: {output_data[:250]}{'...' if len(output_data) > 250 else ''}")
    print("=" * 60)
    
# --- Register it as a structured tool ---
save_tool = StructuredTool.from_function(
    func=save_to_txt,
    name="save_text_to_file",
    description="Saves structured research data to a text file",
    args_schema=SaveToTxtInput,
)


search = DuckDuckGoSearchRun() 
search_tool = Tool(
    name="search", # no spaces in the name field 
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)