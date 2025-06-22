from langchain_groq import ChatGroq
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain import hub
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

search_tool = TavilySearchResults(search_depth="basic",api_key=os.getenv("TAVILY_API_KEY"))
# search_tool = DuckDuckGoSearchResults()

react_prompt = hub.pull("hwchase17/react")


tools = [get_system_time, search_tool]

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)