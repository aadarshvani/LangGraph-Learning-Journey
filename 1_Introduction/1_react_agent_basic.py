from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    model = 'Gemma2-9b-it',
    api_key=api_key
)
search_tool = TavilySearchResults(search_depth = 'basic')
tools = [search_tool]

agent = initialize_agent(tools=tools, llm=llm, agent='zero-shot-react-description', verbose=True)

parser = StrOutputParser()
result = agent.invoke('Tell me about wheather today in Indore')
print(result)
