from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    model='gemma2-9b-it',
    api_key=groq_api_key,
    temperature= 0.9
)

search_tool = TavilySearchResults(search_depth = 'basic')
tools = [search_tool]

parser = StrOutputParser()

agent = initialize_agent(tools = tools, 
                         llm = llm, 
                         agent='zero-shot-react-description',
                        verbose=True)


result = agent.invoke('what does OneTabAI actually does? and how well do they pay to AI/Ml Engineer ?')

print(result)