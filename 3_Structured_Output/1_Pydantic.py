from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(model='llama3-8b-8192')

class Country(BaseModel):
    
    '''Information about a country'''
    name: str = Field(description='Name of the country')
    language : str = Field(description= 'Language of the country')
    capital : str = Field(description='Capital fo the country')
    
structured_llm = llm.with_structured_output(Country)

structured_llm.invoke('Give the name, official language, and capital of India.')