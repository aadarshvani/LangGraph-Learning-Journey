from pydantic import BaseModel, Field
from typing import Literal, Optional, Annotated, List, Dict
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq

llm = ChatGroq(
    model = 'Gemma2-9b-it',
    api_key= os.getenv('GROQ_API_KEY')
)

class Country(BaseModel):
    
    '''Information about a country'''
    name : str = Field(description='Name of the Country')
    language: str = Field(description='National Language of the country')
    capital : str = Field(description='Name of the capital city of the country')
    

structured_llm = llm.with_structured_output(Country)
