from typing import Optional
from typing_extensions import Annotated, TypedDict
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

llm = OllamaLLM(model='llama3')

class Joke(TypedDict):
    """Joke to tell the user"""
    setup: Annotated[str, 'The setup of the joke']
    punchline: Annotated[str, 'The punchline of the joke']
    rating: Annotated[Optional[int], 'How funny the joke is, on a scale from 1 to 10']

# Wrap the LLM with structured output
structured_llm = llm.with_structured_output(Joke)

# Invoke it with a simple prompt
response = structured_llm.invoke("Tell me a joke about cats")
print(response)
