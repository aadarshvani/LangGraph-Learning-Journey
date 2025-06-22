from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import operator
load_dotenv()

##Defining SimpleState in Langgraph
from typing import TypedDict,List,Annotated

class SimpleState(TypedDict):
    count: int
    sum: Annotated[int,operator.add ]
    history: Annotated[List[int], operator.concat]
    

def increment(state:SimpleState)->SimpleState:
    new_count = state['count'] + 1
    return{
        'count' : new_count,
        'sum':  new_count,
        'history':[new_count]
    }

def should_continue(state: SimpleState):
    if state['count'] < 5:
        return 'continue'
    else: 
        return 'stop'
    
graph = StateGraph(SimpleState)

graph.add_node('increment', increment)
graph.add_conditional_edges(
    'increment', 
    should_continue,
    {
        'continue':'increment',
        'stop': END
    }
)
graph.set_entry_point('increment')
app = graph.compile()

state= {
    'count': 0,
    'sum': 0,
    'history' : []
}

result = app.invoke(state)

print(result)
