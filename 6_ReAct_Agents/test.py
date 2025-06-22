from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchRun

# Use Tavily if you have API key
# search = TavilySearchResults()

# Or use DuckDuckGo (no key needed)
search = DuckDuckGoSearchRun()

result = search.invoke("When was the last ISRO rocket launch?")
print(result)
