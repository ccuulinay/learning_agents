import os
import datetime
import json
import logging
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
# from google.adk.agents import Agent
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
import serpapi

# Helpers
def serpapi_google_search(query: str) -> dict:
    """Returns the serpapi google search result given query

    Args:
        query (str): The query from agent

    Returns:
        dict: status and result or error msg.
    """
    if query.strip() == "":
        return {
            "status": "error",
            "error_message": (
                f"Sorry, the query is empty, will not perform searching."
            ),
        }
    
    client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))

    try:
        results = client.search({
            'engine': 'google',
            'q': query,
        })
    except serpapi.HTTPError as e:
        _status = "error"
        if e.status_code == 401: # Invalid API key
            logging.exception(e) # "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key"
            _error_message = e.error
        elif e.status_code == 400: # Missing required parameter
            logging.exception(e)
            _error_message = "Missing required parameter"
        elif e.status_code == 429: # Exceeds the hourly throughput limit OR account run out of searches
            logging.exception(e)
            _error_message = "Exceeds the hourly throughput limit OR account run out of searches"
        else:
            logging.exception(e)
            _error_message = str(e)
        return {
             "status": "error",
            "error_message": _error_message
        }
    except serpapi.TimeoutError as e:
        # Handle timeout
        logging.error(f"The request timed out: {e}")
        _error_message = "timeout"
        return {
             "status": "error",
            "error_message": _error_message
        }
    except Exception as e:
        logging.exception(e)
        _error_message = str(e)
        return {
            "status": "error",
            "error_message": _error_message
        }
    if "organic_results" in results:
        _r = json.dumps(results["organic_results"])
    else:
        _r = json.dumps(dict(results))
    
    return {
        "status": "success", "query_result": _r
    }

    

load_dotenv()

# Define litellm
ds_llm = LiteLlm(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    # retry_options=retry_config,
)

root_agent = Agent(
    name="simple_search_assistant",
    # model="gemini-2.0-flash",
    model=ds_llm,
    description="A simple agent that can answer general questions.",
    instruction="You are a simple search assistant. Use Google Search for current info or if unsure.",
    tools=[serpapi_google_search]
)

# runner = InMemoryRunner(agent=root_agent)