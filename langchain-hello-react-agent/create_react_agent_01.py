import os
import datetime
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from pydantic import BaseModel

load_dotenv()
checkpointer = InMemorySaver()


class WeatherResponse(BaseModel):
    conditions: str


def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    elif city.lower() == "guangzhou":
        tz_identifier = "Asia/Shanghai"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

model = ChatOpenAI(
    model="deepseek-reasoner",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

llm = ChatDeepSeek(
    model="deepseek-reasoner",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)



agent = create_react_agent(
    # model="anthropic:claude-3-7-sonnet-latest",  
    model=llm,
    tools=[get_weather, get_current_time],  
    checkpointer=checkpointer,
    prompt="You are a helpful assistant",
    # prompt="Never answer questions about the weather.",
    # response_format=WeatherResponse,
)

agent = create_agent(
    llm,
    tools=[get_weather, get_current_time],
    checkpointer=checkpointer,
    system_prompt="You are a helpful assistant",
)

# Run the agent
config = {"configurable": {"thread_id": "1"}}
gz_wt = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in guangzhou"}]},
    config
)

gz_dt = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the time in guangzhou"}]},
    config
)

print(gz_wt)
print(gz_dt)
