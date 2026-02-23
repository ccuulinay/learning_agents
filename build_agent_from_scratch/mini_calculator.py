# Source doc: https://nikhilpentapalli.medium.com/building-ai-agents-from-scratch-no-frameworks-7e75b11396d8


import re
import os
import openai
from dotenv import load_dotenv
from serpapi import GoogleSearch
import json

class MiniCalculatorAgent:
    def __init__(self):
        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_API_URL"))
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    def calculate_expression(self, expression):
        try:
            return eval(expression)
        except:
            return None

    def search_knowledge(self, query):
        params = {
            "api_key": self.serpapi_api_key,
            "engine": "google",
            "q": query,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        if "organic_results" in results:
            return json.dumps(results["organic_results"])
        return "Could not find any relevant information."


    def query_openai_tool_selector(self, user_input):
        messages = [
            {
                "role": "system",
                "content": (
                    "You're an AI assistant. Based on the user's message, "
                    "decide which tool to use: 'calculator', 'knowledge_search', or 'none'. "
                    "Respond ONLY with a JSON object like this:\n"
                    "{ \"tool\": \"calculator\", \"input\": \"5 * (4 + 3)\" }\n"
                    "or\n"
                    "{ \"tool\": \"knowledge_search\", \"input\": \"who is the founder of openai\" }\n"
                    "or\n"
                    "{ \"tool\": \"none\", \"input\": \"Hello!\" }"
                )
            },
            {"role": "user", "content": user_input}
        ]
        response = self.client.chat.completions.create(
            model=os.getenv("DEEPSEEK_API_MODEL"),
            messages=messages
        )
        try:
            print(response.choices[0].message.content)
            tool_call = json.loads(response.choices[0].message.content)
            return tool_call.get("tool"), tool_call.get("input")
        except:
            return "none", user_input

    def query_openai(self, user_input):
        tool, tool_input = self.query_openai_tool_selector(user_input)
        if tool == "calculator":
            result = self.calculate_expression(tool_input)
            return f"The result is: {result}" if result else "I couldn't compute that."
        elif tool == "knowledge_search":
            result = self.search_knowledge(tool_input)
            return result if result else "I couldn't find relevant information."
        else:
            response = self.client.chat.completions.create(
                model=os.getenv("DEEPSEEK_API_MODEL"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.choices[0].message.content

    def run(self):
        print("Welcome! Ask me anything. Type 'quit' to stop.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                print("Agent: Goodbye!")
                break
            print("Agent:", self.query_openai(user_input))


if __name__ == "__main__":
    agent = MiniCalculatorAgent()
    agent.run()