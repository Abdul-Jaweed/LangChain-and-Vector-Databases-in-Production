from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain import OpenAI

import os
from dotenv import load_dotenv

api_key = os.get_env("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=api_key,
    temperature=0
)


tools = load_