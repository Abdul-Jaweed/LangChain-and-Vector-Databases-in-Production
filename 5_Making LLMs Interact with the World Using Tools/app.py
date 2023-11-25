from langchain.agents import load_tools
from langchain.agents import initialize_agents
from langchain.agents import AgentType
from langchain import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.get_env("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=api_key,
    temperature=0
)

tools = load_tools(
    [
        'serpapi',
        'requests_all'
    ],
    llm=llm,
    serpapi_api_key=SERPAPI_API_KEY
)

agent = initialize_agents(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("tell me ehat is midjourney ?")

