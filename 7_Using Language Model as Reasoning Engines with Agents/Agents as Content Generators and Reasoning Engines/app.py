import os
os.environ["OPENAI_API_KEY"] = "<YOUR-OPENAI-API-KEY>"
os.environ["GOOGLE_API_KEY"] = "<YOUR-GOOGLE-SEARCH-API-KEY>"
os.environ["GOOGLE_CSE_ID"] = "<YOUR-CUSTOM-SEARCH-ENGINE-ID>"


from langchain.agents import load_agent, initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI



# Loading the language model to control the agent
llm = OpenAI(model="text-davinci-003", temperature=0)



# Loading some tools to use. The llm-math tool uses an LLM, so we pass that in.

tools = load_agent(["google-search", "llm-math"], llm=llm)


# Initializing an agent with the tools, the language model, and the type of agent we want to use.

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


# Testing the agent

query = "What's the result of 1000 plus the number of goals scored in the soccer world cup in 2018?"

response = agent.run(query)
print(response)