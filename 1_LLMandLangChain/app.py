# Tracking Token Usage

from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0,
    n=2,
    best_of=2
)

# text = "What would be a good company name for a company that makes colorful socks?"

# print(llm(text))


with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)



