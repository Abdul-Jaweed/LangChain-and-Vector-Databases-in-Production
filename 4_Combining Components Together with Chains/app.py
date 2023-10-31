## LLMChain

# Several methods are available for utilizing a chain, each yielding a distinct output format. The example in this section is creating a bot that can suggest a replacement word based on context. The code snippet below demonstrates the utilization of the GPT-3 model through the OpenAI API. It generates a prompt using the PromptTemplate from LangChain, and finally, the LLMChain class ties all the components. Also, It is important to set the OPENAI_API_KEY environment variable with your API credentials from OpenAI. Remember to install the required packages with the following command: pip install langchain==0.0.208 deeplake openai tiktoken.


from langchain import PromptTemplate, OpenAI, LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

prompt_template = "What is a word to replace the following: {word}?"

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

# llm_chain("artificial")


input_list = [
    {"word": "artificial"},
    {"word": "intelligence"},
    {"word": "robot"}
]

# llm_chain.apply(input_list)

# llm_chain.generate(input_list)



prompt_template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=prompt_template, input_variables=["word", "context"]))

llm_chain.predict(word="fan", context="object")
# or llm_chain.run(word="fan", context="object")


# 💡 We can directly pass a prompt as a string to a Chain and initialize it using the .from_string() function as follows.

LLMChain.from_string(llm=llm, template=template).