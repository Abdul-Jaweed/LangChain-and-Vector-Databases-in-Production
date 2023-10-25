## Role Prompting
# Role prompting involves asking the LLM to assume a specific role or identity before performing a given task, such as acting as a copywriter. This can help guide the model's response by providing a context or perspective for the task. To work with role prompts, you could iteratively:

#  1. Specify the role in your prompt, e.g., "As a copywriter, create some attention-grabbing taglines for AWS services."
#  2. Use the prompt to generate an output from an LLM.
#  3. Analyze the generated response and, if necessary, refine the prompt for better results.



from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    llm=llm,
    model="text-davinci-003",
    temperature=0
)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""


prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template
)

# Input dat for the prompt 

input_data = {
    "theme":"intersteller travel",
    "year":"3030"
}

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

response = chain.run(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)