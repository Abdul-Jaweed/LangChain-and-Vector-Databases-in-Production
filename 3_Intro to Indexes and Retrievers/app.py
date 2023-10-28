from langchain.document_loaders import TextLoader

# text to write to a local file
# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai

text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""


# Write text to local file

with open("my_file.txt", "w") as file:
    file.write(text)
    
# use TextLoader to laod text from local file.

loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

print(len(docs_from_file))



# Then, we use CharacterTextSplitter to split the docs into texts.


from langchain.text_splitter import CharacterTextSplitter

# Create a text splitter

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

# split documents into chunks

docs = text_splitter.split_documents(docs_from_file)

print(len(docs))




from langchain.embeddings import OpenAIEmbeddings

import os
from dotenv import loadenv

loadenv()

apikey = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")





# Let’s create an instance of a Deep Lake dataset.

from langchain.vectorstores import DeepLake


deeplake_token = os.getenv("ACTIVELOOP_TOKEN")
os.environ["ACTIVELOOP_TOKEN"] = deeplake_token


my_activeloop_org_id = "abduljaweed"
my_activeloop_dataset_name = "langchain_course_indexers_retrievers"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)


# create retriever from db
retriever = db.as_retriever()

# Once we have the retriver, we can start the question-answering.

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


# Create a retrieval chain

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(
        openai_api_key=apikey,
        model="text-davinci-003",
    ),
    chain_type="stuff",
    retriever=retriever
)


# We can query our document that is an about specific topic that can be found in the documents.

query = "How Google plans to challenge OpenAI?"
response = qa_chain.run(query)
print(response)


