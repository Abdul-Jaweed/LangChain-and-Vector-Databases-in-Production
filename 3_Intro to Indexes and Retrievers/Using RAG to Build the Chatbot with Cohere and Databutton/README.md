# **Using RAG to Build the Chatbot with Cohere and Databutton**


### **What is Retrieval Augmented Generation (RAG) in AI?**


Retrieval Augmented Generation, or RAG, is an advanced technique in AI that bridges information retrieval and text generation. It is designed to handle intricate and knowledge-intensive tasks by pulling relevant information from external sources and feeding it into a Large Language Model for text generation. When RAG receives an input, it searches for pertinent documents from specified sources (e.g., Wikipedia, company knowledge base, etc.), combines this retrieved data with the input, and then provides a comprehensive output with references. This innovative structure allows RAG to seamlessly integrate new and evolving information without retraining the entire model from scratch. It also enables you to fine-tune the model, enhancing its knowledge domain beyond what it was trained on.

![img](https://images.ctfassets.net/qtqp2awm2ktd/5zg1COwycY10KlN4QtW53a/355a8e99628a50ad127df9aff1c96a58/what_is_retrieval_augmented_generation.webp)


## **Introduction to Retrieval Augmented Generation (RAG) in AI**

Retrieval Augmented Generation (RAG), a new frontier in AI technology, is transforming the digital landscape. With platforms like Cohere & Activeloop, this advanced technology is now easily accessible and customizable, catalyzing a wave of AI-first businesses.

RAG’s impact is considerable. MIT research shows businesses incorporating RAG report up to 50% productivity gains on knowledge-based tasks. By automating mundane tasks, businesses improve resource allocation and employee satisfaction. Notably, Goldman Sachs estimates that such advancements could boost global GDP by 7%.

RAG’s versatility is seen across industries. In customer support, it leads to a 14% productivity increase per hour. In sales, AI-assisted representatives send five times more emails per hour. With the maturation of this technology, these figures will rise even further.

The future of RAG points towards the development of Knowledge Assistants. Acting as intelligent tools for workers, they will retrieve and process corporate data, interact with enterprise systems, and take action on a worker’s behalf. This heralds a new age of AI-driven productivity.

As the third significant revolution in human-computer interfaces, RAG, and LLMs could unlock an estimated $1 trillion in economic value in the U.S. alone. Therefore, businesses and developers must adopt these technologies to remain competitive in the rapidly evolving AI-centric future.

At the end of this article, we cover the Retrieval Augmented Generation History and other fun facts.



## **Build LLM-powered Chatbot with RAG**

**Application Building Steps :**

1. Data Loading

2. Retrieving Data

3. Building Conversation Chain with Memory and Retrieval

4. Building the Chat UI


## **Setting up LangChain & Databutton**


LangChain is a standard interface through which you can interact with a variety of large language models (LLMs). It provides modules you can use to build language model applications. It also provides chains and agents with memory capabilities.

The flowchart below demonstrates the pipeline initiated through LangChain to complete the Conversation Process. The tutorial goes into each of the steps in the pipeline, this visual helps to give you an overview of how the components are working together and in what order.

The design pattern started by thinking about the following:

- What problem am I trying to solve?
- Who is going to benefit from this solution?
- How am I going to get and pre-process my data sources?
- How am I going to store and retrieve my data sources?
- How is the user going to interact with my data sources?

Taking a step back before building a solution can really help to save time and importantly considers your end user.


![img](https://images.ctfassets.net/qtqp2awm2ktd/7umVAvZ2FggWG4Lq0rODvr/1914bcbf455b566ddc2fa48cfffd67b6/flowchart.webp)


# **Application Platform and Required API Keys**

**Databutton:** All-in-one app workspace where we will build and deploy our application. $25 free monthly quota (covers one app a month), community and student plans available.

**Cohere API key:** Generative AI endpoint for embeddings, rerank and chatbot. Get free, rate-limited usage for learning and prototyping. Usage is free until you go into production

**Apify API Key:** Web scraping data for the chatbot to retrieve. $5 free usage (more than enough for website contents)

**Activeloop token:** We will use Deep Lake to store the text scraped from a website. Deep Lake Community version is free to use.


## **Build your Databutton Application**

1. Create a free account with Databutton

2. Create a new app

Once you have signed up for your free Databutton account, you can create a new app in seconds by clicking on ‘New app’

1. Add secrets and packages

![img](https://images.ctfassets.net/qtqp2awm2ktd/2UeQozetB4VqlBdVLD5hrZ/cdd83106ee46f9def3342120a93be87a/secrets_in_databutton.webp)

- To use the API Key in your app, copy the code snippet from the secret, this will look something like this: ‘COHERE_API_KEY = db.secrets.get(name=”COHERE_API_KEY”)’
Add the packages below and click install.

- Add the packages below and click install.

```

langchain
deeplake
openai
cohere
apify-client
tiktoken

```

![img](https://images.ctfassets.net/qtqp2awm2ktd/6Ri1RmNxXDnsTCNbfJCQRr/4542d6b11fbbd5ed37b1129a0b78ab2b/package_installation.webp)

1. Add entire code from the tutorial to either the Jobs section or the Home Page as specified in the steps below.

## **Step 1: Loading the Data with RecursiveCharacterTextSplitter**

In this stage, we are gathering the data needed to provide context to the chatbot. We use ApifyLoader to scrape the content from a specific website. The RecursiveCharacterTextSplitter is then used to split the data into smaller, manageable chunks. Next, we embed the data using CohereEmbeddings which translates the text data into numerical data (vectors) that the chatbot can learn from. Lastly, we load the transformed data into Deep Lake.

The code for this step is located in the ‘Jobs’ section within Databutton because this is a task that only needs to be run once. Once the data is collected and loaded into DeepLake, it can be retrieved by the chatbot.

### **Helper Functions**


- **ApifyWrapper():** Scrapes the content from websites.


```python

from langchain.document_loaders import ApifyDatasetLoader
from langchain.utilities import ApifyWrapper
from langchain.document_loaders.base import Document
import os

os.environ["APIFY_API_TOKEN"] = db.secrets.get("APIFY_API_TOKEN")

apify = ApifyWrapper()
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "ENTER\YOUR\URL\HERE"}]},
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"] if dataset_item["text"] else "No content available",
        metadata={
            "source": dataset_item["url"],
            "title": dataset_item["metadata"]["title"]
        }
    ),
)

docs = loader.load()

```



- **ApifyWrapperRecursiveCharacterTextSplitter():** Splits the scraped content into manageable chunks.


```python

from langchain.text_splitter import RecursiveCharacterTextSplitter

# we split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20, length_function=len
)
docs_split = text_splitter.split_documents(docs)

```


- **CohereEmbeddings():** Translates text data into numerical data.

- **DeepLake():** Stores and retrieves the transformed data.


```python

from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake
import os

os.environ["COHERE_API_KEY"] = db.secrets.get("COHERE_API_KEY")
os.environ["ACTIVELOOP_TOKEN"] = db.secrets.get("APIFY_API_TOKEN")

embeddings = CohereEmbeddings(model = "embed-english-v2.0")

username = "jaweed" # replace with your username from app.activeloop.ai
db_id = 'kb-material'# replace with your database name
DeepLake.force_delete_by_path(f"hub://{username}/{db_id}")

dbs = DeepLake(dataset_path=f"hub://{username}/{db_id}", embedding_function=embeddings)
dbs.add_documents(docs_split)

```


## **Step 2: Retrieve Data**

In this step, we’re setting up the environment to retrieve data from DeepLake using the CohereEmbeddings for transforming numerical data back to text. We’ll then use ContextualCompressionRetriever & CohereRerank to search, rank and retrieve the relevant data.

Add this code to your home page in Databutton

First we set the COHERE_API_KEY and ACTIVELOOP_TOKEN environment variables, using db.secrets.get, allowing us to access the Cohere and ActiveLoop services.

- DeepLake() retrieve data

- CohereEmbeddings()

Following this, we create a DeepLake object, passing in the dataset path to the DeepLake instance, setting it to read-only mode and passing in the embedding function.

Next, we define a data_lake function. Inside this function, we instantiate a CohereEmbeddings object with a specific model, **embed-english-v2.0**.

- ContextualCompressionRetriever() & CohereRerank()

- Reranking (cohere.com)

We then instantiate a **CohereRerank** object with a specific model and number of top items to consider (top_n), and finally create a **ContextualCompressionRetriever** object, passing in the compressor and retriever objects. The data_lake function returns the DeepLake object, the compression retriever, and the retriever.

The data retrieval process is set up by calling the data_lake function and unpacking its return values into dbs, compression_retriever, and retriever.

The Rerank endpoint acts as the last stage reranker of a search flow.


![img](https://images.ctfassets.net/qtqp2awm2ktd/2206m51f7KKehA4D4kvgNJ/6fa8740a4b910c1764947b501e8fc7e9/cohere_rerank_endpoint.webp)


## **A Brief Intro to Cohere’s Rerank Endpoint for Enhanced Search Results**


Within a search process, Cohere’s Rerank endpoint serves as a final step to refine and rank documents in alignment with a user’s search criteria. Businesses can seamlessly integrate it with their existing keyword-based (also called “lexical”) or semantic search mechanisms for initial retrieval. The Rerank endpoint will take over the second phase of refining results.

Cohere’s Rerank & Deep Lake: The Solution to Imprecise Search Outcomes:
This tool is powered by Cohere’s large language model, which determines a relevance score between the user’s query and each of the preliminary search findings. This approach surpasses traditional embedding-based semantic searches, delivering superior outcomes, especially when dealing with intricate or domain-specific search queries.

![img](https://images.ctfassets.net/qtqp2awm2ktd/6mt009JHsJkruV3kRyOMEj/e08f099422ed78a21cc91d33a43115ce/rerank_endpoint_reordering.webp)


This tool is powered by Cohere’s large language model, which determines a relevance score between the user’s query and each of the preliminary search findings. This approach surpasses traditional embedding-based semantic searches, delivering superior outcomes, especially when dealing with intricate or domain-specific search queries.

The DeepLake instance is then turned into a retriever with specific parameters for distance metric, number of items to fetch (fetch_k), use of maximal marginal relevance and the number of results to return (k).



```python

from langchain.vectorstores import DeepLake
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import os

os.environ["COHERE_API_KEY"] = db.secrets.get("COHERE_API_KEY")
os.environ["ACTIVELOOP_TOKEN"] = db.secrets.get("ACTIVELOOP_TOKEN")

@st.cache_resource()
def data_lake():
    embeddings = CohereEmbeddings(model = "embed-english-v2.0")

    dbs = DeepLake(
        dataset_path="hub://elleneal/activeloop-material", 
        read_only=True, 
        embedding_function=embeddings
        )
    retriever = dbs.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    compressor = CohereRerank(
        model = 'rerank-english-v2.0',
        top_n=5
        )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    return dbs, compression_retriever, retriever

dbs, compression_retriever, retriever = data_lake()

```


## **Step 3: Use ConversationBufferWindowMemory to Build Conversation Chain with Memory**


In this step, we will build a memory system for our chatbot using the **ConversationBufferWindowMemory**.

The memory function instantiates a ConversationBufferWindowMemory object with a specific buffer size (k), a key for storing chat history, and parameters for returning messages and output key. The function returns the instantiated memory object.

We then instantiate the memory by calling the memory function.



```python

@st.cache_resource()
def memory():
    memory=ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True, 
        output_key='answer'
        )
    return memory

memory=memory()

```

The chatbot uses the AzureChatOpenAI() function to initiate our LLM Chat model. You can very easily swap this out with other chat models [listed here](https://python.langchain.com/docs/integrations/chat/).


```python

from langchain.chat_models import AzureChatOpenAI

BASE_URL = "<URL>"
API_KEY = db.secrets.get("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = "<deployment_name>"
llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-03-15-preview",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    streaming=True,
    verbose=True,
    temperature=0,
    max_tokens=1500,
    top_p=0.95
)

```


Next, we build the conversation chain using the **ConversationalRetrievalChain**. We use the from_llm class method, passing in the llm, retriever, memory, and several additional parameters. The resulting chain object is stored in the qa variable.


```

qa = ConversationalRetrievalChain.from_llm(
llm=llm,
retriever=compression_retriever,
memory=memory,
verbose=True,
chain_type="stuff",
return_source_documents=True
)

```


## **Step 4: Building the Chat UI**


In this final step, we set up the chat user interface (UI).

We start by creating a button that, when clicked, triggers the clearing of cache and session states, effectively starting a new chat session.

Then, we initialize the chat history if it does not exist and display previous chat messages from the session state.


```python

# Create a button to trigger the clearing of cache and session states
if st.sidebar.button("Start a New Chat Interaction"):
    clear_cache_and_session()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

```


The chat_ui function is used to handle the chat interactions. Inside this function, we accept user input, add the user’s message to the chat history and display it, load the memory variables which include the chat history, and predict and display the chatbot’s response.

The function also displays the top 2 retrieved sources relevant to the response and appends the chatbot’s response to the session state. The chat_ui function is then called, passing in the ConversationalRetrievalChain object.


```python

def chat_ui(qa):
    # Accept user input
    if prompt := st.chat_input(
        "Ask me questions: How can I retrieve data from Deep Lake in Langchain?"
    ):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Load the memory variables, which include the chat history
            memory_variables = memory.load_memory_variables({})

            # Predict the AI's response in the conversation
            with st.spinner("Searching course material"):
                response = capture_and_display_output(
                    qa, ({"question": prompt, "chat_history": memory_variables})
                )

            # Display chat response
            full_response += response["answer"]
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            #Display top 2 retrieved sources
            source = response["source_documents"][0].metadata
            source2 = response["source_documents"][1].metadata
            with st.expander("See Resources"):
                st.write(f"Title: {source['title'].split('·')[0].strip()}")
                st.write(f"Source: {source['source']}")
                st.write(f"Relevance to Query: {source['relevance_score'] * 100}%")
                st.write(f"Title: {source2['title'].split('·')[0].strip()}")
                st.write(f"Source: {source2['source']}")
                st.write(f"Relevance to Query: {source2['relevance_score'] * 100}%")

        # Append message to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

# Run function passing the ConversationalRetrievalChain
chat_ui(qa)

```


## **Verbose Display Code for Streamlit**

```python

import databutton as db
import streamlit as st
import io
import re
import sys
from typing import Any, Callable

def capture_and_display_output(func: Callable[..., Any], args, **kwargs) -> Any:
    # Capture the standard output
    original_stdout = sys.stdout
    sys.stdout = output_catcher = io.StringIO()

    # Run the given function and capture its output
    response = func(args, **kwargs)

    # Reset the standard output to its original value
    sys.stdout = original_stdout

    # Clean the captured output
    output_text = output_catcher.getvalue()
    clean_text = re.sub(r"\x1b[.?[@-~]", "", output_text)

    # Custom CSS for the response box
    st.markdown("""
    <style>
        .response-value {
            border: 2px solid #6c757d;
            border-radius: 5px;
            padding: 20px;
            background-color: #f8f9fa;
            color: #3d3d3d;
            font-size: 20px;  # Change this value to adjust the text size
            font-family: monospace;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create an expander titled "See Verbose"
    with st.expander("See Langchain Thought Process"):
        # Display the cleaned text in Streamlit as code
        st.code(clean_text)

    return response

```


That is all you need to start building your own RAG Chatbot on your own data! I can’t wait to see what you build and how you develop this idea forward.


## **Conclusion: Retrieval Augmented Generation to Power Chatbots & Economy**


In conclusion, Retrieval Augmented Generation (RAG) is not just an emerging AI technology but a transformative force reshaping how businesses operate. With its proven potential to boost productivity, catalyze AI-first businesses, and increase GDP, it’s clear that adopting RAG and Large Language Models is crucial for maintaining a competitive edge in today’s rapidly-evolving digital landscape. The potential of applications like the Educational Chatbot demonstrates how these AI tools can streamline tasks, making operations more efficient and user-friendly. Businesses, developers, and technology enthusiasts need to understand and leverage these advancements. The ongoing development of AI tools like Knowledge Assistants emphasizes the importance of keeping pace with these technological evolutions. As we stand at the brink of the third revolution in human-computer interfaces, we are reminded of the immense value and opportunities RAG and LLMs hold, estimated to unlock $1 trillion in the U.S. economy alone. The future is here, and it’s AI-driven.


## **Retrieval Augmented Generation FAQs**


## **What is Retrieval Augmented Generation (RAG)?**


Retrieval Augmented Generation, or RAG, is a machine learning technique combining the best aspects of retrieval-based and generative language models. This method cleverly integrates the strength of retrieving relevant documents from a large set of data and the creative ability of generative models to construct coherent and diverse responses. Moreover, RAG allows the internal knowledge of the model to be updated efficiently without retraining the entire model.


## **How does Retrieval Augmented Generation work?**

RAG operates in two distinct stages. The first stage involves retrieving relevant documents from a vast vector database like Deep Lake using “dense retrieval.” This process leverages vector representations of the query and documents to identify the most relevant document matches. The second stage is the generation phase, where a sequence-to-sequence model is utilized to create a response, considering not just the input query but also the retrieved documents. The model learns to generate responses based on the context of these retrieved documents.

## **Where is Retrieval Augmented Generation used?**

RAG is useful for complex, knowledge-intensive tasks, such as question-answering and fact verification. It has been used to improve the performance of large language models (LLMs) like GPT-4 or LLama-v2, fine-tuning their performance to be more factual, domain-specific, and diverse.

## **What are Retrieval Augmented Generation advantages?**

RAG combines the benefits of both retrieval-based and generative models. This means it gains from the specificity and factual correctness typical of retrieval-based methods while leveraging the flexibility and creativity inherent in generative models. This combination often results in more accurate, detailed, and contextually appropriate responses.


## **What are the benefits of using Retrieval Augmented Generation**


- RAG offers several advantages over traditional LLMs:

- RAG can easily acquire knowledge from external sources, improving the performance of LLMs in domain-specific tasks.

- RAG reduces hallucination and improves the accuracy of generated content.

- It requires minimal training, only needing to index your knowledge base.

- RAG can utilize multiple sources of knowledge, allowing it to outperform other models.

- It has strong scalability and can handle complex queries.

- It can overcome the context-window limit of LLMs by incorporating data from larger document collections.

- RAG provides explainability by surfacing the sources used to generate text.


## **How to implement Retrieval Augmented Generation?**

Implementation of RAG involves three key components: a knowledge-base index like Deep Lake, a retriever that fetches indexed documents, and an LLM to generate the answers. Libraries like Deep Lake and LangChain have made it easier to implement these complex architectures.


## **What is the historical Ccntext of Retrieval Augmented Generation?**


Retrieval Augmented Generation, as a concept, has its roots in foundational principles of Information Retrieval (IR) and Natural Language Processing (NLP). Retrieving relevant information before generating a response is common in IR. With the rise of neural network-based models in NLP, these approaches started merging, leading to the development of RAG.


## **What are the Complexities Involved in RAG?**

The main challenge with RAG lies in its dual nature - retrieval and generation. The retrieval phase requires an efficient system to sift through vast data. On the other hand, the generation phase needs a model capable of constructing high-quality responses. Both phases require significant computational resources and advanced machine-learning expertise. Using libraries like Deep Lake for efficient data storage and retrieval helps streamline using RAG.

## **What are the Current Challenges with Retrieval Augmented Generation?**

Current challenges with RAG include:

- Handling complex queries that require deep understanding.

-Managing computational resources efficiently.

- Ensuring response relevance and quality.

- Improving these aspects would make RAG even more effective in tasks like chatbots, question-answering systems, or dialogue generation.
