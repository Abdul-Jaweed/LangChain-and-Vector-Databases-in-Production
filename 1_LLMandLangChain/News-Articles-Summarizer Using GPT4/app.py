import os
import json
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

import requests
from newspaper import Article

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)
    
    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()
        
        print(f"Title : {article.title}")
        print(f"Text : {article.text}")

except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")
    



from langchain.schema import (
    HumanMessage
)

# We get the article data from the scraping part

article_title = article.title
article_text = article.text


# prepare template for prompt

template = """You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

Write a summary of the previous article.

"""


prompt = template.format(
    article_title = article.title,
    article_text= article.text
)

messages = [HumanMessage(content=prompt)]



from langchain.chat_models import ChatOpenAI

# load the model
chat = ChatOpenAI(
    openai_api_key=apikey,
    model="gpt-4",
    temperature=0
)

# generate the summary

summary = chat(messages)
print(summary.content)




# =====================================================================

## Additionally 

# If we want a bulleted list, we can modify a prompt and get the result.

# prepare template for prompt

template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.

Here's the article you need to summarize.

==================
Title: {article_title}

{article_text}
==================

Now, provide a summarized version of the article in a bulleted list format.
"""

# format prompt
prompt = template.format(article_title=article.title, article_text=article.text)

# generate summary
summary = chat([HumanMessage(content=prompt)])
print(summary.content)




# If you want to get the summary in French, you can instruct the model to generate the summary in French language. However, please note that GPT-4's main training language is English and while it has a multilingual capability, the quality may vary for languages other than English. Here's how you can modify the prompt.


# prepare template for prompt
template = """You are an advanced AI assistant that summarizes online articles into bulleted lists in French.

Here's the article you need to summarize.

==================
Title: {article_title}

{article_text}
==================

Now, provide a summarized version of the article in a bulleted list format, in French.
"""

# format prompt
prompt = template.format(article_title=article.title, article_text=article.text)

# generate summary
summary = chat([HumanMessage(content=prompt)])
print(summary.content)