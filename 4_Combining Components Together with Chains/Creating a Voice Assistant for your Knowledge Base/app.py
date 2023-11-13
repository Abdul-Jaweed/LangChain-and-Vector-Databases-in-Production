import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
elenven_apikey = os.getenv("ELEVEN_API_KEY")
activeloop_token = os.getenv("ACTIVELOOP_TOKEN")

os.environ['OPENAI_API_KEY']=apikey
os.environ['ELEVEN_API_KEY']=elenven_apikey
os.environ['ACTIVELOOP_TOKEN']=activeloop_token


