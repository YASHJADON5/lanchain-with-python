from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embed = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
)

documents = [
    "Lucknow is the capital of Uttar Pradesh",
    "Delhi is the capital of India",
    "Jaipur is the capital of Rajasthan"
]

result = embed.embed_query("My name is yash")

result_for_doc= embed.embed_documents(documents)

print(result)

print(result_for_doc)