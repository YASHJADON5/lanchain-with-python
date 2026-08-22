from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

# model

embed=MistralAIEmbeddings(
                   model="mistral-embed",
)

documents = [
    "Lucknow is the capital of Uttar Pradesh",
    "Delhi is the capital of India",
    "Jaipur is the capital of Rajasthan"
]


# sending data to model

result = embed.embed_documents(documents)

# it will genrate a 2d list where every each list inside the 2d list will represnt the each value of the document list

print(str(result))