from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

# model

embed=MistralAIEmbeddings(
                   model="mistral-embed",
)


# sending data to model

result = embed.embed_query("My name is yash")



print(str(result))