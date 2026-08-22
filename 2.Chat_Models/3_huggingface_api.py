from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()


# setting up llm

llm = HuggingFaceEndpoint(
    
    repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",
    task="text-generation",
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    provider="novita"
   


)

# setting up model with llm


model = ChatHuggingFace(llm=llm)


result = model.invoke("Hello what is your name and specification")


# print(result)


print(result.content)

