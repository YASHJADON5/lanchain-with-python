from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers  import  StrOutputParser
import os

from langchain_core.prompts import PromptTemplate


load_dotenv()


llm= HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")

)


model = ChatHuggingFace(llm=llm)


# 1st prompt

template_1 = PromptTemplate.from_template(
    "Write a detailed report on {topic}"
)

template_2 = PromptTemplate.from_template(
    "Write a summary of this content in exactly 10 words. \n\n {topic}"
)

# formatted_template = template_1.invoke({'topic':'aws'})

# result_1= model.invoke(formatted_template)

# # 2nd prompt

# formatted_template_2=template_2.invoke({'topic':f'{result_1.content}'})

# result_2 = model.invoke(formatted_template_2)

# print(result_2.content)


parser = StrOutputParser()

chain = template_1 | model | parser | template_2 | model |parser

result= chain.invoke({'topic':'black hole'})

print(result)



2