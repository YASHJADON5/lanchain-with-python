from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers  import  JsonOutputParser, StrOutputParser
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

parser = JsonOutputParser()

# 1st prompt

template = PromptTemplate(
    template="Generate a person's data in which include name, age and other features  \n\n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt= template.invoke({})

# string_parser=StrOutputParser()

# print(prompt)



chain = template | model | parser 

result = chain.invoke({})


print(result)

# print(result.name)


# json output parser does not enforce the schema



