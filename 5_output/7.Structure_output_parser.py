from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers  import JsonOutputParser
from pydantic import BaseModel,Field
import os

from langchain_core.prompts import PromptTemplate


load_dotenv()


class schema(BaseModel):
    fact: str = Field(description="Store the fact name")
    fact_description: str = Field(description="Store the fact itself")

class Facts(BaseModel):
    facts: list[schema]

parser = JsonOutputParser(pydantic_object=Facts)

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",
    task="text-generation",
    provider="novita",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")

)

model = ChatHuggingFace(llm=llm)

# 1st prompt

template = PromptTemplate(
    template="give 3 facts about the topic {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)



chain = template | model | parser 

result = chain.invoke({'topic':'Cricket'})


print(result)


# structure output parser can not do data validation on its own, 





