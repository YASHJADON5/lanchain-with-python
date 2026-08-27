import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


class ProductReview(BaseModel):
    summary: str = Field(description="Summary in 3 words")
    rating: int = Field(description="Rating from 1 to 5")


base_parser = PydanticOutputParser(pydantic_object=ProductReview)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
)
model = ChatHuggingFace(llm=llm)


robust_parser = OutputFixingParser.from_llm(parser=base_parser, llm=model)

template = PromptTemplate(
    template="Analyze this review: {review}\n{format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": base_parser.get_format_instructions()}
)

chain = template | model | robust_parser

result = chain.invoke({"review": "Great phone, super fast delivery!"})
print(result)