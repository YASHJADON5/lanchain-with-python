from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()



prompt = PromptTemplate(
    template="Generate 5 important facts about {topic}",
    input_variables = ["topic"]
)


model =ChatMistralAI(
    model="mistral-small-latest"
)


parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'skeeing'})

print(result)