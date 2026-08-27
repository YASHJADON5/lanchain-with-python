from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



load_dotenv()



model = ChatMistralAI(
    model="mistral-small-latest"
)



prompt_1 = PromptTemplate(
    template="Give the a detailed report on the topic. \n {topic}",
    input_variable=["topic"]
)

prompt_2 = PromptTemplate(
    template="Give the a 20 words summary of the report on the topic. \n {topic}",
    input_variable=["topic"]
)

parser= StrOutputParser()

chain= prompt_1 | model | parser | prompt_2 | model | parser

result = chain.invoke({'topic':'langchain'})


chain.get_graph().print_ascii()

print(result)