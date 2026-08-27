from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal






load_dotenv()


model = ChatMistralAI(
    model="mistral-small-latest"
)


parser_1 = StrOutputParser()

class output(BaseModel):
    sentiment: Literal['positive','negative']


parser = PydanticOutputParser(pydantic_object=output)


prompt_1 = PromptTemplate(
    template="Analyze this feedback and give a sentiment about this feedback in either positive or negative. {feedback} \n\n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)



classifier_chain = prompt_1 | model | parser



prompt_2 = PromptTemplate(
    template="Generate a appriate response for the customer care team to take the action according to positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt_3 = PromptTemplate(
     template="Generate a appriate response for the customer care team to take the action according to negative feedback \n {feedback}",
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x["sentiment"].sentiment == "positive" , prompt_2 | model | parser_1),
    (lambda x: x["sentiment"].sentiment == "negative",prompt_3|model |parser_1),
    RunnableLambda(lambda x: "The sentiment is not clear")
)


final_chain = (RunnableParallel(
    feedback=RunnableLambda(lambda x: x["feedback"]),
    sentiment= classifier_chain
)) | branch_chain

result = final_chain.invoke({'feedback':"The product looks good, but the performance is disappointing and needs improvement."})

print(result)




