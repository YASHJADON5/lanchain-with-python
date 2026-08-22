from langchain_mistralai import ChatMistralAI


from dotenv import load_dotenv


load_dotenv()


llm = ChatMistralAI(model="mistral-small-latest")


result = llm.invoke("What is the capital of india?")


# print(result) 


print(result.content)