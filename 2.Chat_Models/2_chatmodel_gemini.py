from langchain_google_genai  import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os


load_dotenv()

model = ChatGoogleGenerativeAI(

    model="gemini-3.7-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=2

)



result = model.invoke("How are you google and what you can do for me?")

print(result.content[0]['text'])