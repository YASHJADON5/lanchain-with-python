from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import date


load_dotenv()

model =ChatMistralAI(
    model="mistral-small-latest"
)

prompt_template = ChatPromptTemplate.from_messages([
     ("system", "Your name is jenny and you are a female! Today's date is {date}"),
     MessagesPlaceholder(variable_name="messages"),
     ("human", "{human_message}")
])

messages =[]

while True:

    user_input = input("Enter your prompt: ")

    if user_input == "exit":
        break
    
    # messages.append({
    #     "role":"human",
    #     "content":user_input
    # })

    messages.append(HumanMessage(content=user_input))

    formatted_prompt= prompt_template.invoke({"date":date.today(),"messages":messages,"human_message":user_input})
    # print(date.today())

    # print(formatted_prompt)
    result = model.stream(formatted_prompt)

    full_response=""

    for chunk in result:
        print(chunk.content,end="",flush=True)
        full_response+=chunk.content

    print()
    # messages.append({
    #     "role":"assistant",
    #     "content":result.content
    # })

    messages.append(AIMessage(content=full_response))


print()
print("Thanks for using python chatbot")



