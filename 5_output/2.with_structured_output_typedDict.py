from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated


load_dotenv()

# class Review(TypedDict):
#     summary:"str"
#     sentiment:"str"

# This is a way to explicitly tell the LLM what it should return for each key.

class Review(TypedDict):
    summary: Annotated[str, "Create a short summary of the mobile phone review UNBDER 5 words."]
    sentiment: Annotated[str, "Determine whether the review is positive, negative, or neutral."]


class review(TypedDict):
    reviews:list[Review]

reviews = [
    "The phone has an excellent camera and the photos look very sharp. The battery also lasts all day, but the phone gets slightly warm while gaming.",
    
    "I am disappointed with this mobile. The battery drains quickly and the device becomes slow when I open multiple apps.",
    
    "This phone offers great performance for its price. The display is bright and smooth, and I really like the overall design.",
    
    "The camera quality is average, but the battery life is impressive. The phone feels durable and works well for everyday tasks.",
    
    "I love this mobile. It is fast, the display looks beautiful, and the speakers are loud and clear. The only issue is that it takes a long time to charge."
]


model=ChatMistralAI(
    model="mistral-small-latest"
)


# result = model.invoke(reviews)

structured_model = model.with_structured_output(review)


result = structured_model.invoke(
    "\n".join(reviews)
)


print(result)

