from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




load_dotenv()



cricketers_doc = [
    "Virat Kohli is an Indian batsman known for his aggressive batting and consistency.",
    "Rohit Sharma is an Indian batsman famous for his elegant stroke play and record-breaking ODI double centuries.",
    "MS Dhoni is an Indian wicketkeeper-batsman known for his leadership, finishing ability, and calm nature.",
    "Sachin Tendulkar is an Indian batting legend widely regarded as one of cricket's greatest batsmen.",
    "Jasprit Bumrah is an Indian fast bowler known for his unusual action, accuracy, and deadly yorkers.",
    "AB de Villiers is a South African batsman famous for his 360-degree stroke play and versatility.",
    "Ben Stokes is an English all-rounder known for his powerful batting, bowling, and performances under pressure.",
    "Babar Azam is a Pakistani batsman known for his technically sound batting and consistency across formats.",
    "Kane Williamson is a New Zealand batsman admired for his calm temperament and technically correct batting.",
    "Pat Cummins is an Australian fast bowler known for his pace, accuracy, leadership, and effectiveness in Test cricket."
]


embeddings = MistralAIEmbeddings(
    model = "mistral-embed"
)


query = "Tell me about Sachin Tendulkar"


doc_embeddings = embeddings.embed_documents(cricketers_doc)

query_embedding = embeddings.embed_query(query)

similarity_scores = cosine_similarity([query_embedding],doc_embeddings)[0]





# print(similarity_scores)


index, score = sorted(list(enumerate(similarity_scores)),key=lambda x:x[1])[-1]

print(index)

print(query)
print(cricketers_doc[index])
print("Similarity Score is", score)










