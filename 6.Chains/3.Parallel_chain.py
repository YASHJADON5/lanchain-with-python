from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel



load_dotenv()


model = ChatMistralAI(
    model="mistral-small-latest"
)

prompt_1= PromptTemplate(
    template="Generate short and simple notes from the following text. \n {topic}",
    input_variables=['topic']
)

prompt_2= PromptTemplate(
    template="Generate 5 short question answers in the form of quiz from this topic \n {topic}",
    input_variables=['topic']
)

prompt_3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes','quiz']
)


parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'notes': prompt_1 | model | parser,
    'quiz' : prompt_2 | model | parser
})


merging_chain = prompt_3 | model |parser


chain = parallel_chain | merging_chain


topic="Employment in India is a major driver of economic growth and social development, providing millions of people with opportunities to earn a livelihood, support their families, and improve their standard of living. India has a large and diverse workforce employed across agriculture, manufacturing, construction, information technology, healthcare, education, finance, retail, transportation, hospitality, and many other sectors. In recent years, the services sector has become particularly important, with areas such as IT, software development, digital services, e-commerce, banking, and telecommunications creating new employment opportunities, especially for educated and skilled workers. At the same time, agriculture continues to employ a significant portion of the population, particularly in rural areas. One of the major challenges facing employment in India is the gap between the skills possessed by workers and the skills demanded by employers. Rapid technological development, automation, artificial intelligence, and digitalisation are also changing the nature of jobs, making continuous learning and upskilling increasingly important. Another important issue is the difference between formal and informal employment, as a large number of workers continue to work in the informal sector without stable income, social security, or other employment benefits. The Indian government and private sector have introduced various initiatives focused on skill development, entrepreneurship, manufacturing, startups, digital infrastructure, and job creation to address these challenges. India also has significant potential to create employment through emerging sectors such as renewable energy, electric vehicles, artificial intelligence, semiconductor manufacturing, healthcare, and digital businesses. Overall, improving the quality and availability of employment in India requires continued investment in education, vocational training, infrastructure, entrepreneurship, and industries capable of generating productive and sustainable jobs."



result = chain.invoke({'topic':topic})


chain.get_graph().print_ascii()

print()

print(result)