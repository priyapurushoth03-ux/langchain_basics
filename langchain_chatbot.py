from langchain_core.prompts import ChatPromptTemplate

general_prompt = ChatPromptTemplate.from_template(
    "You are a college assistant. Answer: {question}"
)

admission_prompt = ChatPromptTemplate.from_template(
    "You are an admission assistant. Answer: {question}"
)

exam_prompt = ChatPromptTemplate.from_template(
    "You are an exam assistant. Answer: {question}"
)

library_prompt = ChatPromptTemplate.from_template(
    "You are a library assistant. Answer: {question}"
)

placement_prompt = ChatPromptTemplate.from_template(
    "You are a placement assistant. Answer: {question}"
)

print("Five Prompt Templates Created Successfully!")
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import os

llm = ChatOllama(model="gemma3", temperature=0.2)

chain = general_prompt | llm | StrOutputParser()

questions = [
    "What courses are available?",
    "How to apply for admission?",
    "What are admission requirements?",
    "What documents are required?",
    "What is the college fee?",
    "When will exams start?",
    "How to download hall ticket?",
    "What is the passing mark?",
    "How to check results?",
    "What is the exam timetable?",
    "What are library timings?",
    "How to borrow a book?",
    "How many books can I borrow?",
    "Is library card required?",
    "Where is the library?",
    "Does college provide placement training?",
    "Which companies visit college?",
    "How to register for placement?",
    "Are internships available?",
    "What skills are needed for placement?"
]

responses = []

for q in questions:
    answer = chain.invoke({"question": q})
    print("\nQuestion:", q)
    print("Answer:", answer)
    responses.append(answer)

os.makedirs("dataset", exist_ok=True)

pd.DataFrame({
    "Question": questions
}).to_csv("dataset/user_queries.csv", index=False)

pd.DataFrame({
    "Response": responses
}).to_csv("dataset/chatbot_responses.csv", index=False)

print("20 Questions Completed!")
import pandas as pd
import os

os.makedirs("dataset", exist_ok=True)

# 1. Prompts CSV
prompts = {
    "Prompt_Name": [
        "General",
        "Admission",
        "Exam",
        "Library",
        "Placement"
    ],
    "Prompt_Template": [
        "You are a college assistant. Answer: {question}",
        "You are an admission assistant. Answer: {question}",
        "You are an exam assistant. Answer: {question}",
        "You are a library assistant. Answer: {question}",
        "You are a placement assistant. Answer: {question}"
    ]
}

pd.DataFrame(prompts).to_csv(
    "dataset/prompts.csv", index=False
)

# 2. Chain Results CSV
chain_results = {
    "Chain_Type": [
        "Summarization",
        "Question Answering"
    ],
    "Input": [
        "College provides library and placement facilities.",
        "What are the library timings?"
    ],
    "Output": [
        "College provides library and placement facilities.",
        "Library timings are 9 AM to 5 PM."
    ]
}

pd.DataFrame(chain_results).to_csv(
    "dataset/chain_results.csv", index=False
)

print("All four CSV files are ready!")
