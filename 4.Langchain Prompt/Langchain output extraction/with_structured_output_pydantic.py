"""
=============================================================================
Jira Ticket: AI-043 - Build PrepSync Question Generator API
=============================================================================
Context:
We are building the backend generation engine for PrepSync, an AI-powered 
placement prep app. We need a secure, validated pipeline that generates 
structured interview questions to be served to a mobile frontend via FastAPI.

Task:
Write a Python script using model.with_structured_output() and a Pydantic 
BaseModel schema to generate a clean, strictly validated interview question.

Schema Requirements (InterviewQuestion BaseModel):
1. 'question_text': A string containing the actual interview question.
2. 'difficulty': Constrain to EXACTLY one of: 'Easy', 'Medium', or 'Hard'.
3. 'topic_tags': A list of strings (e.g., ["System Design", "Databases"]).
4. 'hints': A list of exactly two strings (hints if the student gets stuck).
5. 'ideal_answer_summary': A brief string explaining the core concept to pass.

Test Payload (Prompt Variables):
Construct a ChatPromptTemplate that accepts these variables:
- topic = "Database Indexing"
- target_company = "Datamato"

Validation:
Execute the chain passing the test variables. 
Because the output is a Pydantic object (not a standard dictionary), 
use print(result.model_dump()) to print it cleanly in your terminal 
to verify the strict validation worked.
=============================================================================
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Literal

load_dotenv()
model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class InterviewQuestion(BaseModel):
    question_text:str= Field(description='Return a string containing the actual interview question.')
    difficulty: Literal['Easy', 'Medium', 'Hard']= Field(description='Return difficulty level')
    topic_tags: list[str]= Field(description='return which topic question is?')
    hints: list[str]=Field(min_length=2,max_length=2,description='return hints for the student if gets stuck')
    ideal_answer_summary: str= Field(description='A brief string explaining the core concept the student should hit to "pass" the question.')

structure_output= model.with_structured_output(InterviewQuestion)

prompt=ChatPromptTemplate.from_messages([
    ('system','You are a technical interview question generator. Output precise, well-structured technical questions, hints, and answers based on the provided topic and target company.'),
    ('human', "Generate a technical interview scenario.\nTopic: {topic}\nTarget Company: {target_company}")])

chain= prompt | structure_output

question_topic = "DSA"
target_company_specific = "TCS"

result= chain.invoke({
    'topic':question_topic,
    'target_company':target_company_specific
})

print(result)
print('----------------------------------------------------')
print(result.question_text)
print('----------------------------------------------------')

print(result.difficulty)
print('----------------------------------------------------')

print(result.topic_tags)
print('----------------------------------------------------')

print(result.hints)
print('----------------------------------------------------')

print(result.ideal_answer_summary)
