"""
=============================================================================
Jira Ticket: AI-042 - Build Automated Support Ticket Classifier
=============================================================================
Context:
Customer support agents currently waste hours reading incoming emails to 
determine routing and urgency. We need to automate this triage process 
using LangChain, Gemini, and strict structured output.

Task:
Write a Python script that takes a raw customer email and uses 
model.with_structured_output() along with a TypedDict schema to extract 
specific metadata for automatic routing.

Schema Requirements (TicketMetadata):
1. 'department': Constrain to EXACTLY one of: 'Billing', 'Technical', or 'General'.
2. 'urgency_level': Integer from 1 to 5 (1=low priority, 5=critical failure).
3. 'customer_frustration': Boolean (True if angry/all-caps/threatening, else False).
4. 'mentioned_product_features': Optional list of strings (specific app features 
   mentioned, e.g., 'dashboard', 'export tool'). Empty list if none mentioned.

Test Payload:
"URGENT!! Your new update completely broke my dashboard. I tried to use the 
PDF export tool this morning for a massive client presentation and it just 
keeps spinning! I have been a loyal customer for 3 years but if this isn't 
fixed in the next hour I am canceling my subscription and moving to your 
competitor. FIX IT NOW."

Validation:
Execute the chain against the test payload and print the resulting dictionary.
Ensure all types and constraints (especially the Literal department) are respected.
=============================================================================
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated, Literal,List, Optional

from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class TicketMetadata(TypedDict):
    department: Annotated[Literal['Billing', 'Technical', 'General'],'return the department as either Billing, Technica General']
    urgency_level:Annotated[int, 'return Integer from 1 to 5 (1=low priority, 5=critical failure)']
    customer_frustration:Annotated[bool,'(True if angry/all-caps/threatening, else False)']
    mentioned_product_features:Annotated[Optional[list[str]]," (specific app features mentioned, e.g., 'dashboard', 'export tool'). Empty list if none mentioned"]

structure_output= model.with_structured_output(TicketMetadata)

prompt=ChatPromptTemplate([
    ('system','You are an AI customer experience analyst. Extract data precisely.'),
    ('human','{chat_query}')
])

chain= prompt| structure_output

customer_review="URGENT!! Your new update completely broke my dashboard. I tried to use the PDF export tool this morning for a massive client presentation and it just keeps spinning! I have been a loyal customer for 3 years but if this isn't fixed in the next hour I am canceling my subscription and moving to your competitor. FIX IT NOW."

result= chain.invoke({'chat_query':customer_review})
print(result)