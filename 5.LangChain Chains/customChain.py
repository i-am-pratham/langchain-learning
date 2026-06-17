# Scenario: You're a backend dev at a healthcare startup. Build a Patient Report Summarizer that:

# Takes a patient_id as input
# Custom step — looks up patient data from a Python dict (simulating a DB call), returns name, age, and symptoms
# Passes patient data to LLM → generates a short 3-line clinical summary a doctor can read quickly
# Custom step — post-processes the output by adding an urgent flag if the word "chest" or "breathing" appears in the summary: "\n\n🚨 URGENT: Requires immediate attention", otherwise add "\n\n✅ Routine checkup"

from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

from dotenv import load_dotenv

load_dotenv()
# model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.1 
)

model = ChatHuggingFace(llm=llm)




def get_patient(input):  #input= {"id":"P001"}
    patients = {
        "P001": {"name": "Rahul Sharma", "age": 45, "symptoms": "chest pain, dizziness, shortness of breath"},
        "P002": {"name": "Priya Mehta",  "age": 30, "symptoms": "mild fever, headache, body ache"},
        "P003": {"name": "Arun Patel",   "age": 60, "symptoms": "difficulty breathing, fatigue, swollen legs"}
}
    data = patients[input["id"]]
    return {"name": data["name"], "age": data["age"], "symptoms": data["symptoms"]}

    
prompt = ChatPromptTemplate([
    ("system", "You are a clinical assistant. Write a short 3-line summary for a doctor."),
    ("human", "Patient: {name}, Age: {age}, Symptoms: {symptoms}")
])

# Custom Step 2 — post-processing
def add_flag(text):
    if "chest" in text.lower() or "breathing" in text.lower():
        return text + "\n\n🚨 URGENT: Requires immediate attention"
    else:
        return text + "\n\n✅ Routine checkup"

# Full chain
chain = RunnableLambda(get_patient) | prompt | model | parser | RunnableLambda(add_flag)

# Test all 3
for pid in ["P001", "P002", "P003"]:
    print(f"\n--- {pid} ---")
    print(chain.invoke({"id": pid}))