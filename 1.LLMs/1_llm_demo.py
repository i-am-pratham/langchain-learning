from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize the Base LLM (Notice the class name changed)
llm = GoogleGenerativeAI(model='gemini-2.5-flash')

# Base LLMs are designed for "text completion" rather than conversation.
# So, we give it an incomplete sentence instead of a question.
prompt = "Which is the best anime?"

# 2. Invoke the model
result = llm.invoke(prompt)

# 3. Print the result directly! 
# No .content needed because Base LLMs return raw strings.
print("Prompt:", prompt)
print("Result:", result)