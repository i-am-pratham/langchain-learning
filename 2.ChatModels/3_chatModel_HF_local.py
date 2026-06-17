import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# 1. Load the variables from your .env file
load_dotenv()

# 2. Map the token so the local transformers library can authenticate the download
os.environ['HF_TOKEN'] = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# 3. Set your custom cache directory
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# 4. Load the local pipeline (This will take a few minutes the first time it downloads!)
llm = HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen2.5-1.5B-Instruct',  # <--- THIS IS THE ONLY LINE THAT CHANGED
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What you think who will win thios year IPL ?")
print(result.content)