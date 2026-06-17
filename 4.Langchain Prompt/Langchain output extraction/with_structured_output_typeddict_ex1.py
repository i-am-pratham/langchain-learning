# analyze a customer review and return a strict TypedDict containing the sentiment and key themes.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import List ,TypedDict, Literal, Annotated,Optional

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ReviewAnalysis(TypedDict):
    sentiment: Annotated[Literal['pos','neg','neutral'],'return the sentiment of review either positive, negative or neutral']
    rating: Annotated[int,'return the rating between 1 to 5']
    key_complaints: Annotated[Optional[list[str]], 'return key complaints if there are any in review. If not mention dont give any ']

structural_model= model.with_structured_output(ReviewAnalysis)

prompt=ChatPromptTemplate([
    ('system', "You are an AI customer experience analyst. Extract data precisely."),
    ('human','{review_text}')
])

chain= prompt | structural_model

customer_review = "I bought this smart vacuum cleaner hoping it would save me time, but it has turned into a total nightmare and a massive waste of money. The initial setup was incredibly frustrating because the companion app kept crashing and refused to connect to my 5GHz Wi-Fi network. Once I finally got it running, the navigation system proved to be completely incompetent. It constantly gets stuck under basic furniture, eats rug fringes, and repeatedly cleans the exact same spot while ignoring entire rooms. To make matters worse, the battery dies after barely 20 minutes of use, forcing it to crawl back to the dock, which it usually fails to find anyway. When I contacted customer support to request a refund, they made me jump through hoops for two weeks before finally telling me that software glitches aren't covered under the warranty. Save your sanity and avoid this brand entirely."
result= chain.invoke({"review_text": customer_review})
print(result)


