from app.schema import IntentOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model =os.getenv("MODEL_NAME"))

template = PromptTemplate(
    input_variables=["input"],
    template="""
You are a classification model.

Classify the query into one or more of the following intents ONLY (case sensitive, exact spelling):
- "risk_analysis"
- "summary"
- "query_answer"
Dont use any other keyword than above 3.
Dont say summarize instead of summary.
Dont say risk analysis instead of risk_analysis.
Respond ONLY with a valid JSON array of these exact strings.
Query: {input}
"""
)

structured_model = llm.with_structured_output(IntentOutput)

chain = template | structured_model



if __name__ == "__main__":
    result = chain.invoke("give me summary of this file")
    print(result)
    print(result.intent)
    result1 = chain.invoke("give me summary of this file and tell me risk level of this file")
    print(result1)
    print(result1.intent)
    result2 = chain.invoke("what is turnover period")
    print(result2.intent)