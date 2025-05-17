from pydantic import BaseModel
from typing import List,  Annotated , Literal

class IntentOutput(BaseModel):
    intent: Annotated[List[Literal["risk_analysis", "summary", "query_answer",]], " List of Intent of the query . return list of [risk_analysis, summary, query_answer] depending on query "]