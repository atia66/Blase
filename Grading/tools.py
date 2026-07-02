from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field 
from typing import Dict
import os
from dotenv import load_dotenv
import time
load_dotenv()


class GradeComplete(BaseModel):
    scores: Dict[str, float] 
    
class GradeWritten(BaseModel):
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Score from 0 to 10, must be a numeric value like 7 or 2 not a string like '7'"
    )

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("model"),
        google_api_key=os.getenv("api_key"),
        temperature=0.0,
        max_output_tokens=256,
    )

def model_invoke(llm, schema: type, prompt: str):
    structured_llm = llm.with_structured_output(schema, include_raw=True)
    response = structured_llm.invoke(prompt)

    raw_msg = response["raw"]
    parsed  = response["parsed"]


    if parsed is None:
        parsing_error = response.get("parsing_error")
        raw_content   = getattr(raw_msg, "content", repr(raw_msg))
        raise ValueError(
            f"Structured output parsing failed for schema '{schema.__name__}'. "
            f"Parsing error: {parsing_error} | "
            f"Raw response: {raw_content}"
        )

    usage_meta = getattr(raw_msg, "usage_metadata", None) or {}
    if hasattr(usage_meta, "__dict__"):
        usage_meta = usage_meta.__dict__

    usage = {
        "input":  usage_meta.get("input_tokens",  0),
        "output": usage_meta.get("output_tokens", 0),
    }
    return parsed, usage
