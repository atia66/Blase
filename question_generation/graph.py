from typing import List, TypedDict, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import time
import os
from dotenv import load_dotenv

load_dotenv()

max_tokens = 4096
SLEEP_BETWEEN_CALLS = 4  



class KeyPoints(BaseModel):
    points: List[str] = Field(
        description=(
            "A list of detailed, self-contained keypoints extracted from the text. "
            "Each keypoint must be a full sentence that includes the concept and its definition."
        )
    )

class MCQQuestion(BaseModel):
    questions: List[str]       = Field(description="Multiple choice questions.")
    choices:   List[List[str]] = Field(description="Exactly four SHORT answer choices per question. Each choice must be under 8 words.")
    answers:   List[str]       = Field(description="Correct answer, copied EXACTLY word-for-word from one of the four choices. No prefixes.")
    levels:    List[str]       = Field(description="Difficulty: Easy, Medium, or Hard, one per question.")

class CompleteQuestion(BaseModel):
    questions: List[str] = Field(description="Fill-in-the-blank questions.")
    answers:   List[str] = Field(description="Correct answer, two words or less.")
    levels:    List[str] = Field(description="Difficulty: Easy, Medium, or Hard, one per question.")

class WrittenQuestion(BaseModel):
    questions: List[str] = Field(description="Open-ended written questions.")
    answers:   List[str] = Field(description="Written answer, two sentences max.")
    levels:    List[str] = Field(description="Difficulty: Easy, Medium, or Hard, one per question.")

schema_map = {
    "mcq":      MCQQuestion,
    "written":  WrittenQuestion,
    "complete": CompleteQuestion,
}



class State(TypedDict):
    text:        str
    keypoints:   List[str]
    results:     Dict[str, Any]
    token_usage: List[Dict[str, Any]]
    usage_acc:   Dict[str, int]



def make_llm(max_tok: int) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("model"),
        google_api_key=os.getenv("api_key"),
        temperature=0.3,
        max_output_tokens=max_tok,
    )



def invoke_structured(llm, schema: type, prompt: str):
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



def clean_result(data: Dict[str, Any], qtype: str) -> Dict[str, Any]:
    if not data:
        return {}

    questions = data.get("questions", [])
    answers   = data.get("answers",   [])
    levels    = data.get("levels",    [])
    choices   = data.get("choices",   None)
    out: Dict[str, Any] = {}
    counter = 1

    if qtype == "mcq" and choices is not None:
        for q, c, a, l in zip(questions, choices, answers, levels):
            if not q.strip() or not a.strip():
                continue
            if len(c) != 4 or not all(opt.strip() for opt in c):
                continue
            matched = a if a in c else next(
                (opt for opt in c if opt in a or a.endswith(opt)), None
            )
            if matched is None:
                continue
            out[f"q{counter}"] = {"question": q, "choices": c, "answer": matched, "level": l}
            counter += 1
    else:
        seen = set()
        for q, a, l in zip(questions, answers, levels):
            if not q.strip() or not a.strip() or q in seen:
                continue
            seen.add(q)
            out[f"q{counter}"] = {"question": q, "answer": a, "level": l}
            counter += 1

    return out



def extractor_node(state: State) -> Dict[str, Any]:
    llm = make_llm(max_tokens)

    prompt = (
        "Extract key points from the text below. "
        "Key points must be about core concepts only, not about the lecturer or meta-content. "
        "Each must be one self-contained sentence.\n\n"
        f"{state['text']}"
    )

    try:
        parsed, usage = invoke_structured(llm, KeyPoints, prompt)
        keypoints = parsed.points
    except Exception as e:
        print(f"[extractor_node] Failed to extract keypoints: {type(e).__name__}: {e}")
        keypoints = []
        usage = {"input": 0, "output": 0}

    time.sleep(SLEEP_BETWEEN_CALLS)

    return {
        "keypoints": keypoints,
        "usage_acc": {
            "input":     usage["input"],
            "output":    usage["output"],
            "api_calls": 1,
        },
    }


def generator_node(state: State) -> Dict[str, Any]:
    keypoints = state["keypoints"]
    acc = dict(state.get("usage_acc") or {"input": 0, "output": 0, "api_calls": 0})

    if not keypoints:
        print("[generator_node] No keypoints available — skipping question generation.")
        return {
            "results": {"mcq": {}, "complete": {}, "written": {}},
            "token_usage": [{
                "tokens_input":  acc.get("input",     0),
                "tokens_output": acc.get("output",    0),
                "api_calls":     acc.get("api_calls", 0),
            }],
        }

    results: Dict[str, Any] = {}
    kp_block = "\n".join(f"{i+1}. {kp}" for i, kp in enumerate(keypoints))
    n = len(keypoints)

    def call(qtype: str) -> None:
        llm = make_llm(max_tokens)

        extra = (
            " Each choice must be under 8 words. "
            "The answer must be copied EXACTLY from one choice, no prefixes."
            "choices must be realstic and varient not same answer"
            if qtype == "mcq" else ""
        )
        answer_hint = " Each answer must be under 3 sentances." if qtype == "written" else ""

        prompt = (
            f"You are a teacher writing an exam.\n"
            f"Write exactly {5} {qtype} questions  in order.\n"
            f"Stop after question {n}. Do not repeat or loop.{extra}{answer_hint}\n"
            f"Levels must be Easy, Medium, or Hard.\n\n"
            f":\n{kp_block}"
        )

        try:
            parsed, usage = invoke_structured(llm, schema_map[qtype], prompt)
            results[qtype]    = clean_result(parsed.model_dump(), qtype)
            acc["input"]     += usage["input"]
            acc["output"]    += usage["output"]
            acc["api_calls"] += 1
        except Exception as e:
            print(f"[generator_node] [{qtype}] {type(e).__name__}: {e}")
            results[qtype] = {}

        time.sleep(SLEEP_BETWEEN_CALLS)

    call("mcq")
    call("complete")
    call("written")

    token_snapshot = {
        "tokens_input":  acc["input"],
        "tokens_output": acc["output"],
        "api_calls":     acc["api_calls"],
    }

    return {"results": results, "token_usage": [token_snapshot]}



def graph_init():
    graph = StateGraph(State)
    graph.add_node("extract",  extractor_node)
    graph.add_node("generate", generator_node)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "generate")
    graph.add_edge("generate", END)
    return graph.compile()