from langgraph.graph import StateGraph, START, END
from Grading.preprocess import preprocess_node, GraphState
from Grading.tools import model_invoke, GradeWritten ,get_llm
import time



DELAY_BETWEEN_CALLS = 2  
def grade_all(state: GraphState):
    results = {}
    units   = []
    llm = get_llm()              
    with open("./Grading/prompts/written.txt") as f:
        prompt_template = f.read()

    for i, task in enumerate(state["tasks"], start=1):
        key    = task["key"]
        prompt = prompt_template.replace("{MODEL_ANSWER}", task["model_answer"])
        prompt = prompt.replace("{STUDENT_ANSWER}", task["student_answer"])

        result, usage = model_invoke(llm,GradeWritten,prompt )
        results[key]  = result.model_dump()

        units.append({
            "unit_index": i,
            "label":      key,
            "tokens_input":  usage["input"],
            "tokens_output": usage["output"],
            "api_calls":     1,
        })

        time.sleep(DELAY_BETWEEN_CALLS)

    return {"results": results, "usage": {"unit_label": "question", "units": units}}



def written_graph():
    builder = StateGraph(GraphState)
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("grade_all", grade_all)
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "grade_all")
    builder.add_edge("grade_all", END)
    return builder.compile()

def written_producer(model, student):
    graph  = written_graph()
    result = graph.invoke({"model": model, "student": student})
    score = sum(i["overall_score"] for i in result["results"].values())
    total = len(result["results"]) * 10
    usage = result["usage"]

    return result,score, total, usage
