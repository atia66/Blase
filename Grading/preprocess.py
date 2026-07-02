from typing import TypedDict, Dict, List, Annotated

class GraphState(TypedDict):
    model: Dict[str, str]
    student: Dict[str, str]
    tasks: List[Dict]
    results: Annotated[dict, lambda a, b: {**a, **b}]
    usage: None

def preprocess_answer(model_answer, student_answer):
    new_model_answer = {}
    new_student_answer = {}

    for key in model_answer:
        if model_answer[key] == "":
            break

        new_model_answer[key] = model_answer[key]
        new_student_answer[key] = student_answer.get(key, "")

    return new_model_answer, new_student_answer

def preprocess_node(state: GraphState):

    m, s = preprocess_answer(
        state["model"],
        state["student"]
    )
    tasks = [
        {
            "key": key,
            "model_answer": m[key],
            "student_answer": s[key],
        }
        for key in m
    ]
    return {
        "tasks": tasks,
        "results": {}
    }
