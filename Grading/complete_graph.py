from langgraph.graph import StateGraph, START, END
from Grading.preprocess import  GraphState
from Grading.tools import model_invoke, GradeComplete ,get_llm


def grade(state: GraphState):
    llm     = get_llm()        

    model   = state["model"]
    student = state["student"]
    questions = []

    for key in model:
        questions.append(f"""
        QID: {key}
        Model Answer: {model[key]}
        Student Answer: {student.get(key, "")}""")

    prompt = f"""
        You are an exam grader. 
        Return JSON:
        {{
        "scores": {{
            "q1": 1,
            "q2": 0
        }}
        }}
        Rules:
        - Use EXACT question IDs
        - 1 or 0 only 
        - No explanation
        Allow:
        - spelling and grammar mistakes.
        Questions:
        {''.join(questions)}
        """

    result, usage = model_invoke(llm, GradeComplete,prompt)
    unit={
            "unit_index": 1,
            "label":      "Sheet",
            "tokens_input":  usage["input"],
            "tokens_output": usage["output"],
            "api_calls":     1,
        }
    
    return {
        "results": result.scores,
        "usage": {
                "unit_label": "complete",
                "unit":unit

        }
    }

def complete_graph():
    builder = StateGraph(GraphState)

    builder.add_node("grade", grade)

    builder.add_edge(START, "grade")
    builder.add_edge("grade", END)

    return builder.compile()

def complete_producer(model, student):
    graph  = complete_graph()
    result = graph.invoke({"model": model, "student": student})

    score = sum(result["results"].values())
    total = len(result["results"])
    usage = result["usage"]

    return result,score, total, usage

