def MCQ_worker(student_answer, model_answer):
    score = 0
    total = 0
    result = {}

    for key, value in model_answer.items():
        answers = student_answer.get(key, [])

        if len(answers) != 1:     
            result[key] = 0
        elif answers[0] == value[0]:  
            result[key] = 1
            score += 1
        else:
            result[key] = 0

        total += 1

    return result, score, total

