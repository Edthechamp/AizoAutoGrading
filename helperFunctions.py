def dataNormalize(stSheet,ansSheet):
    stQuestions = list(stSheet.keys())
    rightQuestions = list(ansSheet.keys())
    missing = {q: None for q in rightQuestions if q not in stQuestions}
    print("the missing questions are: ", missing)
    stSheet.update(missing)
    stSheet = dict(sorted(
            stSheet.items(), #item ir dict line "name":"value" parveidots uz (name,value)
            key=lambda item: int(item[0].replace('question', '')))) #shrinkoju uz lambda function bet basically nonem "question" un sorto pec integer

    return stSheet