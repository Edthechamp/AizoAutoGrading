import os
import json
import time
from pathlib import Path

debug=False


def getFilePath(file):
    base_path = Path(__file__).parent.parent
    print(base_path / "Resources" / file)
    return base_path / "Resources" / file

#manliekas better approach, ari stradas ar multiple answer sheets.

def dataNormalize(stSheet, ansSheet):
    normalized = {}
    for q in ansSheet.keys():
        normalized[q] = stSheet.get(q, None)
    try:
        normalized = dict(sorted(normalized.items(), key=lambda x: int(x[0])))
    except ValueError:
        normalized = dict(sorted(normalized.items()))
    return normalized

def load_answer_sheet():
    with open(getFilePath("answers.json"), "r") as f:
        return json.load(f)
        
answerSheet=load_answer_sheet()

def GradeTest(test):
    answerSheet=load_answer_sheet()
    stSheet=test["answers"]

    for section in stSheet.keys():
        if len(stSheet[section])!=len(answerSheet[section]):
            stSheet[section] = dataNormalize(stSheet[section],answerSheet[section])
            if debug:
                print("missing questions, filling in blanks")
    
    stGrade=0

    grades={}

    for section in answerSheet.keys():
        for q in list(answerSheet[section].keys()):
            if stSheet[section][q]==answerSheet[section][q]:
                stGrade+=1 #possibly receive the specific questions point value in the future(not just 1 but 1/2/3... depending on question weight)
        grades[section]=stGrade
        stGrade=0

    #TODO:write to sql or atleast something permanent, both grade and answers

    return {"code":test['studentID'],"scores":grades}
