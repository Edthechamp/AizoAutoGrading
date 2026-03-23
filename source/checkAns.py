import os
import json
import time
from pathlib import Path

from main import getFilePath
debug=False


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


answerSheet=None

with open(getFilePath("answers.json"), "r") as f:
    answerSheet = json.load(f)

if not answerSheet:
    print("!!!ERROR: answers not found!!!")
    exit()


def GradeTest(test):
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

    return {"studentID":test['StudentID'],"scores":grades}
