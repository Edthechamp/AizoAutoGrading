import os
from pathlib import Path
import json

import checkAns
import process_image

def tempStructure(grades):
    return {"StudentID":"Example","answers":grades}

def getGrade(image_path):
    test=process_image.get_answers(image_path, debug=False)
    testStructured = tempStructure(test)
    
    studentGrades = checkAns.GradeTest(testStructured)
    return studentGrades

def setAnswers(answers):
    try:
        with open(getFilePath("answers.json"),"w") as f:
            json.dump(answers,f)
    except:
        return "Failed updating answers, reason unknown"
    return "success"    

def getFilePath(file):
    base_path = Path(__file__).parent.parent
    print(base_path / "Resources" / file)
    return base_path / "Resources" / file

answerBuffer = process_image.get_answers(getFilePath("image.jpg"), debug=False)
print(answerBuffer)
setAnswers(answerBuffer)

print("------------")
print(getGrade(getFilePath("image.jpg")))