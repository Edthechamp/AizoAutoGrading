import os
from pathlib import Path
import json

import subprocess 
import re
import time

import checkAns
import process_image

def tempStructure(answers):
    return {"StudentID":"Example","answers":answers}

def getTest():

    test = process_image.get_answers(debug=False)
    #sis returnos error ka nav imagepath, bet mes vairs neizmantojam fileBased image passing bet ka saja momenta janofoce
    #ka ari temp Structure jaaizvieto ar legit ID iegusano 
    testStructured = tempStructure(test)
    return testStructured

    # studentGrades = checkAns.GradeTest(testStructured)
    # return studentGrades

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





# answerBuffer = process_image.get_answers(getFilePath("image.jpg"), debug=False)
# print(answerBuffer)
# setAnswers(answerBuffer)

# print("------------")
# print(getGrade(getFilePath("image.jpg")))


#!!!!!!!!!!!!!!-------------------------------------------------------------------!!!!!!!!!!!!!!
#galvena dala type shit 

from qt_app import main.py

#seit prly vjg kk parameters = main.get_params() kas kipa returno tos punktuts ko setuo users during setup, lai tos izmantotu visos scans pec tam


#!!!not tested, bet imo jastrada!!!
while True:
    if dispensePage:

        grade = checkAns.GradeTest(test)
        #sql.write(studentID, PhysicsGrades, etc..)
    else:
        print("Grading is Done!")
        break

        

