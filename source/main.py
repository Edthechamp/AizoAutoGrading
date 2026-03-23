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


def dispensePage(timeout=25):
    result = subprocess.run(
        ["lp", "-d", "Dispenser"], 
        input="\f", 
        text=True, 
        capture_output=True
    )

    if result.returncode != 0:
        print(f"Genuine hz kam janotiek lai butu sads error")
        return False
    
    match = re.search(r'request id is (\S+)', result.stdout)
    if not match:
        print(f"Failed to get job ID from output: {result.stdout}")
        return False
        
    job_id = match.group(1)

    start_time = time.time()
    while time.time() - start_time < timeout:

        completed = subprocess.run(["lpstat", "-W", "completed"], capture_output=True, text=True)
        
        if job_id in completed.stdout:
            print("Dispensed successfully.")
            return True
            
        time.sleep(0.5)
    
    return False


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
        test = getTest()

        if test["StudentID"]=="Teacher":
            setAnswers(test["answers"])
            print("Answers have ben changed based on scanned test")
            #idejiski sada gadijuma japarlabo viss kas ir biijs bet nu hz ka handelot so lwk 
            #ganjau so janem ara un vnk teacheriem ir kk manually jaentero atbildes pirms pat sak skenesanu
            continue

        grade = checkAns.GradeTest(test)
        #sql.write(studentID, PhysicsGrades, etc..)
    else:
        print("Grading is Done!")
        break

        

