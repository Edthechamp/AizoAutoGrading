import os
import json
from helperFunctions import dataNormalize
import time


with open("AnswerSheets/answers1.json","r",encoding="utf-8") as f:
    answerSheet = json.load(f)
    rightAnswers = list(answerSheet["answers"].values())

retries=0

while True:
    if len(os.listdir("GeneratedAnswers"))==0:
        retries+=1
        if retries==6:
            print("timeout, assuming done")
            break
        print("no files found, press ctrl+c to close")
        time.sleep(retries)
        continue
    retries=0
    filename= os.listdir("GeneratedAnswers")[0]
    path = os.path.join("GeneratedAnswers",filename)
    try:
        with open(path,"r",encoding="utf-8") as f:
            test =json.load(f)
    except (json.JSONDecodeError, PermissionError):
        time.sleep(0.2)
        continue


    if not test["examID"] == answerSheet["examID"] or not test["examVersion"]==answerSheet["examVersion"]:
        print("test mismatch") 
        os.rename(path,os.path.join("ManualChecking",f"{test['studentID']}.json"))
        continue


    if len(test["answers"])!=len(answerSheet["answers"]):
        stSheet = dataNormalize(test["answers"],answerSheet["answers"])
    else:
        stSheet=test["answers"]
    stGrade=0 
    
    for q in list(answerSheet["answers"].keys()):
        if stSheet[q]==answerSheet["answers"][q]:
            stGrade+=1 #possibly receive the specific questions point value in the future

    print(f"{test['studentID']} got {stGrade} out of {len(rightAnswers)}") #TODO:write to sql
 

    os.rename(path, os.path.join("GradedAnswers",f"{test["studentID"]}.json") )
    