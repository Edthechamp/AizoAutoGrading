import os
import json
import time

debug=False
# with open("AnswerSheets/answers1.json","r",encoding="utf-8") as f:
#     answerSheet = json.load(f)
#     rightAnswers = list(answerSheet["answers"].values())

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


answerSheets=[]
for file in os.listdir("AnswerSheets"):
    if os.path.splitext(file)[1]==".json":
        with open(os.path.join("AnswerSheets", file), "r") as f:
            answerSheets.append(json.load(f))


#OLD LOOPING MECHANISM
# while True:
#     if len(os.listdir("GeneratedAnswers"))==0:
#         retries+=1
#         if retries==6:
#             print("timeout, assuming done")
#             break
#         print("no files found, press ctrl+c to close")
#         time.sleep(retries)
#         continue

#     retries=0

#STORING ANSWERS AS JSON, OPTIONAL, PROLLY UNNECCESARY
    # filename= os.listdir("GeneratedAnswers")[0]
    # path = os.path.join("GeneratedAnswers",filename)

    # try:
    #     with open(path,"r",encoding="utf-8") as f:
    #         test =json.load(f)
    # except (json.JSONDecodeError, PermissionError):
    #     time.sleep(0.2)
    #     continue

def GradeTest(test):
    # if not test["examID"] == answerSheet["examID"]:
    #     print("test mismatch") 
    #     os.rename(path,os.path.join("ManualChecking",f"{test['studentID']}.json"))
    #     continue

    answers = None

    for ash in answerSheets:
        if test["examID"] == ash["examID"]:
            answers=ash["answers"]
            print(f"loaded correct answers for examID{test['examID']}")
            break

    if not answers:
        if debug:
            print(f"failed to find examIS {test['examID']} in answerSheet")
        return {"error":"tests examID not foudn in answers"}

    stSheet=test["answers"]

    if len(stSheet)!=len(answers):
        stSheet = dataNormalize(stSheet,answers)
        print("missing questions, filling in blanks")

    stGrade=0 
    
    for q in list(answers.keys()):
        if stSheet[q]==answers[q]:
            stGrade+=1 #possibly receive the specific questions point value in the future(not just 1 but 1/2/3... depending on question weight)

    if debug:
        print(f"{test['studentID']} got {stGrade} out of {len(answers)}")
         #TODO:write to sql or atleast something permanent, both grade and answers

    # maybe TODO: sagrupet pa dazadam sekcijam stGrade lai redz fizikas/kimijas seperate atzimes

    #pagaidam returnoju tikai pasu grade un studentID for redundency, most kk vairak vjg hz
    return stGrade

    #From old philosophy of managing files with different folders
    # os.rename(path, os.path.join("GradedAnswers",f"{test["studentID"]}.json") )
    