import os
import json


with open("AnswerSheets/answers1.json","r",encoding="utf-8") as f:
    answers = json.load(f)

while len(os.listdir("GeneratedAnswers"))>0:
    filename= os.listdir("GeneratedAnswers")[0]
    path= os.path.join("GeneratedAnswers",filename)

    with open(path,"r",encoding="utf-8") as f:
        #TODO:write to sql
        test=json.load(f)

        if not test["examID"] == answers["examID"] or not test["examVersion"]==answers["examVersion"]:
            print("test mismatch")
            #TODO: ko lwk darit saja situacija
            break

        for ans in test["answers"]:
            print(ans)
            
        os.remove(path)
        #TODO 
        print("---------end of file",path,"-----------")