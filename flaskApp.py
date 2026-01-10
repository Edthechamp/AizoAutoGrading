import json
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

#TODO:gettot correct answers no JSON
#TODO:parejiem JSONS salidzinat answers ar tiem pareizaja
#TODO:score uploadot uz database

#viens no goals ir lai in ram viena bridi butu tikai answers un 1 skolena atbildes un ne visas atbildes vienlaicigi

app = Flask(__name__)

UPLOAD_DIR = "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#image upload daļa, saliek "/imgaes" folderā, pārstrādājot atbildes jāliek uz "/GeneratedAnswers"
@app.route("/imgUpload", methods=["GET", "POST"])
def imgUpload():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify(error="No image"), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify(error="Empty filename"), 400

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_DIR, filename)
        file.save(path)

        return jsonify(
            message="Upload successful",
            filename=filename
        ), 200

    #runosis kad method ir get
    #temprorary basic HTML
    return """
    <form method="POST" action="/imgUpload" enctype="multipart/form-data">
        <input type="file" name="image">
        <button type="submit">Upload</button>
    </form>
    """




#random testing
# def getAnswers(n):
#     with open(f"GeneratedAnswers/gen{n}.json","r",encoding="utf-8") as f:
#         return(json.load(f))

# def ansCount():
#     path = "GeneratedAnswers"
#     return sum(
#         1 for entry in os.listdir(path)
#         if os.path.isfile(os.path.join(path, entry))
#     )





