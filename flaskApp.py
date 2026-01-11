import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_DIR = "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#image upload daļa, saliek "/imgaes" folderā, pārstrādājot ar OCR atbildes jāliek uz "/GeneratedAnswers"
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

    #executos kad method ir get
    #temprorary basic HTML
    return """
    <form method="POST" action="/imgUpload" enctype="multipart/form-data">
        <input type="file" name="image">
        <button type="submit">Upload</button>
    </form>
    """








