from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = &#39;uploads&#39;
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config[&#39;UPLOAD_FOLDER&#39;] = UPLOAD_FOLDER
@app.route(&#39;/upload&#39;, methods=[&#39;POST&#39;])
def upload_file():
if &#39;file&#39; not in request.files:
return jsonify({&#39;error&#39;: &#39;No file uploaded&#39;}), 400
file = request.files[&#39;file&#39;]
if file.filename == &#39;&#39;:
return jsonify({&#39;error&#39;: &#39;No selected file&#39;}), 400
filename = secure_filename(file.filename)
filepath = os.path.join(app.config[&#39;UPLOAD_FOLDER&#39;], filename)
file.save(filepath)
return jsonify({&#39;message&#39;: &#39;File uploaded successfully&#39;,
&#39;file_path&#39;: filepath})
if __name__ == &#39;__main__&#39;:
app.run(debug=True)