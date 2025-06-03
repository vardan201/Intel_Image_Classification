from flask import Flask, request, render_template_string
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
model = tf.keras.models.load_model("intel_model.keras")
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Intel Image Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding-top: 50px;
            background-color: #f0f0f0;
        }
        h1 {
            color: #333;
        }
        form {
            margin: 20px auto;
            padding: 20px;
            background: white;
            border-radius: 8px;
            width: 300px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        input[type="file"] {
            margin-bottom: 10px;
        }
        button {
            padding: 10px 20px;
            background: #007BFF;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            font-weight: bold;
            font-size: 1.2em;
            color: #007BFF;
        }
    </style>
</head>
<body>
    <h1>Intel Image Classifier</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required><br>
        <button type="submit">Predict</button>
    </form>
    {% if prediction %}
        <div class="result">{{ prediction }}</div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            img = Image.open(image_file).resize((224, 224))
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            predicted_class = class_names[np.argmax(preds[0])]
            prediction = f"Prediction: {predicted_class}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
