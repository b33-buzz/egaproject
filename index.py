import os
from flask import Flask, request, render_template_string, redirect, send_from_directory, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
import base64
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load your pre-trained model
model_path = 'model/Klasifikasi_ikan.h5'
loaded_model = load_model(model_path)

# Preprocess image function
def preprocess_image(img, target_size):
    try:
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        return img_array
    except UnidentifiedImageError:
        logging.error("Cannot identify image file")
        return None

# Predict fish species function
def predict_fish_species(img):
    img_array = preprocess_image(img, target_size=(224, 224))
    if img_array is None:
        return "Unknown Fish", 0

    img_array = np.expand_dims(img_array, axis=0)
    prediction = loaded_model.predict(img_array)
    class_names = ['Black Sea Sprat', 'Gilt-Head Bream', 'Horse Mackerel', 'Red Mullet',
                   'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout']

    max_probability = np.max(prediction)
    if max_probability < 0.85:
        predicted_class = "Unknown Fish"
    else:
        predicted_class = class_names[np.argmax(prediction)]

    return predicted_class, max_probability

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# HTML template with WebRTC for camera
template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fish Detect</title>
    <style>
    body {
        font-family: "EB Garamond", serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        height: 80vh;
        background: radial-gradient(#1cb5e0, #000066);
        margin: 0;
        color: #fff;
        padding-top: 20px;
    }

    .container {
        background: #FFF;
        color: #333;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        width: 100%;
        max-width: 800px;
        margin-top: 30px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    h1 {
        font-size: 3em;
        margin-bottom: 20px;
        margin-top: 40px;
    }

    h2, h3 {
        text-align: center;
    }

    form {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    input[type="file"], input[type="submit"], button {
        margin: 10px 0;
        padding: 10px;
        border: none;
        border-radius: 5px;
        background-color: #000066;
        color: white;
        cursor: pointer;
    }

    input[type="submit"]:hover, button:hover {
        background-color: #150754;
    }

    video {
        border-radius: 10px;
        width: 400px;
        margin-bottom: 10px;
    }

    .realtime-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    #result {
        font-size: 20px;
        color: #000066;
        margin-top: 20px;
    }

    .fish-card-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 40px;
    }

    .fish-card {
        background: #fff;
        color: #333;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 10px;
        padding: 20px;
        width: 180px;
    }

    .fish-card img {
        width: 100%;
        height: auto;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .fish-card p {
        margin: 0;
        font-size: 1em;
        font-weight: bold;
        color: #000066;
    }
</style>
</head>
<body>
    <h1>Fish Detect</h1>
    <div class="container">
        <h2>Upload an Image or Use Your Camera</h2>

        <!-- Form for uploading image -->
        <form action="/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" id="file">
            <input type="submit" value="Upload">
        </form>

        <!-- Video and Capture Button -->
        <h2>Or use the camera to detect in real-time:</h2>
        <div class="realtime-container">
            <video id="videoElement" autoplay></video>
            <button id="captureButton">Capture and Detect</button>
        </div>

        <!-- Display Prediction Result -->
        <div id="result">
            {% if predicted_class and confidence %}
                <p>Detected: {{ predicted_class }}, Confidence: {{ confidence }}%</p>
            {% endif %}
        </div>

        <h3>Jenis Ikan yang dapat di deteksi</h3>
        <div class="fish-card-container">
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/BlackSeaSprat.png') }}" alt="Black Sea Spar">
                <p>Black Sea Spar</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/GiltHeadBream.JPG') }}" alt="Gilt Head Bream">
                <p>Gilt Head Bream</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/HorseMackerel.png') }}" alt="Horse Mackerel">
                <p>Horse Mackerel</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/RedMullet.png') }}" alt="Red Mullet">
                <p>Red Mullet</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/RedSeaBream.JPG') }}" alt="Red Sea Bream">
                <p>Red Sea Bream</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/SeaBass.JPG') }}" alt="Sea Bass">
                <p>Sea Bass</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/Shrimp.png') }}" alt="Shrimp">
                <p>Shrimp</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/StripedRedMullet.png') }}" alt="Striped Red Mullet">
                <p>Striped Red Mullet</p>
            </div>
            <div class="fish-card">
                <img src="{{ url_for('static', filename='images/Trout.png') }}" alt="Trout">
                <p>Trout</p>
            </div>
        </div>
    </div>

    <!-- WebRTC and capture image -->
    <script>
        const video = document.querySelector('#videoElement');
        const captureButton = document.querySelector('#captureButton');
        const resultDiv = document.querySelector('#result');
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        // Access the user's webcam
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
                video.srcObject = stream;
                video.play();
            });
        }

        // Capture image and send to the server for prediction
        captureButton.addEventListener('click', function() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
            
            const imageDataURL = canvas.toDataURL('image/jpeg');
            const formData = new FormData();
            formData.append('image', imageDataURL);
            
            fetch('/capture', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                resultDiv.innerHTML = `Detected: ${data.predicted_class}, Confidence: ${data.confidence}%`;
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
'''

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    predicted_class = None
    confidence = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img = Image.open(filepath)
            predicted_class, confidence = predict_fish_species(img)
            confidence = "{:.1f}".format(confidence * 100)
    
    # Pass predicted values to the template
    return render_template_string(template, predicted_class=predicted_class, confidence=confidence)

# Route for handling image capture from WebRTC
@app.route('/capture', methods=['POST'])
def capture_image():
    image_data = request.form['image']
    image_data = image_data.replace('data:image/jpeg;base64,', '')
    image_data = base64.b64decode(image_data)
    img = Image.open(BytesIO(image_data))
    predicted_class, confidence = predict_fish_species(img)
    confidence = "{:.1f}".format(confidence * 100)
    
    return jsonify({'predicted_class': predicted_class, 'confidence': confidence})

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
