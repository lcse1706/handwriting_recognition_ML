import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from PIL import Image
import keras


# Flask application initialization
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Constant model output length
MAX_LENGTH = 34

keras.config.enable_unsafe_deserialization()


# Paths to model and tokenizer
MODEL_PATH = 'model/handwriting_recognition_34_fullDS.keras'
TOKENIZER_PATH = "tokenizer.json"

# Loading the model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Loading the tokenizer
tokenizer = None
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)
    print("Tokenizer loaded successfully.")
else:
    print("Missing tokenizer.json file – decoding may not work properly.")

# Function to preprocess the image
def preprocess_image(image_path, target_size=(128, 32)):
    try:
        img = Image.open(image_path).convert("L")  # Grayscale
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalization
        img_array = np.expand_dims(img_array, axis=-1)  # Adding channel
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to predict text
def predict_text(image_path):
    if model is None or tokenizer is None:
        return "Error: Model or tokenizer was not loaded correctly."

    img_array = preprocess_image(image_path)

    if img_array is None:
        return "Error processing image."

    img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
    print(f"Processed image shape: {img_array.shape}")

    # Model prediction
    prediction = model.predict(img_array)
    print(f"Prediction shape: {prediction.shape}")  # It should be (1, MAX_LENGTH, number_of_classes)

    # Check if the prediction length is correct
    if prediction.shape[1] != MAX_LENGTH:
        print(f"Error: Expected {MAX_LENGTH}, but model returned {prediction.shape[1]}")
        return "Prediction output length error."

    # Convert to character indices
    predicted_sequence = np.argmax(prediction, axis=-1)[0]
    print("Predicted sequence:", predicted_sequence)

    # Decode text from indices
    decoded_text = ''.join([tokenizer.index_word.get(int(i), '') for i in predicted_sequence if i > 0])
    print("Decoded text:", decoded_text)

    return decoded_text

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction handling
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Error: No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "Error: No file selected."

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    predicted_text = predict_text(file_path)

    return render_template('result.html', predicted_text=predicted_text, file_path=file_path)

# Running the server
if __name__ == '__main__':
    app.run(debug=True)
