import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/plant_disease_model.h5'

model = load_model(MODEL_PATH)
CLASS_NAMES = sorted(os.listdir('dataset'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file uploaded!', 400

    image = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(path)

    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    label = CLASS_NAMES[class_idx]

    return render_template('index.html', prediction=label, image_file=image.filename)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(path)

    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    label = CLASS_NAMES[class_idx]

    return jsonify({'prediction': label})

# Add this block to start Flask app
if __name__ == '__main__':
    app.run(debug=True)



