import os
# 1. Disable GPU for Render Free Tier (Required to avoid memory crashes)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
# 2. Allow requests from your React Frontend
CORS(app)

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model is live and ready!")

def prepare_image(image):
    # Resize must match your training (128x128)
    img = image.convert('RGB').resize((128, 128))
    img_array = np.array(img, dtype=np.float32) 
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        img = Image.open(file)
        processed_img = prepare_image(img)
        
        # Run inference using the TFLite Interpreter
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        
        # Extract the prediction (Real=1, Fake=0)
        prediction = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        
        is_authentic = bool(prediction > 0.5)
        # Calculate percentage: if Fake, we show confidence in "Fakeness"
        confidence = float(prediction if is_authentic else 1 - prediction)

        return jsonify({
            'is_authentic': is_authentic,
            'confidence_score': round(confidence * 100, 2),
            'result': 'REAL' if is_authentic else 'FAKE'
        })
    except Exception as e:
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

@app.route('/')
def home():
    return "TrustGuard AI Backend is Online!"

if __name__ == '__main__':
    # Render binds to port 10000 by default
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)