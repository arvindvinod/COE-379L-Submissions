from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model('final_best_model.h5')

@app.route('/summary', methods=['GET'])
def model_summary():
    """Endpoint that returns model metadata"""
    return jsonify({
        "name": "Alternative LeNet-5 for Damage Classification",
        "description": "Binary classifier for building damage after Hurricane Harvey",
        "version": "1.0",
        "architecture": "Alternative LeNet-5",
        "input_shape": [150, 150, 3],  
        "output_classes": ["no_damage", "damage"],
        "test_accuracy": 0.9809
    })

@app.route('/inference', methods=['POST'])
def classify_image():
    """Endpoint that performs image classification"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        image_file = request.files['image']
        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({"error": "Invalid file type"}), 400
            
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode in ('L', 'LA', 'RGBA', 'P'):
            image = image.convert('RGB')
        
        image = image.resize((150, 150))  
        image_array = np.array(image) / 255.0
        
        if image_array.shape != (150, 150, 3):
            return jsonify({"error": f"Unexpected image shape after processing: {image_array.shape}"}), 400
            
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction_prob = model.predict(image_array)[0][0]  
        threshold = 0.5
        class_label = "no_damage" if prediction_prob >= threshold else "damage"
    
        return jsonify({
            "prediction": class_label,
            "confidence": float(prediction_prob) 
        })
    
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)