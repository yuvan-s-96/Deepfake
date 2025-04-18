from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

API_URL = "https://api-inference.huggingface.co/models/motheecreator/Deepfake-audio-detection"
headers = {"Authorization": "Bearer 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"}

# Ensure the temp directory exists
if not os.path.exists('./temp'):
    os.makedirs('./temp')

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    
    # Log response details for debugging
    print("API Status Code:", response.status_code)
    print("API Response:", response.text)
    
    return response.json()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        # Save the file temporarily
        filepath = f"./temp/{file.filename}"
        file.save(filepath)
       
        # Perform the query
        output = query(filepath)
        
        # Clean up the temporary file
        os.remove(filepath)
        
        # Return the output as JSON
        return jsonify(output)

@app.route('/')
def home():
    return jsonify({"message": "Flask server is running."})

if __name__ == "__main__":
    app.run(debug=True)
