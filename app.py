from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import time
import os

app = Flask(__name__)
CORS(app)

HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def get_embedding_from_huggingface(text):
    payload = {"inputs": text}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        output = response.json()
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], float):
            return output
        elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], list): # Handle potential nested list
            return output[0]
        else:
            print(f"Unexpected output from Hugging Face API: {output}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during Hugging Face API request: {e}")
        return None

@app.route('/get-embedding', methods=['POST'])
def get_embedding():
    start_time = time.time()
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    # Generate embedding using Hugging Face API
    embedding = get_embedding_from_huggingface(data['text'])

    if embedding is None:
        return jsonify({'error': 'Failed to get embedding from Hugging Face API'}), 500

    processing_time = time.time() - start_time
    return jsonify({
        'embedding': embedding,
        'processing_time_ms': processing_time * 1000
    })

if __name__ == '__main__':
    app.run(debug=True)