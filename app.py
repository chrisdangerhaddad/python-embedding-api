from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import time

app = Flask(__name__)
CORS(app)

# Load the model when the server starts
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

@app.route('/get-embedding', methods=['POST'])
def get_embedding():
    start_time = time.time()
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    # Generate embedding
    embedding = model.encode(data['text']).tolist()
    
    processing_time = time.time() - start_time
    return jsonify({
        'embedding': embedding,
        'processing_time_ms': processing_time * 1000
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)