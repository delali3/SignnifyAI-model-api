from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print("Error loading the model:", e)
    model = None

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

@app.route('/')
def index():
    return "Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('landmarks')  # Expecting normalized landmark array
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        input_data = np.asarray(data).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        confidence = max(model.predict_proba(input_data)[0])
        label = labels_dict.get(prediction, 'Unknown')

        return jsonify({'prediction': label, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
