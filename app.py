import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import warnings
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = '/tmp'

# Initialize Socket.IO with CORS allowed
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

# Load model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    model = None

# Function to process frames and predict signs
def process_image(frame):
    data_aux = []
    x_ = []
    y_ = []

    # Flip horizontally to match the right-hand orientation
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    # Convert to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)
    
    prediction = None
    confidence = 0
    predicted_character = None
    hand_box = None
    
    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract hand landmarks coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            
            # Normalize coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            
            # Calculate bounding box coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            hand_box = (x1, y1, x2, y2)
            
            # Make prediction if model is loaded
            if model is not None:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    prediction_proba = model.predict_proba([np.asarray(data_aux)])
                    confidence = float(max(prediction_proba[0]))  # Get the highest confidence score
                    predicted_character = labels_dict[int(prediction[0])]
                    
                    # Draw bounding box and prediction on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, f"{predicted_character} ({confidence*100:.1f}%)", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.3, (0, 0, 0), 3, cv2.LINE_AA)
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
            break  # Process only the first detected hand
    
    return frame, predicted_character, confidence, hand_box

# Routes and Socket.IO events
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status})

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    socketio.emit('status', {'status': 'connected', 'model_loaded': model is not None})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('process_frame')
def handle_process_frame(data):
    """Process a frame sent through Socket.IO"""
    try:
        # Decode the base64 image
        if 'frame' not in data:
            logger.error("No frame data in the request")
            return
        
        image_data = data['frame']
        if image_data.startswith('data:image'):
            # Remove data URL prefix if present
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            return
        
        # Process the frame
        processed_frame, predicted_character, confidence, hand_box = process_image(frame)
        
        # If a hand was detected and prediction made
        if predicted_character and confidence >= 0.7:  # Only emit predictions with confidence >= 70%
            emit('prediction', {
                'text': predicted_character,
                'confidence': confidence
            })
        
        # Optionally return the processed frame with drawings
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        emit('processed_frame', {'image': f'data:image/jpeg;base64,{processed_image}'})
        
    except Exception as e:
        logger.error(f"Error processing frame via socket: {e}")

@app.route('/process_frame', methods=['POST'])
def process_frame_api():
    """REST API endpoint to process frames"""
    try:
        # Check if request has the file part
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame part in the request'}), 400
        
        file = request.files['frame']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            # Save the file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read the image
            frame = cv2.imread(filepath)
            
            # Process the image
            _, predicted_character, confidence, _ = process_image(frame)
            
            # Clean up
            os.remove(filepath)
            
            # Return the prediction
            if predicted_character:
                return jsonify({
                    'text': predicted_character,
                    'confidence': confidence
                })
            else:
                return jsonify({'text': None, 'confidence': 0, 'message': 'No hand detected'})
        
        return jsonify({'error': 'Error processing file'}), 500
    
    except Exception as e:
        logger.error(f"Error in process_frame_api: {e}")
        return jsonify({'error': str(e)}), 500

def generate_frames():
    """Generate frames for web-based video stream"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               open('static/camera_error.jpg', 'rb').read() + b'\r\n')
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame, predicted_character, confidence, _ = process_image(frame)
        
        # If prediction made with good confidence, emit to all clients
        if predicted_character and confidence >= 0.7:
            socketio.emit('prediction', {
                'text': predicted_character,
                'confidence': confidence
            })
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the format expected by Flask's Response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route for web-based video streaming"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Alternative endpoint that accepts base64 encoded images
@app.route('/process_base64', methods=['POST'])
def process_base64():
    """Process a base64 encoded image"""
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data in the request'}), 400
    
    try:
        image_data = request.json['image']
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Process the image
        _, predicted_character, confidence, _ = process_image(frame)
        
        # Return the prediction
        return jsonify({
            'text': predicted_character if predicted_character else None,
            'confidence': confidence if confidence else 0
        })
        
    except Exception as e:
        logger.error(f"Error in process_base64: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/supported_signs')
def supported_signs():
    """Return a list of supported signs"""
    return jsonify({
        'signs': labels_dict
    })

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app with Socket.IO
    socketio.run(app, host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'False').lower() == 'true')