from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import time
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model_optimal.h5')

# Dictionary to label all the emotions
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load a faster face detection model
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", 
    "weights.caffemodel"
)

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Global variables for threading
frame = None
predictions = []

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, (48, 48))
    normalized_img = resized_img / 255.0
    reshaped_img = np.reshape(normalized_img, (1, 48, 48, 1))
    return reshaped_img

def detect_and_predict():
    global frame, predictions
    while True:
        if frame is not None:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            
            new_predictions = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    face = frame[y:y + (y1 - y), x:x + (x1 - x)]
                    
                    if face.size > 0:
                        preprocessed_face = preprocess_image(face)
                        result = model.predict(preprocessed_face)
                        img_index = np.argmax(result[0])
                        predicted_label = label_dict[img_index]
                        probabilities = result[0].tolist()  # Convert to Python list for JSON serialization
                        new_predictions.append({
                            'x': int(x),
                            'y': int(y),
                            'w': int(x1 - x),
                            'h': int(y1 - y),
                            'label': predicted_label,
                            'probabilities': {label_dict[i]: prob for i, prob in enumerate(probabilities)}
                        })
            
            predictions = new_predictions
        time.sleep(0.1)

def gen_frames():
    global frame
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture image")
            continue
        time.sleep(0.03)  # Adjust the sleep to balance between performance and update rate
        if frame is not None:
            frame_copy = frame.copy()
            for pred in predictions:
                x, y, w, h = pred['x'], pred['y'], pred['w'], pred['h']
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame_copy, pred['label'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame_copy)
            if not ret:
                print("Error: Failed to encode frame")
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/predictions')
def get_predictions():
    return jsonify(predictions)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=detect_and_predict, daemon=True).start()
    app.run(debug=True, threaded=True)
