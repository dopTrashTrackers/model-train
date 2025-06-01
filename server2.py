
# # pip install fastapi uvicorn onnxruntime opencv-python python-multipart
# # pip install websockets

import numpy as np
import cv2
import onnxruntime as ort
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX model
onnx_model_path = "bestm.onnx"
session = ort.InferenceSession(onnx_model_path)

# Define the threshold for detection and NMS IoU
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Assuming 29 classes for the model
CLASSES = ['Aluminium foil', 'Bottle cap', 'Broken glass', 'Cigarette', 'Clear plastic bottle', 'Crisp packet', 
           'Cup', 'Drink can', 'Food Carton', 'Food container', 'Food waste', 'Garbage bag', 'Glass bottle', 
           'Lid', 'Other Carton', 'Other can', 'Other container', 'Other plastic bottle', 'Other plastic wrapper', 
           'Other plastic', 'Paper bag', 'Paper', 'Plastic bag wrapper', 'Plastic film', 'Pop tab', 
           'Single-use carrier bag', 'Straw', 'Styrofoam piece', 'Unlabeled litter']

# Threshold-based alert settings
DETECTION_THRESHOLD = 20  # Spillage area threshold in percent
ALERT_COUNT_THRESHOLD = 10  # Number of times threshold is exceeded to trigger an alert
detection_counter = 0  # Counts consecutive threshold exceedances
alert_triggered = False  # Track if alert has been triggered

# Function to perform non-maximum suppression (NMS)
def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, iou_threshold)
    if len(indices) > 0:
        return indices.flatten()
    else:
        return []

# Function to calculate area of detected trash
def calculate_spillage_area(bboxes, frame_size):
    frame_area = frame_size[0] * frame_size[1]
    trash_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes)
    spillage_percentage = (trash_area / frame_area) * 100
    return spillage_percentage

# Function to process frame through ONNX model and get bounding boxes
def detect_trash(frame):
    # Preprocess the frame (assuming input size 640x640)
    input_size = (640, 640)
    img = cv2.resize(frame, input_size)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Transpose the image from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run the model inference
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img})

    # Extract the model output
    output = result[0]  # [1, 25200, 34]
    output = np.squeeze(output, axis=0)  # Remove batch dimension, shape becomes [25200, 34]

    bboxes = []
    confidences = []
    class_ids = []
    detected_classes = []

    for detection in output:
        # Extract the box coordinates and confidence
        x_center, y_center, width, height = detection[:4]
        confidence = detection[4]  # Objectness score
        class_scores = detection[5:]  # Class probabilities

        if confidence > CONFIDENCE_THRESHOLD:
            # Get the class with the highest score
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]

            if class_confidence > CONFIDENCE_THRESHOLD:
                # Convert the center x, y, width, height to x1, y1, x2, y2 (top-left and bottom-right)
                x1 = int((x_center - width / 2) * frame.shape[1])
                y1 = int((y_center - height / 2) * frame.shape[0])
                x2 = int((x_center + width / 2) * frame.shape[1])
                y2 = int((y_center + height / 2) * frame.shape[0])

                bboxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_classes.append(CLASSES[class_id])

    # Perform non-maximum suppression to filter overlapping boxes
    indices = non_max_suppression(bboxes, confidences, NMS_THRESHOLD)

    # Only keep the boxes after NMS
    final_bboxes = [bboxes[i] for i in indices]
    final_confidences = [confidences[i] for i in indices]
    final_class_ids = [class_ids[i] for i in indices]
    final_detected_classes = [detected_classes[i] for i in indices]

    return final_bboxes, final_confidences, final_class_ids, final_detected_classes

# Function to draw bounding boxes on the frame
def draw_boxes(frame, bboxes, class_ids, confidences):
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        class_id = class_ids[i]
        confidence = confidences[i]

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label and confidence
        label = f"{CLASSES[class_id]}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# WebSocket route with threshold-based alert system
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global detection_counter, alert_triggered

    await websocket.accept()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        await websocket.send_text("Error: Unable to open webcam.")
        await websocket.close()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform trash detection
        bboxes, confidences, class_ids, detected_classes = detect_trash(frame)

        # Print detected classes to terminal
        print("Detected Classes:", detected_classes)

        # Calculate spillage percentage
        spillage_percentage = calculate_spillage_area(bboxes, frame.shape[:2])

        # Update detection counter and check threshold
        if spillage_percentage > DETECTION_THRESHOLD:
            detection_counter += 1
        else:
            detection_counter = 0  # Reset if threshold is not exceeded

        # Trigger an alert if threshold is met
        if detection_counter >= ALERT_COUNT_THRESHOLD and not alert_triggered:
            alert_triggered = True
            detection_counter = 0  # Reset counter after alert
            await websocket.send_text("Alert: Trash spillage exceeded threshold!")

        # Draw boxes on the frame
        draw_boxes(frame, bboxes, class_ids, confidences)

        # Send frame and detected classes via WebSocket
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        await websocket.send_bytes(frame_bytes)
        await websocket.send_text(f"Detected Classes: {', '.join(detected_classes)}")

        # Small delay to prevent overloading
        await asyncio.sleep(0.03)

    cap.release()

# Serve a basic HTML page to display the webcam feed and detected classes
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_content = """
    <html>
        <head>
            <title>Trash Detection</title>
        </head>
        <body>
            <h1>Trash Detection</h1>
            <img id="video-feed" src="" alt="Webcam feed">
            <p id="detected-classes">Detected Classes: None</p>
            <script>
                const videoFeed = document.getElementById('video-feed');
                const detectedClasses = document.getElementById('detected-classes');
                const ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onmessage = function(event) {
                    if (typeof event.data === 'string') {
                        if (event.data.startsWith("Alert")) {
                            alert(event.data);  // Alert text message for alerts
                        } else {
                            detectedClasses.innerText = event.data;
                        }
                    } else {
                        const blob = new Blob([event.data], {type: 'image/jpeg'});
                        videoFeed.src = URL.createObjectURL(blob);
                    }
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Run the app
# To run, use: uvicorn this_file_name:app 
