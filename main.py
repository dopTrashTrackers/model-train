
# # pip install fastapi uvicorn onnxruntime opencv-python python-multipart
# # pip install websockets

import numpy as np
import cv2
import onnxruntime as ort
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, Form
from fastapi.responses import JSONResponse
import io
from PIL import Image
import base64

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
    img = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Transpose the image from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)+
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
                x1 = int((x_center - width / 2))
                y1 = int((y_center - height / 2))
                x2 = int((x_center + width / 2))
                y2 = int((y_center + height / 2))

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

        # Ensure coordinates are within the frame bounds
        x1 = max(0, min(x1, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label and confidence
        label = f"{CLASSES[class_id]}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def fun():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform trash detection
        input_size = (640, 640)
        frame = cv2.resize(frame, input_size)
        bboxes, confidences, class_ids, detected_classes = detect_trash(frame)

        # Print detected classes to terminal
        print("Detected Classes:", detected_classes)

        # Calculate spillage percentage
        spillage_percentage = calculate_spillage_area(bboxes, frame.shape[:2])
        print("Spillage Percentage:", spillage_percentage)

        # Draw boxes on the frame
        frame = draw_boxes(frame, bboxes, class_ids, confidences)
        cv2.imshow("Trash Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

def funny(frame):

    if frame is None:
        return False

    # Perform trash detection
    input_size = (640, 640)
    frame = cv2.resize(frame, input_size)
    bboxes, confidences, class_ids, detected_classes = detect_trash(frame)

    if(detected_classes == []):
        return False

    # Print detected classes to terminal
    print("Detected Classes:", detected_classes)

    # Calculate spillage percentage
    spillage_percentage = calculate_spillage_area(bboxes, frame.shape[:2])
    print("Spillage Percentage:", spillage_percentage)

    # Draw boxes on the frame
    frame = draw_boxes(frame, bboxes, class_ids, confidences)

    return frame

@app.get("/")
async def get():
    return HTMLResponse(content="<h1>Welcome to the Trash Detection API</h1>")
    
@app.post("/detect-trash/")
async def detect_trash_api(file: UploadFile = File(...), postoffice_id: str = Form("")):
    print("Postoffice ID:", postoffice_id)
    # Read the image file in memory
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    frame = funny(frame)
    if frame is False:
        return JSONResponse(content={"status": False ,"message": "No trash detected."})

    # Convert the frame back to an image
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    return JSONResponse(content={"status": True, "image": img_base64})
