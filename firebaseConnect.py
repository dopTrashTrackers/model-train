import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import firestore
import numpy as np
import cv2
import onnxruntime as ort
import asyncio
from datetime import datetime

# Load the ONNX model
import os

onnx_model_path = "C:\\Users\\rohan\\Desktop\\ml-sih\\model-train\\bestm.onnx"

if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"Model file not found: {onnx_model_path}")

session = ort.InferenceSession(onnx_model_path)
session = ort.InferenceSession(onnx_model_path)


# Define the threshold for detection and NMS IoU
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


# Assuming 29 classes for the model
CLASSES = ['Aluminium foil', 'Bottle cap', 'Broken glass', 'Cigarette', 'Clear plastic bottle', 'Crisp packet', 'Cup', 'Drink can', 'Food Carton', 'Food container', 'Food waste', 'Garbage bag', 'Glass bottle', 'Lid', 'Other Carton', 'Other can', 'Other container', 'Other plastic bottle', 'Other plastic wrapper', 'Other plastic', 'Paper bag', 'Paper', 'Plastic bag wrapper', 'Plastic film', 'Pop tab', 'Single-use carrier bag', 'Straw', 'Styrofoam piece', 'Unlabeled litter']


# Function to perform non-maximum suppression (NMS)
def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, iou_threshold)
    if len(indices) > 0:
        return indices.flatten()
    else:
        return []

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


    # Perform non-maximum suppression to filter overlapping boxes
    indices = non_max_suppression(bboxes, confidences, NMS_THRESHOLD)


    # Only keep the boxes after NMS
    final_bboxes = [bboxes[i] for i in indices]
    final_confidences = [confidences[i] for i in indices]
    final_class_ids = [class_ids[i] for i in indices]


    return final_bboxes, final_confidences, final_class_ids


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


cred = credentials.Certificate("C:\\Users\\rohan\\Desktop\\ml-sih\\model-train\\credentials.json")
firebase_admin.initialize_app(cred,{'databaseURL': "https://sih2024-559e6-default-rtdb.firebaseio.com/"})


ref = db.reference('/postOffices')
post_office_ids = [
    '-O6AlggG6a7efBfMAB3z',
    '-O6AlggG6a7efBfMAB3z', 
    '-O6AmQckL3MJ36SREqXE',
    '-O6An786QIBmDsJ3QEAP',
    '-O6Ao4tTkT4kyVzE1DP0',
    '-O6AojnmE__MgHoW-jTW',
    '-O6ApFneoDjs91SmDVMP',
    '-O6ApSE9BRJZXBcC0ehw',
    '-O6Apd0RABobdrWo8joQ',
    '-O6Apq_C1CJ69QGMQG1h',
    '-O6Aq-sPBIJniGY1Elej',
    '-O6BFhj6STc4KsNkSgBC',
    '-O6BZj9AKWj3skiLAUwk',
    '-O6B_N_5cUpJb1z5Vd7D',
    '-O9l-zRIiEM-9aVak3WZ',

]

async def capture_and_detect():
    cap = cv2.VideoCapture(0)
    while True:


        ret, frame = cap.read()
        if not ret:
            break


        # Detect trash in the frame
        bboxes, confidences, class_ids = detect_trash(frame)


        # Draw bounding boxes on the frame
        draw_boxes(frame, bboxes, class_ids, confidences)
       
        
        
        # Display the frame
        cv2.imshow('Trash Detection', frame)


        # # Save the frame to Firebase
        # _, buffer = cv2.imencode('.jpg', frame)
        # image_data = buffer.tobytes()
        # ref.push({'image': image_data})

        total_detections = len(class_ids)

        # Get today's date in "YYYY-MM-DD" format
        today = datetime.now().strftime('%Y-%m-%d')

        
        for post_office_id in post_office_ids:
            
            garbage_data_ref = ref.child(f'{post_office_id}/garbageDetectionData/{today}')
            day_data = garbage_data_ref.get()

            if day_data:
              
                if 'detections' in day_data:
                    new_detections = day_data['detections'] + total_detections
                    garbage_data_ref.update({'detections': new_detections})
                else:
                    # Add the detections field if it doesn't exist
                    garbage_data_ref.update({'detections': total_detections})
            else:
                # If the date node doesn't exist, create it with the detections
                garbage_data_ref.set({'detections': total_detections})
        



        
        # Print the detected classes and confidences
        for i, class_id in enumerate(class_ids):
            class_ref = ref.child(f'-O6AlggG6a7efBfMAB3z/garbageTypeData/{class_id}')
            class_data = class_ref.get()
            print(class_data)##########
            if class_data and 'frequency' in class_data:
                class_ref.update({'frequency': class_data['frequency'] + 1})
            else:
                class_ref.update({'frequency': 1})


            print(f"Detected {CLASSES[class_id]} with confidence {confidences[i]}")

        
        # Check for 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # Wait for 5 seconds
        await asyncio.sleep(5)


    cap.release()
    cv2.destroyAllWindows()


# Run the capture and detect function
asyncio.run(capture_and_detect())
