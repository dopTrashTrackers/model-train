
import cv2
import time
import requests
from datetime import datetime, timezone

API_URL = "https://modelapi-pniz.onrender.com/detect-trash/"  
CAMERA_INDEX = 0

postoffice_id = "-O6BFhj6STc4KsNkSgBC"
place = "0"

def capture_image():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while not cap.isOpened():
        print("Waiting for camera to warm up...")
        time.sleep(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from camera")
    _, img_encoded = cv2.imencode('.jpg', frame)
    return img_encoded.tobytes()

def send_image(image_bytes):
    files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
    time_str = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    data = {
        'postoffice_id': postoffice_id,
        'place': place,
        'time': time_str
    }
    response = requests.post(API_URL, files=files, data=data)
    return response

if __name__ == "__main__":
    while True:
        try:
            img_bytes = capture_image()
            resp = send_image(img_bytes)
            print(f"Sent image, status code: {resp.status_code}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(300)  # 5 minutes
