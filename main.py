from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
import os
import requests

app = Flask(__name__)

# API Key for access
API_KEY = "eyJzdWIiOiIxMjM0NTY4ODkwIiwibmFtZSI6IkpqaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"

class FaceShapeDetector:
    def __init__(self):
        self._LBFModel = "data/lbfmodel.yaml"
        self._haarcascade = "data/lbpcascade_frontalface.xml"
        self._face_cascade = cv.CascadeClassifier(self._haarcascade)
        self._landmark_detector = cv.face.createFacemarkLBF()
        self._landmark_detector.loadModel(self._LBFModel)

    def detect_face_shape(self, image_path):
        image = cv.imread(image_path)
        if image is None:
            return {"status": "error", "status_code": "image_not_found", "message": "Image not found"}

        # Resize the image to a specific width (e.g., 800 pixels)
        target_width = 800
        ratio = target_width / image.shape[1]
        image = cv.resize(image, (target_width, int(image.shape[0] * ratio)))

        frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        faces = self._face_cascade.detectMultiScale(
            frame_gray, scaleFactor=1.2, minNeighbors=5)

        if len(faces) == 0:
            return {"status": "error", "status_code": "no_face_detect", "message": "No face detected"}

        for (x, y, w, h) in faces:
            _, landmarks = self._landmark_detector.fit(frame_gray, faces)
            landmark_points = [(int(x), int(y)) for landmark in landmarks for (x, y) in landmark[0]]
            
            if len(landmark_points) >= 68:
                cheek_left = landmark_points[1]
                cheek_right = landmark_points[15]
                chin_left = landmark_points[6]
                chin_right = landmark_points[10]
                nose_left = landmark_points[3]
                nose_right = landmark_points[13]
                eye_brow_left = landmark_points[17]
                eye_brow_right = landmark_points[26]
                bottom_chin = landmark_points[8]
                cheek_bone_right_down_one = landmark_points[11]

                cheek_distance = cheek_right[0] - cheek_left[0]
                top_jaw_distance = nose_right[0] - nose_left[0]
                forehead_distance = eye_brow_right[0] - eye_brow_left[0]
                chin_distance = chin_right[0] - chin_left[0]
                head_length = bottom_chin[1] - y
                
                jaw_width = top_jaw_distance
                jaw_right_to_down_one = cheek_bone_right_down_one[1] - nose_right[1]
                jaw_left_to_down_one = cheek_bone_right_down_one[0] - nose_left[0]

                jaw_angle = self._calculate_angle(jaw_width, jaw_right_to_down_one, jaw_left_to_down_one)

                return self.calculate_face_shape(cheek_distance, top_jaw_distance, forehead_distance, chin_distance, head_length, jaw_angle)
        
        return {"status": "error", "status_code": "unable_to_determine", "message": "Unable to determine face shape"}

    def calculate_face_shape(self, cheek, jaw, forehead, chin, head_length, jaw_angle):
        cheek_ratio = cheek / head_length
        jaw_ratio = jaw / head_length
        forehead_ratio = forehead / head_length
        chin_ratio = chin / head_length
        head_ratio = head_length / cheek

        result = "Unknown"

        if (0.8 <= cheek_ratio <= 1.0 and 0.7 <= jaw_ratio <= 0.8 and 0.6 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and head_ratio <= 1.25 and jaw_angle <= 50.0):
            result = "Round Face"
        elif (0.5 <= cheek_ratio <= 0.8 and 0.5 <= jaw_ratio <= 0.7 and 0.5 <= forehead_ratio <= 0.7 and
              0.2 <= chin_ratio <= 0.4 and 1.25 <= head_ratio <= 1.6 and jaw_angle > 50.0):
            result = "Oval Face"
        elif (0.5 <= cheek_ratio <= 0.8 and 0.5 <= jaw_ratio <= 0.8 and 0.5 <= forehead_ratio <= 0.8 and
              0.3 <= chin_ratio <= 0.4 and head_ratio >= 1.30 and jaw_angle > 55):
            result = "Rectangle Face"
        elif (0.7 <= cheek_ratio <= 0.99 and 0.7 <= jaw_ratio <= 0.8 and 0.6 <= forehead_ratio <= 0.99 and
              0.3 <= chin_ratio <= 0.5 and head_ratio <= 1.29 and jaw_angle < 55):
            result = "Square Face"
        elif (0.7 <= cheek_ratio <= 0.8 and 0.7 <= jaw_ratio <= 0.8 and 0.5 <= forehead_ratio <= 0.7 and
              0.3 <= chin_ratio <= 0.4 and 1.2 <= head_ratio <= 1.4):
            result = "Heart-Shaped Face"
        elif (0.7 <= cheek_ratio <= 0.8 and 0.7 <= jaw_ratio <= 0.8 and 0.6 <= forehead_ratio <= 0.8 and
              0.3 <= chin_ratio <= 0.4 and 1.2 <= head_ratio <= 1.4):
            result = "Diamond Shaped Face"

        return {"status": "success", "shape": result, "cheek_ratio": cheek_ratio, "jaw_ratio": jaw_ratio, "forehead_ratio": forehead_ratio, "chin_ratio": chin_ratio, "head_ratio": head_ratio, "jaw_angle": jaw_angle}

    def _calculate_angle(self, c, b, a):
        cosine_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        jaw_angle_degrees = np.degrees(np.arccos(cosine_angle))
        return jaw_angle_degrees

@app.route('/detect_face_shape', methods=['POST'])
def detect_face_shape():
    # Check for API key
    api_key = request.headers.get('Authorization')
    if api_key != f"Bearer {API_KEY}":
        return jsonify({"status": "error", "status_code": "unauthorized", "message": "Invalid API key"}), 403

    if 'image_url' not in request.json:
        return jsonify({"status": "error", "status_code": "no_image_url", "message": "No image URL provided"}), 400

    image_url = request.json['image_url']
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, "temp_image.jpg")

    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
        else:
            return jsonify({"status": "error", "status_code": "image_download_failed", "message": "Failed to download image"}), 400
    except Exception as e:
        return jsonify({"status": "error", "status_code": "request_failed", "message": str(e)}), 500

    detector = FaceShapeDetector()
    result = detector.detect_face_shape(image_path)

    # Delete the image after processing
    try:
        os.remove(image_path)
    except Exception as e:
        return jsonify({"status": "error", "status_code": "image_delete_failed", "message": "Failed to delete image"}), 500

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
