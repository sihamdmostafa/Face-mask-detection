import cv2
import os
import mediapipe as mp

RED = (0, 0, 255)
GREEN = (0, 255, 0)

padding =  70

def detect_faces(image, height, width):

    mp_face_detection = mp.solutions.face_detection
    faces = []

    with mp_face_detection.FaceDetection(min_detection_confidence = 0.5) as face_detection:

        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections: 
                bounding_box = detection.location_data.relative_bounding_box
                face = [
                        int(bounding_box.xmin * width), 
                        int(bounding_box.ymin * height), 
                        int(bounding_box.width * width), 
                        int(bounding_box.height * height)
                        ]
                faces.append(face)
    return faces

def crop_image(image, face):

    x, y, w, h = face
    crop_img = image[y - padding // 2: y + h + padding // 2, x - padding // 2: x + h + padding // 2]
    return crop_img
    

def print_face(image, face, is_masked):

    x, y, w, h = face
    if is_masked:
        image = cv2.rectangle(image, (x - padding // 2, y - padding // 2), (x + w + padding // 2, y + h + padding // 2), GREEN)
    else:
        image = cv2.rectangle(image, (x - padding // 2, y - padding // 2), (x + w + padding // 2, y + h + padding // 2), RED)

    return image
