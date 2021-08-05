import cv2
from tensorflow import keras
import os
from face_detection import detect_faces, print_face, crop_image
import numpy as np
import time

import sys 


RED = (0, 0, 255)
GREEN = (0, 255, 0)

def load_model(json, weights):


    json_file = open(json, 'r') 
    model_json = json_file.read()
    json_file.close()

    model = keras.models.model_from_json(model_json)


    model.load_weights(weights)
    return model

def detect_mask(image, model):
 
    x = []
    image_size = (224, 224)
    try:
        image = cv2.resize(image, image_size)
    except:
        return False
    x.append(image)
    image = np.array(x)
    return model.predict(image)[0][0] > 0.9

def update_text(image, num_masks, faces_detected):

    font = cv2.FONT_HERSHEY_TRIPLEX
    thickness = 1
    font_scale = 1
    org = (0, 30)

    if not faces_detected:
        image = cv2.putText(image, 'No face detected', org, font, font_scale, RED, thickness, cv2.LINE_AA)
    elif num_masks > 0:
        image = cv2.putText(image, 'Detected ' + str(num_masks) + ' masks', org, font, font_scale, GREEN, thickness, cv2.LINE_AA)
    else:
        image = cv2.putText(image, 'No Masks Detected', org, font, font_scale, RED, thickness, cv2.LINE_AA)
    
    return image

def main():

    cap = cv2.VideoCapture(0)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    model = load_model('mask_detection.json', 'mask_detection.h5')
    
    prev_time = 0
    faces = []
    masked = []
    masked_counter = 0

    while True:
        success, image = cap.read()
    
        if not success:
            print('Empty camera frame ignored')
            continue

        image = cv2.flip(image, 1)
        
        if time.time() - prev_time > 0.5:
            faces = detect_faces(image, height, width)
            masked_counter = 0
            masked.clear()
            
            for x in range(len(faces)):
                cropped_image = crop_image(image, faces[x])
                is_masked = detect_mask(cropped_image, model)
                masked.append(is_masked)

                if is_masked:
                    masked_counter += 1
                        
            prev_time = time.time()
        
        for x in range(len(faces)):
            image = print_face(image, faces[x], masked[x])
        image = update_text(image, masked_counter, len(faces) > 0)

        cv2.imshow("Mask Detection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()



main()
