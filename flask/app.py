from flask import Flask, render_template, Response
import cv2
import threading
import argparse 
import datetime, time
import imutils
import face_recognition as fr
import cv2
import numpy as np
import os
import pickle
path = "./train/"
known_names = []
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

with open('mypickle.pickle' ,'rb') as f:
    loaded_obj = pickle.load(f)
known_name_encodings = loaded_obj
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
url = 'rtsp://admin:@122.176.110.134:554/ch0_0.264'
outputFrame = None
lock = threading.Lock()

def gen_frams():
    capture = cv2.VideoCapture(url)
    while True:
        _, image = capture.read()
    # image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image= cv2.resize(image, (800, 600))
        faces = face_cascade.detectMultiScale(image, 1.1, 4)
        for (x, y, w, h) in faces:
            image1 = image[y:y+h, x:x+w]
            # cv2.imshow("imaeg1", image1)
            face_locations = fr.face_locations(image1)
            face_encodings = fr.face_encodings(image1, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = fr.compare_faces(known_name_encodings, face_encoding)
                name = "unknown"
                face_distances = fr.face_distance(known_name_encodings, face_encoding)
                best_match = np.argmin(face_distances)

                if matches[best_match]:
                    name = known_names[best_match]
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)

        ret , buffer = cv2.imencode('.jpg',image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frams(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')
    # return '1231322'

if __name__ == "__main__":
    app.run(debug=True) 