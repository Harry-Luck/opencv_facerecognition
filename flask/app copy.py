from flask import Flask, render_template, Response
import cv2
import threading
import argparse 
import datetime, time
import imutils

from flask_cors import CORS
app = Flask(__name__)
CORS(app)
url = 'rtsp://admin:@122.176.110.134:554/ch0_0.264'
outputFrame = None
lock = threading.Lock()

def gen_frams():
    capture = cv2.VideoCapture(url)
    while True:
        bollean , frame = capture.read()
        # frame = cv2.resize(frame,(800, 600))
        ret , buffer = cv2.imencode('.jpg',frame)
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