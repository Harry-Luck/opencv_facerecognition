import face_recognition as fr
import cv2
import numpy as np
import os
import pickle
path = "./train/"
url = 'rtsp://admin:@122.176.110.134:554/ch0_0.264'
known_names = []
pos = []
pos_name = []
name1 = []
loop = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

with open('mypickle.pickle' ,'rb') as f:
    loaded_obj = pickle.load(f)
known_name_encodings = loaded_obj

# image = "./test/face3.jpg"
# image = cv2.imread(test_image)
img12 = cv2.VideoCapture(url)
# img12 = cv2.VideoCapture('test4.mp4')

while (True):
    _, image = img12.read()
    # image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image= cv2.resize(image, (800, 600))
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if loop < 2:
            name1 = []
            pos = []
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
                name1.append(name)
                pos.append([x, y, x+w, y+h])
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(image, name, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)
            # cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            # cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    for i in range(len(name1)):
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(image, (pos[i][0], pos[i][1]), (pos[i][2], pos[i][3]), (0, 0, 255), 2)
        cv2.putText(image, name1[i], (pos[i][0]+6, pos[i][1]-6), font, 1.0, (0, 0, 255), 1)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Result", image)
    loop += 1
    if loop >7:
        loop = 0
    # cv2.imwrite("./output.jpg", image)
    cv2.waitKey(2)
cv2.destroyAllWindows()
