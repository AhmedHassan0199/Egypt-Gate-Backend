import cvlib as cv
import face_recognition
from flask import Flask, request, make_response , send_file
import cv2

app=Flask(__name__)
@app.route('/')
def index():
    filename='D:\GPStuff\Dataset\Akhenaten\Akhenaten_#0\Akhenaten_4.jpg'
    image = cv2.imread(filename)
    #decodedImage=cv2.imdecode(x, cv2.IMREAD_COLOR)
    #cv2.imshow("Image", decodedImage)
    
    faces, confidences = cv.detect_face(image,threshold=0.9) 
    print(faces)
    #rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #boxes = face_recognition.face_locations(rgb,
    #                                        model='cnn')
    #print(boxes)
    for (top, right, bottom, left) in faces:
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return "eshta"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run()