import face_recognition
import argparse
import pickle
import cv2
import numpy as np
from flask import Flask, request, make_response , send_file
from werkzeug.datastructures import FileStorage

app = Flask(__name__)

#ap = argparse.ArgumentParser()
#ap.add_argument("-e", "--encodings", required=True,
#                help="path to serialized db of facial encodings")
#ap.add_argument("-i", "--image", required=True,
#                help="path to input image")
#ap.add_argument("-d", "--detection-method", type=str, default="cnn",
#                help="face detection model to use: either `hog` or `cnn`")
#args = vars(ap.parse_args())
 
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

@app.route('/RecognizeOld' , methods=['GET', 'POST'])
def RecogizeOld():
    x=request.files['ImageFile'].read()
    
    #decodedImage=cv2.imdecode(x, cv2.IMREAD_COLOR)
    #cv2.imshow("Image", decodedImage)
    nparr = np.frombuffer(x, np.uint8)
    
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
 
 
    Returned_Name=CheckForImg(img,'cnn','DatasetEncodings.pickle')
    
    
    return Returned_Name

@app.route('/Recognize' , methods=['GET', 'POST'])
def Recogize():

    x=request.files['ImageFile'].read()
    
    #decodedImage=cv2.imdecode(x, cv2.IMREAD_COLOR)
    #cv2.imshow("Image", decodedImage)
    nparr = np.frombuffer(x, np.uint8)
    
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    DetectionMethod='cnn'

    EncodingsFilePath='DatasetEncodings.pickle'
    
    Image=img
    
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    data = pickle.loads(open(EncodingsFilePath, "rb").read())
    if data is not None:
        print("MALYAN")
    else:
        print(("NONE"))
    # load the input image and convert it from BGR to RGB
    image = Image #Req Image!!!!!!!! 
    if image.shape[1] > 1500 or image.shape[2] > 1500 :
        scale_percent=50
 
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
 
        # dsize
        dsize = (width, height)
 
        # resize image
        image = cv2.resize(image, dsize)
 
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb,
                                            model=DetectionMethod)
    print("Gab el Boxes")
    encodings = face_recognition.face_encodings(rgb, boxes)
    # initialize the list of names for each face detected
    names = []
    # loop over the facial embeddings
    print("Gab el encodings")
    for encoding in encodings:
        print("Sha8al fel loop")
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                encoding, tolerance=0.42)
        name = "Unknown"
    # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            print(counts)
            name = max(counts, key=counts.get)
 
        # update the list of names
        names.append(name)
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        print(name)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    # show the output image
 
 
 
    #retval, buffer = cv2.imencode('.jpg', image)
    #response = make_response(buffer.tobytes())
    #response.headers['Content-Type'] = 'image/png'
    #print(response)
    return name

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run()
#CheckForImg('Nefertiti_Test.jpg','hog','encodings.pickle')