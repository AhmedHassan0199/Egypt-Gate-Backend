from tkinter import filedialog
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


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
#CheckForImg('Nefertiti_Test.jpg','hog','encodings.pickle')