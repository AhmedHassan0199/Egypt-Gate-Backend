from flask import Flask, request, make_response , send_file
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
import cv2
import face_recognition
import numpy as np
from numpy import load
from numpy import expand_dims
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import math
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

app = Flask(__name__)

if not firebase_admin._apps:
  cred = credentials.Certificate('egypt-gate-firestore.json') 
  default_app = firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/Recognize' , methods=['GET', 'POST'])
def get_face(): 
  x=request.files['ImageFile'].read()
  Lang=request.headers['Lang']
  print(Lang)
  nparr = np.frombuffer(x, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
  cv2.imwrite(('Result.jpg'),img)

  #filename="/content/drive/MyDrive/Prom Pictures/_MG_2053 as Smart Object-1.jpg"
  required_size=(160, 160)
    # load image from file
    
  flag=1
    #print(filename)

  image1 =img #cv2.imread(filename)#
  if image1.shape[1] > 1500 or image1.shape[0] > 1500:
    scale_percent = 30

    width = int(image1.shape[1] * scale_percent / 100)
    height = int(image1.shape[0] * scale_percent / 100)
    dsize = (width, height)
    image1 = cv2.resize(image1, dsize)


  rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

  boxes = face_recognition.face_locations(rgb, model='cnn',number_of_times_to_upsample=1)
  #image = image.convert('RGB')

    # convert to array
  pixels = asarray(rgb)



  if len(boxes)<1:
    flag=0
    face_array = asarray(rgb)
    print("no face found")
    return "No face found"

      
  y1=boxes[0][0]
  y2=boxes[0][2]
  x1=boxes[0][3]
  x2=boxes[0][1]
  face = pixels[y1 :y2, x1:x2]  
    # extract the face
    #cv2_imshow(face)
    # resize pixels to the model size
  image = Image.fromarray(face)
  image = image.resize(required_size)
  face_array = asarray(image)
  print("face found")
    #print(len(face_array))

  #savez_compressed('face3.npz', face_array)
  model = load_model('facenet_keras.h5')
  print('Loaded Model')
  face_emb,flag=get_face_embeding(model,face_array)
  if flag==0:
    return "Face embeding can not be found"

  king_name=get_face_name(face_emb)
 
  if Lang=='English':
    docToAccess='Pharaohs'
  else:
    docToAccess='Pharaohs_'+Lang
      
  doc_ref = db.collection(docToAccess).document(king_name.split('#')[0])
  doc = doc_ref.get()
  print(king_name)
  stringToReturn=""
  if doc.exists:
    x=doc.to_dict()
    for key in x:
      if not(key=="long-description"):
        stringToReturn=stringToReturn+(key)+'^'+str(x[key])
        stringToReturn=stringToReturn+"!"
    print(stringToReturn)
  else:
    stringToReturn='No Such Doc'
    print(stringToReturn)
  return stringToReturn
  

  #call
  #Returned_Image.split('#')[0]
def get_face_embeding (model, face_pixels):
  # scale pixel values
  face_pixels = face_pixels.astype('float64')
  flag=1
    # standardize pixel values across channels (global)
  mean, std = face_pixels.mean(), face_pixels.std()
  face_pixels = (face_pixels - mean) / std
    # transform face into one sample
  samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
  face_embedding = model.predict(samples)
  isNan=math.isnan(face_embedding[0][0])
  if isNan==True:
    flag=0
    print(face_embedding[0][0])
  return face_embedding[0],flag


def get_face_name (face_emb):
  data = load('kings-faces-embeddings_train&test.npz')
  trainX, trainy = data['arr_0'], data['arr_1']
  #print('Dataset: train=%d' % (trainX.shape[0]))
  # normalize input vectors
  normalized_vector = Normalizer(norm='l2')

  trainX = normalized_vector.transform(trainX)

  # label encode targets
  out_encoder = LabelEncoder()
  out_encoder.fit(trainy)
  trainy = out_encoder.transform(trainy)
  # fit model
  model = SVC(kernel='poly', probability=True)
  model.fit(trainX, trainy)

  samples = expand_dims(face_emb, axis=0)
  yhat_class = model.predict(samples)
  
  predict_names = out_encoder.inverse_transform(yhat_class)
  print('Predicted: ' ,(predict_names[0]))
  return predict_names[0]


#get_face()
app.run()