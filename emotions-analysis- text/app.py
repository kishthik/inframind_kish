from flask import Flask, render_template, request
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import sys
import logging
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
# nltk.download("wordnet", "whatever_the_absolute_path_to_myapp_is/nltk_data/")
classifier=pickle.load(open("sentiment_model1.pkl","rb"))
cv=pickle.load(open("sentiment_vectorizer.pkl","rb"))


lm=WordNetLemmatizer()

emotions={0:"Angry",1:"Sad",2:'Fear',3:"Surprise",4:"Happy",5:"Love"}

app=Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)
app.config['SEND_F http://127.0.0.1:5000/ILE_MAX_AGE_DEFAULT'] = 0
@app.route("/",methods=['GET'])

def Home():
    return render_template('index.html')
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = '0'
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/predict",methods=['POST'])
def predict():
    
    if request.method=='POST':
        review=request.form['review']
        if review.isnumeric():
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
            
        else:
            corpus=[]
            # review="i hate you you are vary bad"
            text=re.sub('[^a-zA-Z]'," ",review)
            text_lower=text.lower()
            lower_list=text_lower.split()
            lower_list=[lm.lemmatize(i) for i in lower_list if i not in set(stopwords.words('english'))]
            clean_text=" ".join(lower_list)
            corpus.append(clean_text)
            x=cv.transform(corpus).toarray()
            output=classifier.predict(x)
            output=emotions[output[0]]
            
            prob=pd.DataFrame({'Emotions': ['Angry',"Sad","Fear","Surprise","Joy","Love"], 'Probability': classifier.predict_proba(x)[0,:]})
            prob=prob.sort_values(by='Probability',ascending=False)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize =(10, 7)) 
  
            plt.bar(prob['Emotions'], prob['Probability']) 
            plt.savefig("static/people_photo/plot.png")

            path="static/people_photo/plot.png"
         
            return render_template('result.html',review=review,
                                   text=text,
                                   text_lower=text_lower,
                                   lower_list=lower_list,
                                   clean_text=clean_text,
                                   corpus=corpus,
                                   vector=x,
                                   output=output,ax=path
                                   )
    else:
        return render_template('index.html')
    


if __name__=="__main__":
    app.run()
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()









