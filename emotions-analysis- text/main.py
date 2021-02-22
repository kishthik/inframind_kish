from flask import Flask, render_template, request
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import nltk
#from nltk.download import stopwords
import seaborn as sb
import pandas as pd
from keras.applications import MobileNet
from keras.models import Sequential,Model 
from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
# nltk.download("wordnet", "whatever_the_absolute_path_to_myapp_is/nltk_data/")
classifier=pickle.load(open("sentiment_model1.pkl","rb"))
cv=pickle.load(open("sentiment_vectorizer.pkl","rb"))


lm=WordNetLemmatizer()

emotions={0:"Angry",1:"Sad",2:'Fear',3:"Surprise",4:"Happy",5:"Love"}

app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
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
            string="""
                Natural language process take following steps:\n

                    1. Removing special charecter: "{}" \n
                    2. Lowering all words: "{}"\n
                    3. Lemmatization- It usually refers to remove inflectional endings only 
                        and to return the base: "{}"\n
                    4. Making Corpus: "{}"\n
                    5. Making Bag of words: "{}"\n
                    6. Finally prediction: "{}"\n
            """.format(text,text_lower,clean_text,corpus,x,output)
            print(string)
            prob=pd.DataFrame({'Emotions': ['Angry',"Sad","Fear","Surprise","Joy","Love"], 'Probability': classifier.predict_proba(x)[0,:]})
            prob=prob.sort_values(by='Probability',ascending=False)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize =(10, 6)) 
  
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
# MobileNet is designed to work with images of dim 224,224
img_rows,img_cols = 224,224

MobileNet = MobileNet(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))

# Here we freeze the last 4 layers
# Layers are set to trainable as True by default

for layer in MobileNet.layers:
    layer.trainable = True

# Let's print our layers
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i),layer.__class__.__name__,layer.trainable)

def addTopModelMobileNet(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    
    top_model = Dense(1024,activation='relu')(top_model)
    
    top_model = Dense(512,activation='relu')(top_model)
    
    top_model = Dense(num_classes,activation='softmax')(top_model)

    return top_model

num_classes = 5

FC_Head = addTopModelMobileNet(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())

train_data_dir = '/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/fer2013/train'
validation_data_dir = '/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/fer2013/validation'

train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest'
                                   )

validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size = (img_rows,img_cols),
                        batch_size = batch_size,
                        class_mode = 'categorical'
                        )

validation_generator = validation_datagen.flow_from_directory(
                            validation_data_dir,
                            target_size=(img_rows,img_cols),
                            batch_size=batch_size,
                            class_mode='categorical')

from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

checkpoint = ModelCheckpoint(
                             'emotion_face_mobilNet.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(
                          monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=1,restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.0001)

callbacks = [earlystop,checkpoint,learning_rate_reduction]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy']
              )

nb_train_samples = 24176
nb_validation_samples = 3006

epochs = 25

history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples//batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples//batch_size)







