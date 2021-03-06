from flask import Flask, render_template, request
#import joblib
import os.path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import cv2
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,models,Model
import requests
from PIL import Image
#from tqdm.notebook import tqdm
#from sklearn.model_selection import train_test_split


# __name__ is equal to app.py
app = Flask(__name__)

# load model from modelh5
model = tf.keras.models.load_model('my_model2.h5')

#csv files for data
#imagPaths = pd.read_csv("imgPaths.csv")
#tabDat = pd.read_csv("tabData.csv")

#get image pixels
def get_img(path):
    d=cv2.imread(path)
    resize=cv2.resize(d , (224,224))
    return np.array(resize)

#load images from path
def load_images(paths):
    inputImages=[]
    for path in paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224))
        inputImages.append(image)
    return inputImages

def download_n_save(url):
    im = Image.open(requests.get(url, stream=True).raw) # Can be many different formats.
    pix = im.load()
    im.save('data.png')

#model.predict([get_img(imagPaths[:1][0]).reshape(-1,224,224,3),tabDat.iloc[:1,:].values.reshape(-1,16)])

@app.route("/")
def home():
    if (not os.path.exists('./log.txt')):
        f = open("log.txt", "w+")
        f.write("init\n")
        f.close()
    ip_address = request.remote_addr
    f = open("log.txt", "a+")
    now = datetime.now()
    f.write(f"Visit from: {ip_address} @ {now} \n")
    f.close()
    #return "Requester IP: " + ip_address
    return render_template('index.html')


#Takes an attrubuite as 0000 -> [0,0,0,0] -> append to main_list
def convertToLetters(attribute,main_list):
    for letter in list(attribute):
        main_list.append(letter)


# Takes an image and return its hue values
def get_moments(path='./data.png'):
    im = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    _,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(im)
    huMoments = cv2.HuMoments(moments)
    huMoments=[float(i) for i in huMoments]
    return huMoments



@app.route("/predict", methods=["POST"])
def predict():
    age =request.form['age']
    sex_male =request.form['sex_male']
    smoking=request.form['smoking']
    link=request.form['link']
    week=request.form['week']
    skew=request.form['skew']
    kurthosis=request.form['kurthosis']
    real_mean=request.form['real_mean']
    percentage=request.form['percentage']

    #adding them to create a query....
    prediction_list = []
    prediction_list.append(int(week))
    prediction_list.append(float(percentage))
    prediction_list.append(int(age))
    prediction_list.append(float(skew))
    prediction_list.append(float(kurthosis))
    prediction_list.append(float(real_mean))
    download_n_save(link)
    moments= get_moments()
    prediction_list.extend(moments)
    prediction_list.append(int(sex_male))
    prediction_list.extend([float(i) for i in list(smoking)])
    aa = np.array(prediction_list)
    #aa
    print(aa)
    print(get_img('./data.png').reshape(-1,224,224,3))
    regvalue = model.predict([get_img('./data.png').reshape(-1,224,224,3),aa.reshape(-1,16)])
    #print(prediction_list)




    the_result=f"Lung FVC will be {regvalue}"

    #print(model.predict([prediction_list])[0]=='Y')
    #print(the_result)
    return render_template("index.html", prediction=the_result)

    # return str(model.predict([prediction_list]))

    #return render_template("index.html", bike_count=request.form['Rank'])
	#bike_count = int(round(model.predict([[tempreture, humadity, wind, is_spring, is_summer, is_winter]])[0]))
	#return render_template("index.html", bike_count=bike_count)




if __name__ == "__main__":
    app.run()
