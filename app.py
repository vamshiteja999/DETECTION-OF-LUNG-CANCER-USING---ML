from flask import Flask,render_template,request,session, url_for, redirect ,flash
#from flask_mysqldb import MySQL
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from sklearn import datasets
import pickle
import pymysql
import numpy as np
import math
import pickle
import collections

import numpy as np

import pandas as pd
import re
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf


from sklearn import preprocessing 
  



import cv2
import time
import numpy as np
import pandas as pd
import os


import pandas as pd
import cv2
import numpy as np



def dbConnection():
    connection = pymysql.connect(host="localhost", user="root", password="root", database="hyperphase")
    return connection

def dbClose():
    dbConnection().close()
    return


UPLOAD_FOLDER = 'static/uploadedimages'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','wav','mp3'}











app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'random string'
@app.route('/')
def index():
    return render_template('index.html')





@app.route("/uploadimage",methods=['POST','GET'])
def uploadimage():
    if request.method == "POST":
        import imutils
        emotions=[]
        file = request.files['email']
        from werkzeug.utils import secure_filename
        from werkzeug.datastructures import  FileStorage
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread("static/uploadedimages/"+str(filename))
        from tensorflow import keras
        
        from keras.preprocessing import image
        image_size=224
        #img = cv2.imread(path1+"//"+i)
        path="static/uploadedimages/"+"//"+str(filename)
        img = image.load_img(path, target_size=(image_size, image_size))
        x = image.img_to_array(img)
        print(type(x))
        img_4d=x.reshape(1,224,224,3)
        img_4d=img_4d/255
        model = keras.models.load_model('lung.hp5')
        #predictions = model.predict(img_4d)
       # print(predictions)
        predictions = model.predict(img_4d)
        print(predictions)
        pred=np.argmax(predictions[0])
        print(pred)
        #a=list(predictions[0]).index(max(predictions[0]))
        #print(a)
        dict1={0:'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',1:'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',2:'Normal',3:'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'}
        list1=['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib','large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa','squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']
        
            
        
     
        if pred==2:
            flash("No "+str("Cancer")+" Detected in Image")
        else:
            a=dict1[pred]
            flash(str(a)+" Cancer Detected in Image")
          
    path1="static/uploadedimages"
    image_names = os.listdir(path1)
    print(filename)
    print(str(filename))
    path1="uploadedimages/"+str(filename)
    print("path1",path1)
    return render_template('output.html',image_name=str(path1))
        
        
    
    #return render_template('frames.html')





@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    return render_template('frames.html')









@app.route('/register',methods=['POST','GET'] )
def register():
    if request.method == "POST":
        try:
            status=""
            fname = request.form.get("name")
            add = request.form.get("add")
            pno = request.form.get("pno")
            email = request.form.get("email")
            pass1 =  request.form.get("pass1")
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM userdetailes WHERE email = %s', (email))
            res = cursor.fetchone()
            #res = 0
            if not res:
                sql = "INSERT INTO userdetailes (name, address,phone,email,password) VALUES (%s,%s, %s, %s, %s)"
                val = (fname ,add ,pno ,email ,pass1)
                print(sql," ",val)
                cursor.execute(sql, val)
                con.commit()
                status= "success"
                return render_template("login.html")
            else:
                status = "Already available"
            #return status
            return redirect(url_for('index'))
        except Exception as e:
            print(e)
            print("Exception occured at user registration")
            return redirect(url_for('index'))
        finally:
            dbClose()
    return render_template('register.html')


@app.route('/login',methods=['POST','GET'])
def login():
    msg = ''
    if request.method == "POST":
        session.pop('user',None)
        mailid = request.form.get("email")
        password = request.form.get("pass1")
        #print(mobno+password)
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM userdetailes WHERE email = %s AND password = %s', (mailid, password))
        #a= 'SELECT * FROM userdetails WHERE mobile ='+mobno+'  AND password = '+ password
        print(result_count)
        #result_count=cursor.execute(a)
        # result = cursor.fetchone()
        if result_count>0:
            print(result_count)
            session['user'] = mailid
            return render_template("home.html")
        else:
            print(result_count)
            msg = 'Incorrect username/password!'
            return msg
    return render_template('login.html')


@app.route('/project.html')
def contact():
    return render_template('project.html')
@app.route('/analysis.html')
def analysis():
   return render_template('analysis.html')
@app.route('/modification.html')
def Modification():
    return render_template('modification.html')


if __name__=="__main__":
    app.run("0.0.0.0")