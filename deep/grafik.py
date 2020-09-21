import tkinter as tk
from tkinter import filedialog as dialog
import os
import csv
import cv2 as cv
import numpy as np
from skimage import io
import pandas as pd 
import keras
import pickle as cP
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, Convolution2D
from keras.layers import MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.constraints import maxnorm
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from keras.applications.vgg16 import VGG16
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from keras.models import model_from_json


class TransferModel:
    def __init__(self, test, train):
        self.init_main()
        global path_tr
        global path_ts
        path_tr = train
        path_ts = test
        print(path_tr)
        print(path_ts)
        
    def init_main(self):
        top = tk.Toplevel()
        top.title("Transfer Learning Frame")
        top.resizable(False, False)
        top.grab_set()
        top.focus_set()
        
        
        label2 = tk.LabelFrame(top, text='CNN Model', font=("Helvetica", 12))
        label3 = tk.LabelFrame(top, text='Machine Model', font=("Helvetica", 12))
        
        
        #Predict Form start
        label4 = tk.LabelFrame(top, text='Predict', font=("Helvetica", 14))
        
        cnn_frame = tk.LabelFrame(label4,text="CNN Modeller", font=("Helvetica", 14))
        
        global cnn_clicked
        c_l = []
        pathCNN = './cnn_model'
        for i in os.listdir(pathCNN):
            c_l.append(i)
        cnn_clicked = tk.StringVar()
        cnn_clicked.set(c_l[0])
        cnn_list = tk.OptionMenu(cnn_frame, cnn_clicked,*c_l)
        
        
        machine_frame = tk.LabelFrame(label4,text="Machine Modeller", font=("Helvetica", 14))
        global machine_clicked
        m_l = []
        pathMACHINE = './machine_model'
        for i in os.listdir(pathMACHINE):
            m_l.append(i)
        
        machine_clicked = tk.StringVar()
        machine_clicked.set(m_l[0])
        machine_list = tk.OptionMenu(machine_frame, machine_clicked,*m_l)
        
        
        predict_button = tk.Button(label4,text="Tahmin", padx=30, pady=3, command = self.predict)
        
        label4.pack(padx = 10, pady = 10,side = tk.BOTTOM)
        cnn_frame.pack(padx = 10, pady = 10,side = tk.LEFT)
        machine_frame.pack(padx = 10, pady = 10,side = tk.RIGHT)
        cnn_list.pack()
        machine_list.pack()
        predict_button.pack()
        #Predict Form Finish
        
        ################################################################################################
        
        
        #Result Form Start
        label_result = tk.LabelFrame(top, text='Result', font=("Helvetica", 14))
        accuracy_label = tk.Label(label_result, text="Accuracy : ", font=("Helvetica", 12))
        global accuracy
        accuracy = tk.Label(label_result, text="None", font=("Helvetica", 10))
        
        sensitivity_label = tk.Label(label_result, text="Sensitivity : ", font=("Helvetica", 12))
        global sensitivity
        sensitivity = tk.Label(label_result, text="None", font=("Helvetica", 10))
        
        specifity_label = tk.Label(label_result, text="Specifity : ", font=("Helvetica", 12))
        global specifity
        specifity = tk.Label(label_result, text="None", font=("Helvetica", 10))
        
        label_result.pack(padx = 10, pady = 10,side = tk.BOTTOM)
        accuracy_label.pack(padx = 10, pady = 10,side = tk.LEFT)
        accuracy.pack(pady = 10,side = tk.LEFT)
        sensitivity_label.pack(padx = 10, pady = 10,side = tk.LEFT)
        sensitivity.pack(pady = 10,side = tk.LEFT)
        specifity_label.pack(padx = 10, pady = 10,side = tk.LEFT)
        specifity.pack(padx = 10, pady = 10,side = tk.LEFT)
        #Result Form Finish
        
        
        
        
        
        
       
       
        
    
    def predict(self):
        cnn = cnn_clicked.get().split('_')
        machine = machine_clicked.get().split('_')
        if cnn[0] == 'VGG16':
            X_test = []
            Y_test = []
            siniflar = []
            i = -1
            path_first = os.listdir(path_ts)
            for file_first in path_first:
                path_second = path_tr+"/"+file_first
                a = os.listdir(path_second)
                i+=1 
                siniflar.append(i)
                for file_second in a:
                    Y_test.append(int(i))
                    path_third = path_second+"/"+file_second
                    img = io.imread(path_third,0)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img,(224,224))
                    if(np.max(img) > 1):
                        img = img/255.
                    X_test.append(img)
                
            X_test = np.array(X_test)  
            X_test = np.reshape(X_test,(-1,224,224,3))
            
            weight = cnn_clicked.get().split('.')
            json_file = open('./cnn_model/'+cnn_clicked.get(), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.summary()
            model.load_weights('./weight_cnn/'+weight[0]+'.h5')
            
            
            
            layer_name = 'dense1'
            FC_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            features=np.zeros(shape=(X_test.shape[0],4096))
            for i in range(len(X_test)):
                img = np.expand_dims(X_test[i], axis=0)
                FC_output = FC_layer_model.predict(img)
                features[i]=FC_output
            feature_col=[]
            for i in range(4096):
                feature_col.append("f_"+str(i))
                i+=1
            test_features = pd.DataFrame(data=features,columns=feature_col)
            
            if machine[0] == 'RandomForest':
                with open('./machine_model/'+ machine_clicked.get(), 'rb') as f:
                    rf = cP.load(f)
                y_pred = rf.predict(test_features)
                self.analistic(siniflar, Y_test, y_pred)
                
            elif machine[0] == 'SGDRegressor':
                sr = cP.loads('./machine_model/'+machine_clicked.get())
                y_pred = sr.predict(test_features)
                self.analistic(siniflar, Y_test, y_pred)
            
            elif machine[0] == 'LinearRegression':
                lr = cP.loads('./machine_model/'+machine_clicked.get())
                y_pred = lr.predict(test_features)
                self.analistic(siniflar, Y_test, y_pred)
      
        elif cnn[0] == 'AlexNet':
            X_test = []
            Y_test = []
            siniflar = []
            i = -1
            path_first = os.listdir(path_ts)
            for file_first in path_first:
                path_second = path_tr+"/"+file_first
                a = os.listdir(path_second)
                i+=1 
                siniflar.append(i)
                for file_second in a:
                    Y_test.append(int(i))
                    path_third = path_second+"/"+file_second
                    img = io.imread(path_third,0)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img,(224,224))
                    if(np.max(img) > 1):
                        img = img/255.
                    X_test.append(img)
                
            X_test = np.array(X_test)  
            X_test = np.reshape(X_test,(-1,224,224,3))
            
            weight = cnn_clicked.get().split('.')
            json_file = open('./cnn_model/'+cnn_clicked.get(), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.summary()
            model.load_weights('./weight_cnn/'+weight[0]+'.h5')
            model.summary()
            
            layer_name = 'dense1'
            FC_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            features=np.zeros(shape=(X_test.shape[0],4096))
            for i in range(len(X_test)):
                img = np.expand_dims(X_test[i], axis=0)
                FC_output = FC_layer_model.predict(img)
                features[i]=FC_output
            feature_col=[]
            for i in range(4096):
                feature_col.append("f_"+str(i))
                i+=1
            test_features = pd.DataFrame(data=features,columns=feature_col)
            
            if machine[0] == 'RandomForest':
                print('RandomForest')
                with open('./machine_model/'+machine_clicked.get(), 'rb') as f:
                    rf = cP.load(f)
                y_pred = rf.predict(test_features)
                print(y_pred)
                self.analistic(siniflar, Y_test, y_pred)
                
            elif machine[0] == 'SGDRegressor':
                sr = cP.loads('./machine_model/'+machine_clicked.get())
                y_pred = sr.predict(test_features)
                self.analistic(siniflar, Y_test, y_pred)
            
            elif machine[0] == 'LinearRegression':
                lr = cP.loads('./machine_model/'+machine_clicked.get())
                y_pred = lr.predict(test_features)
                self.analistic(siniflar, Y_test, y_pred)
        
        elif cnn[0] == 'Kendi':
            X_test = []
            Y_test = []
            siniflar = []
            i = -1
            path_first = os.listdir(path_ts)
            for file_first in path_first:
                path_second = path_tr+"/"+file_first
                a = os.listdir(path_second)
                i+=1 
                siniflar.append(i)
                for file_second in a:
                    Y_test.append(int(i))
                    path_third = path_second+"/"+file_second
                    img = io.imread(path_third,0)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img,(224,224))
                    if(np.max(img) > 1):
                        img = img/255.
                    X_test.append(img)
                
            X_test = np.array(X_test)  
            X_test = np.reshape(X_test,(-1,128,128,3))
            
            json_file = open('./cnn_model/'+cnn_clicked.get(), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights('./weight_cnn/'+cnn_clicked.get())
            model.summary()
            
            layer_name = 'dense1'
            FC_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            features=np.zeros(shape=(X_test.shape[0],4096))
            for i in range(len(X_test)):
                img = np.expand_dims(X_test[i], axis=0)
                FC_output = FC_layer_model.predict(img)
                features[i]=FC_output
            feature_col=[]
            for i in range(4096):
                feature_col.append("f_"+str(i))
                i+=1
            test_features = pd.DataFrame(data=features,columns=feature_col)
            
            if machine[0] == 'RandomForest':
                rf = cP.loads('./machine_model/'+machine_clicked.get())
                y_pred = rf.predict(test_features)
                self.analistic(siniflar, Y_test, y_pred)
                
            elif machine[0] == 'SGDRegressor':
                sr = cP.loads('./machine_model/'+machine_clicked.get())
                y_pred = sr.predict(test_features)
                self.analistic(siniflar, Y_test, y_pred)
            
            elif machine[0] == 'LinearRegression':
                lr = cP.loads('./machine_model/'+machine_clicked.get())
                y_pred = lr.predict(test_features)
                self.analistic(siniflar, Y_test, y_pred)
                
    def analistic(self,siniflar, y_dogru, y_tahmin):
    
        TP, TN, FP, FN = 0, 0, 0, 0,
        for i in range(len(y_dogru)):
            for j in range(len(y_dogru)):
                if y_dogru[i] == y_dogru[j] and y_dogru[i] == y_tahmin[j]:
                    TP+=1
                elif y_dogru[i] == y_dogru[j] and y_dogru[i] != y_tahmin[j]:
                    TN+=1
                elif y_dogru[i] != y_dogru[j] and y_dogru[i] == y_tahmin[j]:
                    FP+=1
                elif y_dogru[i] != y_dogru[j] and y_dogru[i] != y_tahmin[j]:
                    FN+=1
        TP/=len(y_dogru)
        FP/=len(y_dogru)
        TN/=len(y_dogru)
        FN/=len(y_dogru)
        
        print(str(TP)+" "+str(TN)+" "+str(FP)+" "+str(FN))
        try:
            ac = ((TP+TN)/(TP+TN+FP+FN))*100
        except ZeroDivisionError:
            ac = 0
        try:
            sen= (TP/(TP+FN))*100
        except ZeroDivisionError:
            sen = 0
        try:
            spe = (TN/(TN+FP))*100
        except ZeroDivisionError:
            spe = 0
        accuracy['text'] = str(ac)
        sensitivity['text'] = str(sen)
        specifity['text'] = str(spe)
     
    def equal(self, x, y):
        sayac = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                sayac+=1
        if len(x) == sayac:
            return True
        else:
            return False
