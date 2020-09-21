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
from grafik import TransferModel


class TransferLearning:
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
        
        label = tk.LabelFrame(top, text='Transfer Learning', font=("Helvetica", 14))
        label2 = tk.LabelFrame(label, text='CNN Model', font=("Helvetica", 12))
        label3 = tk.LabelFrame(label, text='Machine Model', font=("Helvetica", 12))
        
        
        global cnn
        cnn = tk.StringVar()
        cnn.set('VGG16_Model')
        cnn_drop = tk.OptionMenu(label2, cnn,'VGG16_Model', 'AlexNet_Model','Kendi_Modelim' )
        
        global machine
        machine = tk.StringVar()
        machine.set('RandomForest')
        machine_drop = tk.OptionMenu(label3, machine,'RandomForest', 'SGDRegressor','LinearRegression' )
        
        frame4 = tk.Frame(label)
        batch_size_label = tk.Label(frame4, text="Batch Size : ", font=("Helvetica", 10))
        global e_batch_size
        e_batch_size = tk.Entry(frame4, width = 10)
        epoch_size_label = tk.Label(frame4, text="Epoch : ", font=("Helvetica", 10))
        global e_epoch
        e_epoch = tk.Entry(frame4, width = 10)
        fit_button = tk.Button(frame4,text="Egit", padx=30, pady=3, command = self.fit)
        
        
       
        predict_button = tk.Button(top,text="Tahmin", padx=30, pady=3, command = self.grafik)
        
        
        ################################################################################################
       
        
        
        
        
        label.pack(padx = 10, pady = 10,side = tk.TOP)
        
      
        predict_button.pack()
        label2.pack(side = tk.LEFT)
        cnn_drop.pack(padx = 10, pady = 10)
        label3.pack(side = tk.RIGHT)
        machine_drop.pack(padx = 10, pady = 10)
        frame4.pack()
        batch_size_label.pack(side = tk.LEFT)
        e_batch_size.pack(side = tk.LEFT)
        epoch_size_label.pack(side = tk.LEFT)
        e_epoch.pack(side = tk.LEFT)
        fit_button.pack(padx = 10, pady = 10, side = tk.BOTTOM)
       
        
    
    def grafik(self):
        TransferModel(path_ts,path_tr)
        
    def fit(self):
        
        X_train = []
        Y_train = []
        y = []
        if cnn.get() == 'VGG16_Model': 
            i = -1
            path_first = os.listdir(path_tr)
            for file_first in path_first:
                path_second = path_tr+"/"+file_first
                a = os.listdir(path_second)
                i+=1 
                for file_second in a:
                    y.append(int(i))
                    path_third = path_second+"/"+file_second
                    img = io.imread(path_third,0)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img,(224,224))
                    if(np.max(img) > 1):
                        img = img/255.
                    X_train.append(img)
            print(i)  
            X_train = np.array(X_train)  
            X_train = np.reshape(X_train,(-1,224,224,3))
            Y_train = to_categorical(y)
            print(len(X_train))
            print(Y_train.shape)
            model = self.get_vgg(Y_train.shape[1])
            machine_name = '_Epoch'+str(e_epoch.get())+'_Batch'+str(e_batch_size.get())
            model_name =  str(cnn.get())+'_Epoch'+str(e_epoch.get())+'_Batch'+str(e_batch_size.get())
            lrate = 0.001
            decay = lrate/int(e_epoch.get())
            sgd = SGD(lr = lrate, momentum=0.9, decay=decay, nesterov=False)
            if 1 == Y_train.shape[1]:
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            else:
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            history = model.fit(X_train,Y_train,validation_split=0.5, epochs=int(e_epoch.get()), batch_size=int(e_batch_size.get()))
            
            model.save_weights('./weight_cnn/'+ model_name +'.h5')
            model_json = model.to_json()
            with open("./cnn_model/"+ model_name +".json", "w") as json_file:
                json_file.write(model_json)
                
            layer_name = 'dense1'
            FC_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            features=np.zeros(shape=(X_train.shape[0],4096))
            for i in range(len(X_train)):
                img = np.expand_dims(X_train[i], axis=0)
                FC_output = FC_layer_model.predict(img)
                features[i]=FC_output
            feature_col=[]
            for i in range(4096):
                feature_col.append("f_"+str(i))
                i+=1
            train_features = pd.DataFrame(data=features,columns=feature_col)
            
            if machine.get() == 'RandomForest':
                rf = RandomForestClassifier(n_estimators = 20, random_state = 42,max_features=4)
                rf.fit(train_features,y)
                with open('./machine_model/'+cnn.get()+'_RandomForest'+machine_name+'.pkl', 'wb') as f:
                    cP.dump(rf, f)
        
            elif machine.get() == 'SGDRegressor':
                sr = SGDRegressor()
                sr.fit(train_features,y)
                with open('./machine_model/'+cnn.get()+'_SGDRegressor'+machine_name+'.pkl', 'wb') as f:
                    cP.dump(sr, f)
                    
            elif machine.get() == 'LinearRegression':
                lr = LinearRegression()
                lr.fit(train_features,y)
                with open('./machine_model/'+cnn.get()+'_LinearRegression'+machine_name+'.pkl', 'wb') as f:
                    cP.dump(lr, f)
             
                
        elif cnn.get() == 'AlexNet_Model':
            i = -1
            path_first = os.listdir(path_tr)
            for file_first in path_first:
                path_second = path_tr +"/"+file_first
                a = os.listdir(path_second)
                i+=1 
                for file_second in a:
                    y.append(int(i))
                    path_third = path_second+"/"+file_second
                    img = io.imread(path_third,0)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img,(224,224))
                    if(np.max(img) > 1):
                        img = img/255.
                    X_train.append(img)
                
            X_train = np.array(X_train)  
            X_train = np.reshape(X_train,(-1,224,224,3))
            Y_train = to_categorical(y)        
            model = self.get_alexnet(Y_train.shape[1])
            model.summary()
            lrate = 0.001
            decay = lrate/int(e_epoch.get())
            sgd = SGD(lr = lrate, momentum=0.9, decay=decay, nesterov=False)
            if i == Y_train.shape[1]:
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            else:
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            machine_name = '_Epoch'+str(e_epoch.get())+'_Batch'+str(e_batch_size.get())
            model_name =  str(cnn.get())+'_Epoch'+str(e_epoch.get())+'_Batch'+str(e_batch_size.get())
            
            history = model.fit(X_train,Y_train,validation_split=0.5, epochs=int(e_epoch.get()), batch_size=int(e_batch_size.get()))
            
            model.save_weights('./weight_cnn/'+ model_name +'.h5')
            model_json = model.to_json()
            with open("./cnn_model/"+ model_name +".json", "w") as json_file:
                json_file.write(model_json)
                
            layer_name = 'dense1'
            FC_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            features=np.zeros(shape=(X_train.shape[0],4096))
            for i in range(len(X_train)):
                img = np.expand_dims(X_train[i], axis=0)
                FC_output = FC_layer_model.predict(img)
                features[i]=FC_output
            feature_col=[]
            for i in range(4096):
                feature_col.append("f_"+str(i))
                i+=1
            train_features = pd.DataFrame(data=features,columns=feature_col)
            
            if machine.get() == 'RandomForest':
                rf = RandomForestClassifier(n_estimators = 20, random_state = 42,max_features=4)
                rf.fit(train_features,y)
                with open('./machine_model/'+cnn.get()+'_RandomForest'+machine_name+'.pkl', 'wb') as f:
                    cP.dump(rf, f)
        
            elif machine.get() == 'SGDRegressor':
                sr = SGDRegressor()
                sr.fit(train_features,y)
                with open('./machine_model/'+cnn.get()+'_SGDRegressor'+machine_name+'.pkl', 'wb') as f:
                    cP.dump(sr, f)
                    
            elif machine.get() == 'LinearRegression':
                lr = LinearRegression()
                lr.fit(train_features,y)
                with open('./machine_model/'+cnn.get()+'_LinearRegression'+machine_name+'.pkl', 'wb') as f:
                    cP.dump(lr, f)
                
        elif cnn.get() == 'Kendi_Modelim':
            i = -1
            path_first = os.listdir(path_tr)
            for file_first in path_first:
                path_second = path_tr + "/"+file_first
                a = os.listdir(path_second)
                i+=1 
                for file_second in a:
                    y.append(int(i))
                    path_third = path_second+"/"+file_second
                    img = io.imread(path_third,0)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img,(128,128))
                    if(np.max(img) > 1):
                        img = img/255.
                    X_train.append(img)
                
                
            model = self.get_kendi(Y_train.shape[1])
            lrate = 0.001
            decay = lrate/int(e_epoch.get())
            sgd = SGD(lr = lrate, momentum=0.9, decay=decay, nesterov=False)
            
            machine_name = '_Epoch'+str(e_epoch.get())+'_Batch'+str(e_batch_size.get())
            model_name =  str(cnn.get())+'_Epoch'+str(e_epoch.get())+'_Batch'+str(e_batch_size.get())
            
            if i == Y_train.shape[1]:
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            else:
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                
            history = model.fit(X_train,Y_train,validation_split=0.5, epochs=int(e_epoch.get()), batch_size=int(e_batch_size.get()))
            
            model.save_weights('./weight_cnn/'+ model_name +'.h5')
            model_json = model.to_json()
            with open("./cnn_model/"+ model_name +".json", "w") as json_file:
                json_file.write(model_json)
                
            layer_name = 'dense1'
            FC_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            features=np.zeros(shape=(X_train.shape[0],2048))
            for i in range(len(X_train)):
                img = np.expand_dims(X_train[i], axis=0)
                FC_output = FC_layer_model.predict(img)
                features[i]=FC_output
            feature_col=[]
            for i in range(2048):
                feature_col.append("f_"+str(i))
                i+=1
            train_features = pd.DataFrame(data=features,columns=feature_col)
            if machine.get() == 'RandomForest':
                rf = RandomForestClassifier(n_estimators = 20, random_state = 42,max_features=4)
                rf.fit(train_features,y)
                with open('./machine_model/'+cnn.get()+'_RandomForest'+machine_name+'.pkl', 'wb') as f:
                    cP.dump(rf, f)
        
            elif machine.get() == 'SGDRegressor':
                sr = SGDRegressor()
                sr.fit(train_features,y)
                with open('./machine_model/'+cnn.get()+'_SGDRegressor'+machine_name+'.pkl', 'wb') as f:
                    cP.dump(sr, f)
                    
            elif machine.get() == 'LinearRegression':
                lr = LinearRegression()
                lr.fit(train_features,y)
                with open('./machine_model/'+cnn.get()+'_LinearRegression'+machine_name+'.pkl', 'wb') as f:
                    cP.dump(lr, f)

    def get_vgg(self,nb_classes):
        vgg = VGG16(weights='imagenet', input_shape=(224,224,3), include_top=False)

        model = Sequential()
        
        for layer in vgg.layers:
            model.add(layer)
            
        for layer in model.layers:
            layer.trainable = False
            
        
        model.add(Flatten())
        model.add(Dense(4096,activation='relu',kernel_constraint=maxnorm(3), name = 'dense1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096,activation='relu',kernel_constraint=maxnorm(3),name = 'dense2'))
        model.add(Dropout(0.5))
        model.add(Dense(2,activation='softmax',name = 'outPut'))
        model.summary()
        return model
        
    
    
    def get_alexnet(self, nb_classes): 
        model = Sequential()
        model.add(Convolution2D(96, 11, 11, input_shape = (224,224,3), border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 5, 5, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(384, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(192, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(4096, init='glorot_normal', name = 'dense1'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, init='glorot_normal'))
        model.add(Activation('tanh'))
        return model
    
    
    
    
    def get_kendi(self,nb_classes):
        model = Sequential()
        model.add(Conv2D(32,(3,3),input_shape=(128,128,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
        model.add(Conv2D(32,(3,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64,(3,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128,(3,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2,2))) 
        model.add(Conv2D(256,(3,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2,2))) 
        model.add(Conv2D(512,(3,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2,2))) 
        model.add(Flatten())
        model.add(Dense(2048,activation='relu',kernel_constraint=maxnorm(3), name = 'dense1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096,activation='relu',kernel_constraint=maxnorm(3), name = 'dense2'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes,activation='softmax'))
        return model