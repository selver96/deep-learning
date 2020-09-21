import tkinter as tk
from tkinter import filedialog as dialog
from modelGoster import ModelGoster
from transferLearning import TransferLearning
import os
import csv
import cv2 as cv
import numpy as np
from skimage import io
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.constraints import maxnorm
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from keras.models import model_from_json


class Anaframe(tk.Frame):
    
    def __init__(self, root):
        super().__init__(root)
        self.init_main()

    def init_main(self):
        label = tk.LabelFrame(root, text="Veri Setlerin Yolları", font=("Helvetica", 14))
        frame = tk.Frame(label)
        tag_name_label = tk.Label(frame, text='Sınıf :', font=("Helvetica", 12))
        global tag_label
        tag_label = tk.Label(frame, text='None')
        
        frame2 = tk.Frame(label)
        train_label = tk.Label(frame2, text="Train :   ", font=("Helvetica", 12))
        test_label = tk.Label(frame2,  text="Test :    ", font=("Helvetica", 12))
        global path_train_label 
        path_train_label = tk.Label(frame2, text="None                                           ")
        global path_test_label 
        path_test_label = tk.Label(frame2,  text="None                                           ")
        
        
        label2 = tk.LabelFrame(root, text='Network', font=("Helvetica", 14))
        
        frame3 = tk.Frame(label2)
        global clicked
        clicked = tk.StringVar()
        clicked.set('VGG16_Model')
        drop = tk.OptionMenu(frame3, clicked,'VGG16_Model', 'AlexNet_Model','Kendi_Modelim' )
        
        frame4 = tk.Frame(label2)
        batch_size_label = tk.Label(frame4, text="Batch Size : ", font=("Helvetica", 10))
        global e_batch_size
        e_batch_size = tk.Entry(frame4, width = 10)
        epoch_size_label = tk.Label(frame4, text="Epoch : ", font=("Helvetica", 10))
        global e_epoch
        e_epoch = tk.Entry(frame4, width = 10)
        dropout_val_label = tk.Label(frame4, text="Dropout Val : ", font=("Helvetica", 10))
        global e_dropout_val
        e_dropout_val = tk.Entry(frame4, width = 10)
        frame5 = tk.Frame(label2)
        frame6 = tk.Frame(root)
        
        train_button = tk.Button(frame2,text="Aç", padx=30, pady=3, command = self.open_file_train)
        test_button = tk.Button(frame2,text="Aç", padx=30, pady=3,  command = self.open_file_test)
        fit_button = tk.Button(frame5,text="Eğit", padx=30, pady=3, command = self.fit)
        model_goster_button = tk.Button(frame6,text="Modelleri Göster", padx=30, pady=3, command = self.open_ModelGoster_frame)
        transfer_learning_button = tk.Button(frame6,text="Transfer Learning", padx=30, pady=3, command = self.open_TransferLearning_frame)
        label.pack(padx = 10, pady = 10,side = tk.TOP)
        
        frame.pack(side = tk.TOP)
        frame2.pack(side = tk.BOTTOM)
        
        tag_name_label.grid(row = 0, column = 0)
        tag_label.grid(row = 0, column = 1)
        
        train_label.grid(row = 0, column = 0)
        path_train_label.grid(row = 0, column = 1)
        train_button.grid(row = 0, column = 2)
                
        test_label.grid(row = 1, column = 0)
        path_test_label.grid(row = 1, column = 1)
        test_button.grid(row = 1, column = 2)
        
        label2.pack(padx = 10, pady = 10,side = tk.BOTTOM)
        
        frame3.pack()
        drop.pack(side = tk.LEFT)
        
        frame4.pack()
        batch_size_label.pack(side = tk.LEFT)
        e_batch_size.pack(side = tk.LEFT)
        epoch_size_label.pack(side = tk.LEFT)
        e_epoch.pack(side = tk.LEFT)
        dropout_val_label.pack(side = tk.LEFT)
        e_dropout_val.pack(side = tk.LEFT)
        
        frame5.pack()
        fit_button.pack(side = tk.RIGHT)
        
        frame6.pack()
        model_goster_button.grid(row=0, column=0)
        transfer_learning_button.grid(row=0, column=1)


    def open_file_train(self):
        path = dialog.askdirectory()
        s = os.listdir(path)
        tag_label['text'] = str(len(s))
        path_train_label['text'] = path
    
    def open_file_test(self):
        path = dialog.askdirectory()
        path_test_label['text'] = path
    
    def open_ModelGoster_frame(self):
        ModelGoster(path_test_label['text'])
    
    def open_TransferLearning_frame(self):
        TransferLearning(path_test_label['text'], path_train_label['text'])

    def fit(self):
        train = path_train_label['text']
        val = float(e_dropout_val.get())
        siniflar = os.listdir(train)
        X_train = []
        Y_train = []

        if clicked.get() == 'VGG16_Model':
            i = 0
            for sinif in siniflar:
                path_sinif_resimler = train+"/"+sinif
                resimler = os.listdir(path_sinif_resimler)
                
                for resim in resimler:
                    Y_train.append(int(i))
                    path_resim = path_sinif_resimler+"/"+resim
                    img = io.imread(path_resim,0)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img,(224,224))
                    if(np.max(img) > 1):
                        img = img/255.
                    X_train.append(img)
                i+=1              
            X_train = np.array(X_train)  
            X_train = np.reshape(X_train,(-1,224,224,3))
            Y_train = to_categorical(Y_train)
            model = self.get_vgg(Y_train.shape[1], val)
            model_name =  str(clicked.get())+'_Epoch'+str(e_epoch.get())+'_Batch'+str(e_batch_size.get())+'_DropVal'+str(e_dropout_val.get())
            lrate = 0.001
            decay = lrate/int(e_epoch.get())
            sgd = SGD(lr = lrate, momentum=0.9, decay=decay, nesterov=False)
            if Y_train.shape[1] == 1:
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            else:
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            history = model.fit(X_train,Y_train,validation_split=0.5, epochs=int(e_epoch.get()), batch_size=int(e_batch_size.get()))
            
            model.save_weights('./weight/'+ model_name +'.h5')
            model_json = model.to_json()
            with open("./models/"+ model_name +".json", "w") as json_file:
                json_file.write(model_json)
            self.cv_writer(history,model_name)
                    
            
            
            
        elif clicked.get() == 'AlexNet_Model':
            i = 0
            for sinif in siniflar:
                path_sinif_resimler = train+"/"+sinif
                resimler = os.listdir(path_sinif_resimler)
                 
                for resim in resimler:
                    Y_train.append(int(i))
                    path_resim = path_sinif_resimler+"/"+resim
                    img = io.imread(path_resim,0)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img,(224,224))
                    if(np.max(img) > 1):
                        img = img/255.
                    X_train.append(img)
                i+=1              
            X_train = np.array(X_train)  
            X_train = np.reshape(X_train,(-1,224,224,3))
            Y_train = to_categorical(Y_train)
            
            model = self.get_alexnet(Y_train.shape[1], val)
            model_name =  str(clicked.get())+'_Epoch'+str(e_epoch.get())+'_Batch'+str(e_batch_size.get())+'_DropVal'+str(e_dropout_val.get())
            lrate = 0.001
            decay = lrate/int(e_epoch.get())
            sgd = SGD(lr = lrate, momentum=0.9, decay=decay, nesterov=False)
            if Y_train.shape[1] == 1:
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            else:
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            history = model.fit(X_train,Y_train,validation_split=0.5, epochs=int(e_epoch.get()), batch_size=int(e_batch_size.get()))
           
            model.save_weights('./weight/'+ model_name +'.h5')
            model_json = model.to_json()
            with open("./models/"+ model_name +".json", "w") as json_file:
                json_file.write(model_json)
            
            self.cv_writer(history,model_name)
        
        elif clicked.get() == 'Kendi_Modelim':
            i = 0
            for sinif in siniflar:
                path_sinif_resimler = train+"/"+sinif
                resimler = os.listdir(path_sinif_resimler)
                  
                for resim in resimler:
                    Y_train.append(int(i))
                    path_resim = path_sinif_resimler+"/"+resim
                    img = io.imread(path_resim,0)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img,(224,224))
                    if(np.max(img) > 1):
                        img = img/255.
                    X_train.append(img)
                i+=1              
            X_train = np.array(X_train)  
            X_train = np.reshape(X_train,(-1,224,224,3))
            Y_train = to_categorical(Y_train)
            model = self.get_kendi(Y_train.shape[1], val)
            model_name =  str(clicked.get())+'_Epoch'+str(e_epoch.get())+'_Batch'+str(e_batch_size.get())+'_DropVal'+str(e_dropout_val.get())
            lrate = 0.001
            decay = lrate/int(e_epoch.get())
            sgd = SGD(lr = lrate, momentum=0.9, decay=decay, nesterov=False)
            if Y_train.shape[1] == 1:
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            else:
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            history = model.fit(X_train,Y_train,validation_split=0.5, epochs=int(e_epoch.get()), batch_size=int(e_batch_size.get()))
            
            model.save_weights('./weight/'+ model_name +'.h5')
            model_json = model.to_json()
            with open("./models/"+ model_name +".json", "w") as json_file:
                json_file.write(model_json)
            
            self.cv_writer(history,model_name)
            
        
           
        
    def cv_writer(self,history,name):
        field = ['loss','accuracy','val_loss','val_accuracy']
        with open('./cv_result/'+name+'.csv',mode='w',encoding='utf8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field)
            writer.writeheader()
            for i in range(0,len(history.history['loss'])):
                info = {'loss':history.history['loss'][i],'accuracy':history.history['accuracy'][i],'val_loss':history.history['val_loss'][i],'val_accuracy':history.history['val_accuracy'][i]}
                writer.writerow(info)

    def categori(self,x):
        sayac = 0
        for i in range(len(x)):
            if x[i] == 0:
                sayac
            elif x[i] == 1:
                break
        return sayac
      
        

    def get_vgg(self,nb_classes,val):
        shape = (224,224,3)
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape = shape, activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(val))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(val))
        model.add(Dense(nb_classes, activation='softmax'))
        return model
    
    def get_alexnet(self, nb_classes,val):
        shape = (224,224,3)
        model = Sequential()
        model.add(Conv2D(filters=96, input_shape = shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(4096, input_shape=(224*224*3,)))
        model.add(Activation('relu'))
        model.add(Dropout(val))
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(val))
        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dropout(val))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        return model
    
    
    def get_kendi(self,nb_classes,val):
        shape = (224,224,3)
        model = Sequential()
        model.add(Conv2D(32,(3,3),input_shape = shape, padding='same',activation='relu',kernel_constraint=maxnorm(3)))
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
        model.add(Dropout(val))
        model.add(Dense(4096,activation='relu',kernel_constraint=maxnorm(3), name = 'dense2'))
        model.add(Dropout(val))
        model.add(Dense(nb_classes,activation='softmax', name = 'out'))
        return model

if __name__ == "__main__":
    root = tk.Tk()
    app = Anaframe(root)
    root.title('Deep Learning')
    root.resizable(False, False)
    root.mainloop()


























