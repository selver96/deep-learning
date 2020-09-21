import tkinter as tk
import csv
import os
import cv2 as cv
import numpy as np
from skimage import io
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from keras.models import model_from_json
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

class ModelGoster:
    
    def __init__(self,test_path):
        self.init_main(test_path)
        global t
        t = test_path
        
    def init_main(self,path):
        top = tk.Toplevel()
        top.title("Modeller Frame")
        top.resizable(False, False)
        top.grab_set()
        top.focus_set()
        
        ###Result Start#######################################################
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
        
        label_result.pack(padx = 10, pady = 10,side = tk.TOP)
        accuracy_label.pack(padx = 10, pady = 10,side = tk.LEFT)
        accuracy.pack(pady = 10,side = tk.LEFT)
        sensitivity_label.pack(padx = 10, pady = 10,side = tk.LEFT)
        sensitivity.pack(pady = 10,side = tk.LEFT)
        specifity_label.pack(padx = 10, pady = 10,side = tk.LEFT)
        specifity.pack(padx = 10, pady = 10,side = tk.LEFT)
        ###Finish#############################################################
        
        ###Model Start########################################################
        label_model = tk.LabelFrame(top, text='Test Set', font=("Helvetica", 14))
        test_set = tk.Listbox(label_model)
        s = os.listdir(path)
        global resim_yol
        resim_yol = []
        for item in s:
            for i in os.listdir(path+'/'+item):
                test_set.insert(tk.END, i)
                resim_yol.append(str(path+'/'+item+'/'+i))
                print(str(path+'/'+item+'/'+i))
        test_set.bind('<<ListboxSelect>>',self.tek_predict)
        model_set__label = tk.LabelFrame(label_model, text='Models', font=("Helvetica", 14))
        m = os.listdir("./models")
        lists = []
        for item in m:
            lists.append(item)
        global model_click
        model_click = tk.StringVar()
        model_click.set(lists[0])
        model_set = tk.OptionMenu(model_set__label ,model_click ,*lists)
        detay = tk.Button(model_set__label,text="detay", padx=30, pady=3, command = self.figur_ciz)
        predict = tk.Button(model_set__label,text="predict", padx=30, pady=3, command = self.tum_predict)
        accury_result = tk.LabelFrame(label_model, text='Accury', font=("Helvetica", 14))
        loss_result = tk.LabelFrame(label_model, text='Loss', font=("Helvetica", 14))
        global accury
        accury = tk.Canvas(accury_result, width=386, height = 278, bg = 'blue')
        global loss
        loss = tk.Canvas(loss_result, width=386, height = 278, bg = 'blue')
        
        label_model.pack(padx = 10, pady = 10,side = tk.TOP)
        test_set.grid(row = 0, column = 0)
        model_set__label.grid(row = 0, column = 1)
        model_set.pack(side = tk.TOP)
        detay.pack(padx = 10, pady = 10, side = tk.BOTTOM)
        accury_result.grid(row = 0, column = 2)
        loss_result.grid(row = 0, column = 3)
        accury.pack()
        loss.pack()
        predict.pack()
        ###Finish#############################################################
        
        
        
    
     
    def tek_predict(self,evt):
        
        shape = []
        l = model_click.get()
        
        if len(l) <= 0:
            print('Model Seciniz')
        else:
            w = evt.widget
            m = l.split('_')
            yol = resim_yol[w.curselection()[0]]
            #print('Model Seciniz')
            
            img = cv.imread(yol)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if m[0] == 'VGG16':
                shape = [224,224]
            elif m[0] == 'AlexNet':
                shape = [224,224]
            elif m[0] == 'Kendi':
                shape = [128,128]
            img = cv.resize(img,(shape[0],shape[1]))
            img = np.expand_dims(img, axis=0)
            weight = model_click.get().split('.')
            json_file = open('./models/'+model_click.get(), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.summary()
            model.load_weights('./weight/'+weight[0]+'.'+weight[1]+'.h5')
            y_pred = model.predict(img)
            print(y_pred)
        
      
     
    def tum_predict(self):
        y_true = []
        X = []
        siniflar = []
        shape = []
        shape = [64,64]
        test = os.listdir(t)
        j = 0
        l = model_click.get()
        m = l.split('_')
        print(m)
        if m[0] == 'VGG16':
            shape = [224,224]
        elif m[0] == 'AlexNet':
            shape = [224,224]
        elif m[0] == 'Kendi':
            shape = [128,128]
            
        for file_first in test:
            path_second = t +"/"+file_first
            a = os.listdir(path_second) 
            for file_second in a:
                siniflar.append(j)
                path_third = path_second+"/"+file_second
                
                img = io.imread(path_third,0)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img,(shape[0],shape[1]))
                if(np.max(img) > 1):
                    img = img/255.
                X.append(img)
            j+=1
        
        X = np.array(X)  
        X = np.reshape(X,(-1,shape[0],shape[1],3)) 
        y_true = to_categorical(siniflar)
        siniflar = to_categorical(siniflar)
       
        
        weight = model_click.get().split('.')
        json_file = open('./models/'+model_click.get(), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.summary()
        model.load_weights('./weight/'+weight[0]+'.'+weight[1]+'.h5')
        
        result = model.predict(X)
        y_pred = np.array(result)
        print(len(y_pred))
        print(list(y_pred))
        print(list(y_true))
        confusion_matrix_output = confusion_matrix(y_true, y_pred) 
        print( confusion_matrix_output)
        self.analistic(siniflar, y_true, y_pred)
        
    def analistic(self,siniflar, y_dogru, y_tahmin):
    
        TP, TN, FP, FN = 0, 0, 0, 0,
        for i in range(len(y_dogru)):
            for j in range(len(siniflar)):
                if self.equal(y_dogru[i], siniflar[j]) == True and self.equal(y_tahmin[i], siniflar[j]) == True:
                    TP+=1
                elif self.equal(y_dogru[i], siniflar[j]) == True and self.equal(y_tahmin[i], siniflar[j]) == False:
                    TN+=1
                elif self.equal(y_dogru[i], siniflar[j]) == False and self.equal(y_tahmin[i], siniflar[j]) == True:
                    FP+=1
                elif self.equal(y_dogru[i], siniflar[j]) == False and self.equal(y_tahmin[i], siniflar[j]) == False:
                    FN+=1
        
        TP/=len(siniflar)
        FP/=len(siniflar)
        TN/=len(siniflar)
        FN/=len(siniflar)
        

        ac = ((TP+TN)/(TP+TN+FP+FN))*100
        sen= (TP/(TP+FN))*100
        spe= (TN/(TN+FP))*100
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

    def figur_ciz(self):
        t = model_click.get().split('.') 
        model_n ='./cv_result/'+t[0]+'.'+t[1]
        
        accuracy = []
        val_accuracy = []
        
        with open(model_n+'.csv',encoding='utf8') as csv_file:
            writer = csv.DictReader(csv_file)
            for i in writer:
                accuracy.append(i['accuracy'])
                val_accuracy.append(i['val_accuracy'])
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(accuracy)
        ax.plot(val_accuracy)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
        fig.savefig('accuracy.png', bbox_inches='tight')
        
        
        imgAccury = tk.PhotoImage(file = './accuracy.png') 
        accury.create_image(0,0,image = imgAccury, anchor=tk.NW)
        
        loss_ = []
        val_loss = []
        
        
        with open(model_n+'.csv',encoding='utf8') as csv_file:
            writer = csv.DictReader(csv_file)
            for i in writer:
                loss_.append(i['loss'])
                val_loss.append(i['val_loss'])
        fig2, ax2 = plt.subplots( nrows=1, ncols=1 )
        ax2.plot(loss_)
        ax2.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        fig2.savefig('loss.png', bbox_inches='tight')
        
       
        imgLoss = tk.PhotoImage(file = 'loss.png') 
        loss.create_image(0,0,image = imgLoss, anchor=tk.NW)
        a
        
    
    def decodeCategory(self,x):
        sayac = 0
        for i in range(len(x)):
            if x[i] == 0:
                sayac
            elif x[i] == 1:
                break
        return sayac
      
    
    