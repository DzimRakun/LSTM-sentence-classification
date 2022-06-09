# -*- coding: utf-8 -*-
"""
Created on Sat Fri 18 23:00:21 2022

@author: Ognjen Pavic 401 2021
"""

import tensorflow as tf
from matplotlib import pyplot
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn import metrics
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences 

def raportirajMetrike(TP,TN,FP,FN,klasa):
    print()
    print("klasa" + str(klasa))
    senzitivnost = TP/(TP+FN) #recall
    specificnost = TN/(FP+TN)
    ppv = TP/(TP+FP) #precision
    npv = TN/(TN+FN)
    f1 = 2*(ppv*senzitivnost)/(ppv+senzitivnost)
    acc = (TP+TN)/(TP+FP+TN+FN)
    print("senzitivnost: " + str(round(senzitivnost,2)) + " specificnost: " + str(round(specificnost,2)))
    print("PPV: " + str(round(ppv,2)) + " NPV: " + str(round(npv,2)))
    print("f1 score: " + str(round(f1,2)))
    print("tacnost: " + str(round(acc,2)))

def uporediRezultate(real,predicted):
    print()
    #print("Dobijeni rezultati")
    #print(predicted)
    #print("Trazeni rezultati")
    #print(real)

    title = "confusion matrix : "
    
    classes = np.unique(real)
    fig, ax = pyplot.subplots()
    cm = metrics.confusion_matrix(real, predicted, labels=classes)
    sb.heatmap(cm, annot=True, fmt='d', cmap=pyplot.cm.Blues, cbar=False)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    ax.set_yticklabels(labels=classes, rotation=0)
    pyplot.show()
    
    #klasa 0
    TP=cm[0,0]
    TN=cm[1,1]
    FP=cm[1,0]
    FN=cm[0,1]
    raportirajMetrike(TP,TN,FP,FN,1)
    #klasa 1
    TP=cm[1,1]
    TN=cm[0,0]
    FP=cm[0,1]
    FN=cm[1,0]
    raportirajMetrike(TP,TN,FP,FN,2)

def iscrtavanjeGrafika(history):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.legend()
    pyplot.show()
    pyplot.close()
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.legend()
    pyplot.show()
    pyplot.close()

def encode(Y):
    output=[]
    for i in Y:
        if i=="negative":
            output.append(0)
        else:
            output.append(1)
    return output

def vocabulary(X):
    reci=[]
    count=0
    
    for i in X:
        temp=i.split()
        for j in temp:
            if j not in reci:
                reci.append(j)
                count+=1
    return count

def maxWords(X):
    maxWords = 0
    for i in X:
        temp=i.split()
        length=len(temp)
        if length > maxWords:
            maxWords=length
    return maxWords

def LSTMmodel(trainx, trainy, testx, testy,vokabular, maxLength):
    model=Sequential()
    model.add(Embedding(vokabular,32,input_length=maxLength)) 
    model.add(LSTM(10,batch_input_shape=(128,len(trainx),32), dropout=0.3, recurrent_dropout=0.3, activation='tanh')) #batch size, velicina trening skupa, duzina vektora koji ulazi
    model.add(Dense(1,activation='sigmoid'))
    #adam pokazuje bolje rezultate nego sgd
    opt=Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
    history=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=100, batch_size=64,callbacks=[es,mc])
    
    model=load_model('best_model.h5')
    
    iscrtavanjeGrafika(history)
    predicted=model.predict(testx)
    rectifiedPredicted=[]
    for i in predicted:
        if i>0.5:
            rectifiedPredicted.append(1)
        else:
            rectifiedPredicted.append(0)
    predicted=rectifiedPredicted
    uporediRezultate(testy,predicted)
    return model


    

colList=['recenica','konotacija']
read=pd.read_csv('recenice.csv',usecols=colList)
data=read.sample(frac=1)

encoded=encode(data['konotacija'])
data['konotacija']=encoded


X=data['recenica']
Y=data['konotacija']

#vokabular ima 20520 reci 
vokabular=vocabulary(X)
#najveca recenica se sastoji od 51 reci
maxLength=maxWords(X)

Xencoded=[one_hot(i,vokabular) for i in X]
Xpadded=pad_sequences(Xencoded,maxlen=maxLength,padding='post')

split=0.75
length=len(data)

splitPoint=int(split*length)

trainx=Xpadded[0:splitPoint]
trainy=Y[0:splitPoint]
testx=Xpadded[splitPoint:length]
testy=Y[splitPoint:length]

model=LSTMmodel(trainx,trainy,testx,testy,vokabular,maxLength)
model.save('final_model.h5')

print()
print()

string='This is the worst piece of media i have ever seen'
string=[string]
encoded=[one_hot(i,vokabular) for i in string]
padded=pad_sequences(encoded,maxlen=maxLength,padding='post')
prediction=model.predict_classes(padded)
if(prediction[0][0]==0):
    print('negativna konotacija')
else:
    print('pozitivna konotacija')

                