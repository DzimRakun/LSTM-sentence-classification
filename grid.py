# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:05:21 2022

@author: Ognjen Pavic 401 2021
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dropout
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences 
from keras.callbacks import EarlyStopping

#ovaj kod predpostavlja improvizovani grid search algoritam koji donosi odluku na osnovu tacnosti predvidjanja nad trening i test podacima
#optimizer je ispitan rucno, pa zbog toga nije dodat kao jedan od parametara jer je ovom programu vec potrebno nekoliko sati da se izvrsi
#ispitivanje optimizatora na ovaj nacin bi samo udvostrucilo vreme trajanja

def encode(Y):
    output=[]
    for i in Y:
        if i=="negative":
            output.append(0)
        else:
            output.append(1)
    return output

def gridLSTM(trainx,trainy,testx,testy,vokabular,maxLength,epochs,batch,dropout,cells):
    model=Sequential()
    model.add(Embedding(vokabular,32,input_length=maxLength))
    model.add(LSTM(cells,batch_input_shape=(32,len(trainx),batch),dropout=dropout,recurrent_dropout=dropout))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=epochs/10)
    history=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=epochs,batch_size=batch, callbacks=[es])
    return history

def getResults(vloss,epochs,batch,dropout):
    minloss=100.0
    position=0
    for i in range(len(vloss)):
        current=vloss[i][len(vloss[i])-1]
        if(current<minloss):
            minloss=current
            position=i
        
    bestEpochs=epochs[position]
    bestBatch=batch[position]
    bestDropout=dropout[position]
    bestCells=dropout[position]
    file=open("parametri.txt","w")
    outputString= "najbolji rezultati dobijeni su za parametre :\nepochs: " 
    + str(bestEpochs) +"\nbatch: " + str(bestBatch) +"\ndopout: " 
    + str(bestDropout) + "\cells: "
    + str(bestCells) + "\ndobijena tacnost na trening skupu:"
    + str(round(vacc[position][len(vacc[position])-1],2))
    file.write(outputString) 
    file.close()

vokabular=20520
maxLength=51

colList=['recenica','konotacija']
read=pd.read_csv('recenice.csv',usecols=colList)
data=read.sample(frac=1)

encoded=encode(data['konotacija'])
data['konotacija']=encoded

split=0.8
length=len(data)

splitPoint=int(split*length)

X=data['recenica']
Y=data['konotacija']

Xencoded=[one_hot(i,vokabular) for i in X]
Xpadded=pad_sequences(Xencoded,maxlen=maxLength,padding='post')

trainx=Xpadded[0:splitPoint]
trainy=Y[0:splitPoint]
testx=Xpadded[splitPoint:length]
testy=Y[splitPoint:length]

cellsParams=[1,5,10,25]
epochsParams=[50,100,200]
batchParams=[32,64,128]
dropoutParams=[0.3,0.5,0.7]

epochs=[]
batch=[]
dropout=[]
acc=[]
loss=[]
vacc=[]
vloss=[]

counter=0
for i in epochsParams:
    for j in batchParams:
        for p in dropoutParams:
            for c in cellsParams:
                counter+=1
                print("\n\n")
                print(counter)
                print("\n\n")
                history=gridLSTM(trainx,trainy,testx,testy,vokabular,maxLength,i,j,p,c)
                acc.append(history.history['accuracy'])
                loss.append(history.history['loss'])
                vacc.append(history.history['val_accuracy'])
                vloss.append(history.history['val_loss'])
                epochs.append(i)
                batch.append(j)
                dropout.append(p)
    
#parametri se biraju po validation loss metrici (bira se trenutak kada je ona minimalna)
getResults(vloss,epochs,batch,dropout)
  
    
    
                
