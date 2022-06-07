# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:47:18 2022

@author: Ognjen Pavic 401 2021
"""
import pandas as pd


#ovaj deo koda namenjen je filtriranju pocetnog skupa i kreiranju novog skupa sa kojim je lakse raditi
file=open("MR/rt-polarity.neg",'r')
negative=file.read()
file=open("MR/rt-polarity.pos",'r')
positive=file.read()

special=['.',',',':',';','=','+','-','*','/','\\','\'','\"','!','?','_','@','#','$','%','^','&','(',')','[',']','{','}']
for spec in special:
    negative=negative.replace(spec,"")
    positive=positive.replace(spec,"")
    

dataset=[]
output=[]

negativeS=negative.split('\n')
positiveS=positive.split('\n')

for x in negativeS:
    dataset.append(x.strip())
    output.append('negative')
    
for x in positiveS:
    dataset.append(x.strip())
    output.append('positive')
    

cols=['recenica','konotacija']
df=pd.DataFrame({'recenica' : dataset,
                 'konotacija' : output})
filename='recenice.csv'
df.to_csv(filename)

    
