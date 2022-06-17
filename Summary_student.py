#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:07:26 2022

@author: dayan alzate hernandez
Id:000502226
"""

#importamos las librerias 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


## convertimos el archivo csv en un data frame 
df= pd.read_csv('student_data.csv')
# creamos x y definimos las variables independiente 
x = df[['age','famrel']]
# cramos y y definimos la variables dependiente 
y = df[['traveltime']]


#mostrar nuetro conjunto de entrenamiento en diagrama de dispercion 
fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ax1.scatter(x["age"], x["famrel"],y, c='g', marker='o')
plt.show()


# estandarizamos los dato seleccionados 
scale = StandardScaler()
scaledx = scale.fit_transform(x)


# dividimos los datos 
#entrenamiento
train_x= scaledx[:315]
train_y= y[:315]
#prueba 
test_x= scaledx[315:]
test_y= y[315:]


## creamos nuetro modelo de regresion multiple
regre_mult= linear_model.LinearRegression()
regre_mult.fit(train_x,train_y)


#Prediccion del modelo 
print ("----- prediccion ----")
pred_traveltime = regre_mult.predict([test_x[0]])
print(pred_traveltime)



print ("valor de r para conjuntos de datos de entrenamiento ")
# R de relacion train y test para nuestro modelo 
r2_train = r2_score(train_y, regre_mult.predict(train_x))
print(r2_train)


regre_mult.fit( test_x,test_y)


print ("valor de r para conjuntos de datos de prueba ")
r2_test = r2_score(test_y, regre_mult.predict(test_x))
print(r2_test)


## se pude observar que el R2 arrojo balores muy bajos lo que significa que nuetro modelo no es el apropiado para este problema  









