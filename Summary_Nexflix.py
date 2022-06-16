#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:17:10 2022

@author: dayan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#selecicionamos el archivo csv para convertirlo en un data frame 
netflix =  pd.read_csv('netflix_titles.csv')

netflix["duracion"] = pd.to_numeric(netflix['duration'].replace('([^0-9]*)','', regex=True), errors='coerce')

# Variables independiente
x_netflix = netflix["release_year"][:1000]       
            
#Definimos la variable dependiente
y_netflix = netflix["duracion"][:1000]



# Estandarizacion de las variables
scale_x_netflix =(x_netflix-x_netflix.mean())/x_netflix.std()




#Train / Test
train_x_netflix = scale_x_netflix[:800]
train_y_netflix = y_netflix[:800]



test_x_netflix = scale_x_netflix[800:]
test_y_netflix = y_netflix[800:]



#mostrar nuetro conjunto de entrenamiento
plt.scatter(train_x_netflix,train_y_netflix)
plt.show()
#mostrar nuetro conjunto de prueba
plt.scatter(test_x_netflix,test_y_netflix)
plt.show()

#Modelo de Regresion Polinomial
poli_netflix =  np.poly1d(np.polyfit(train_x_netflix,train_y_netflix,9))
myline = np.linspace(-6,1,300)
poli_new_y = poli_netflix(myline)

#GRafica del modelo
plt.scatter(train_x_netflix,train_y_netflix)
plt.plot(myline,poli_new_y )
plt.show()


#Prediccion del modelo
print(poli_netflix(-1))

#R de relacion train y test
print ("Valor de r de relacion de train ")
r2_train = r2_score(train_y_netflix, poli_netflix(train_x_netflix))
print(r2_train)

print ("Valor de r de relacion de test ")
r2_test = r2_score(test_y_netflix, poli_netflix(test_x_netflix))
print(r2_test)

## utilizando la variable de duracion y release_year 
## procesada por el medio de la regresion polinomial existe una relacion muy baja 
## esto quiere decir que el modelo no es el indicado para este problema 
  
