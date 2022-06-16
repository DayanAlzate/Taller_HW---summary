#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:17:10 2022

@author: dayan
"""

#Librerias incluidas
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

print("#---------------------------Actividad con BD Cars Regresion Polinomial-------------------------------------")
# Importe de la base de datos
cars_df = pd.read_csv('cars2.csv')

# Variables independiente
x_cars = cars_df["Weight"]

#Definimos la variable dependiente
y_cars = cars_df["CO2"]

# Estandarizacion de las variables
scale_cars = StandardScaler()
scale_x_cars =(cars_df["Weight"]-cars_df["Weight"].mean())/cars_df["Weight"].std()

#Train / Test
x_train_cars = scale_x_cars[:28]
y_train_cars = y_cars[:28]

x_test_cars = scale_x_cars[28:]
y_test_cars = y_cars[28:]


# Grafica de variables
plt.scatter(x_train_cars,y_train_cars)
plt.show()

plt.scatter(x_test_cars,y_test_cars)
plt.show()

#Modelo de Regresion Polinomial
poli_cars =  np.poly1d(np.polyfit(x_train_cars,y_train_cars,8))
myline = np.linspace(-3,2,90)
poli_new_y = poli_cars(myline)

#GRafica del modelo
plt.scatter(x_train_cars,y_train_cars)
plt.plot(myline,poli_new_y )
plt.show()


#Prediccion del modelo
print(poli_cars(0.4))

#R de relacion train y test
r2_train = r2_score(y_train_cars, poli_cars(x_train_cars))


poli_cars =  np.poly1d(np.polyfit(x_test_cars,y_test_cars,8))
myline = np.linspace(-3,2,90)
poli_new_y = poli_cars(myline)

#GRafica del modelo
plt.scatter(x_train_cars,y_train_cars)
plt.plot(myline,poli_new_y )
plt.show()

r2_test = r2_score(y_test_cars, poli_cars(x_test_cars))

print(r2_train)
print(r2_test)



print("#---------------------------Actividad con BD Cars Regresion Multiple-------------------------------------")
# Importe de la base de datos
cars_df = pd.read_csv('cars2.csv')

# Variables independiente
x_cars = cars_df[["Weight","Volume"]]

#Definimos la variable dependiente
y_cars = cars_df["CO2"]

# Estandarizacion de las variables
scale_cars = StandardScaler()
scale_x_cars =scale_cars.fit_transform(x_cars)

#Train / Test
x_train_cars = scale_x_cars[:28]
y_train_cars = y_cars[:28]

x_test_cars = scale_x_cars[28:]
y_test_cars = y_cars[28:]


#Modelo de Regresion Multiple
modelo_cars = linear_model.LinearRegression()
modelo_cars.fit(x_train_cars,y_train_cars)

#Prediccion del modelo
pred_scale_x_cars = modelo_cars.predict([x_test_cars[0]])
print(pred_scale_x_cars)

#R de relacion train y test
print ("Valor de r de relacion de train ")
r2_train = r2_score(y_train_cars, modelo_cars.predict(x_train_cars))
print(r2_train)

#Modelo de Regresion Multiple

modelo_cars.fit(x_test_cars,y_test_cars)
print ("Valor de r de relacion de test ")
r2_test = r2_score(y_test_cars, modelo_cars.predict(x_test_cars))
print(r2_test)
