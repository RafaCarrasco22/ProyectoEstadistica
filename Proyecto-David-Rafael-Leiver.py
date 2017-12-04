# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 20:05:54 2017

@author: cavr0
"""

import pandas as pd #Pandas es una librería de python destinada al análisis de datos, que proporciona unas estructuras de datos flexibles y que permiten trabajar con ellos de forma muy eficiente. Pandas ofrece las siguientes estructuras de datos: series, dataframes, etc.
from pandas_datareader import data, wb #Importamos datos directamente de la web, en este caso usaremos datos directos de /fred.stlouisfed.org/
import statsmodels.api as sm # Es un módulo de Python que proporciona clases y funciones para la estimación de muchos modelos estadísticos diferentes, así como para realizar pruebas estadísticas y explorar datos estadísticos
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt #Es una libreria de Python que nos permite graficar de manera rapida
import numpy as np #Agrega mayor soporte para vectores y matrices, constituyendo una biblioteca de funciones matemáticas de alto nivel para operar con esos vectores o matrices.

trabajo = pd.read_csv('./proyecto.csv') #con estas lineas de codigo extraemos nuestro archivo csv que contiene los datos a analizar
print(trabajo.head())

model1=smf.ols(formula='UNRATE~CIVPART',data=trabajo).fit() #Definimos nuestra relacion de nuestro modelo, en este caso sera el trabajo contra el desempleo

print("params", model1.params) #de manera automatica obtenemos los datos de los parametros para la ecuacion de la recta
print("pvalues", model1.pvalues) #Obtenemos el p-valor de esta relacion
print("R2", model1.rsquared)# Obtenemos el r cuadrada
print("Resumen", model1.summary(), "\n") #Nos imprime un tipo archivo con todos los estadisticos y demas que hay en esta relacion definida anteriormente

trabajo_pred=model1.predict(pd.DataFrame(trabajo['CIVPART'])) #Obtenemos los coeficientes predichos por el modelo y hacemos un nuevo frame con los datos de CIVPART
print("Valores Predichos en arreglo\n",trabajo_pred.head())


trabajo.plot(kind='scatter', x='CIVPART', y='UNRATE') #Para graficar nuestros datos como dispersion definimos el tipo de grafica, asi como que datos ocuparan el eje de las x y el eje de las y
plt.plot(pd.DataFrame(trabajo['CIVPART']),trabajo_pred,c='red',linewidth=2)
plt.show()   #Graficamos nuestra recta de regresion para ver los resultados


trabajo['trabajo_pred']=0.154515*trabajo['CIVPART']+-3.922353 # En una nueva columna del csv agregaremos los valores predichos por el modelo
trabajo['RSE']=(trabajo['UNRATE']-trabajo['trabajo_pred'])**2  #Calculamos el estadistica RSE a cada columna del vector
SSD=trabajo.sum()['RSE'] #Al final sumamos cada resultado
n = len(trabajo["UNRATE"]) #N contendra la magnitud de nuestro vector y
RSE=np.sqrt(SSD/(n-2)) #Calculamos el estadistico RSE
print("RSE", RSE) #Simplemente lo imprimimos
salesmean=np.mean(trabajo['UNRATE']) #Calculamos el promedio con numpy de  del vector UNRATE
print("salesmean", salesmean) #Imprimimos el resultado
error=RSE/salesmean #Calculamos el error dividiendo RSE entre el promedio 
print("error", error) #Imprimimos el error



trabajo['42d']=pd.rolling_mean(trabajo['UNRATE'],window=42) #A traves de rolling_mean podemos obtener los promedios moviles de cada serie de datos y primero lo calculamos con UNRATE y almacenamos los resultados en el vector, cada media sera de 42 numeros
trabajo['25d']=pd.rolling_mean(trabajo['UNRATE'],window=25)
trabajo[['UNRATE','42d','25d']].plot(figsize=(8,5))

trabajo['42di']=pd.rolling_mean(trabajo['CIVPART'],window=42)
trabajo['25di']=pd.rolling_mean(trabajo['CIVPART'],window=25)
trabajo[['CIVPART','42di','25di']].plot(figsize=(8,5))