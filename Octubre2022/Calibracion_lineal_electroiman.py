#!/usr/bin/env python
# coding: utf-8

# In[204]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from matplotlib import pyplot as plt


#Seleccionar voltaje máximo utilizada y distancia entre nucleos
print('selecciona rango del ajuste (0 - 9.5 V)')
rango_ajuste=float(input())
print('selecciona distancia entre nucleos. Multiplos de 0.5 ente 3.5 y 8.5 cm')
distancia_nucleos=float(input())
print('selecciona grado del polinomio')
grado_poly=int(input())
print('Directorio de trabajo (para leer y guardar archivos)')
directorio_medicion=str(input())#str('D:\FePt_BTO\Transporte\MEDIDAS CAC\YAG340-343\YAG340_BTO')
print('Para aplicar la calibracion a una medicion escriba 1, caso contrario otro entero')
control=int(input())

#abro archivo de medicion
if control==1: 
    print('Nombre del archivo de la medicion')
    archivo_medicion=str(input())#str('YAG340_BTO_0deg_sinP_200um_Hmax8.dat')
    abrir=directorio_medicion+'/'+archivo_medicion
    medicion=pd.read_csv(abrir)

#abro archivo caklibracion
d_str=str("%.1f"%float(distancia_nucleos))
archivo=str('20221014_'+d_str+'cm.dat')
prueba=pd.read_csv(archivo)
prueba_train=prueba[(prueba['Voltage(V)']<=rango_ajuste) & (prueba['Voltage(V)']>=-rango_ajuste)]
prueba_train.columns=['Voltage(V)','Field(G)']

X=-prueba['Voltage(V)'].to_numpy().reshape(-1,1)#values.reshape(-1,1)
Y=prueba['Field(G)'].to_numpy().reshape(-1,1)#values.reshape(-1,1)

#ajuste lineal
X_train=-prueba_train['Voltage(V)'].to_numpy().reshape(-1,1)#values.reshape(-1,1)
Y_train=prueba_train['Field(G)'].to_numpy().reshape(-1,1)#values.reshape(-1,1)
X_test=-prueba['Voltage(V)'].to_numpy().reshape(-1,1)#values.reshape(-1,1)
calibracion_lineal = LinearRegression()
calibracion_lineal.fit(X_train,Y_train)
Y_test=calibracion_lineal.predict(X_test)

#calculo error del ajuste
Delta=Y-Y_test
DF_Delta=pd.DataFrame()
DF_Delta['Voltage(V)']=pd.DataFrame(X)
DF_Delta['Field(G)']=pd.DataFrame(Delta)
DF_Delta_rango=DF_Delta[(DF_Delta['Voltage(V)']<=rango_ajuste) & (DF_Delta['Voltage(V)']>=-rango_ajuste)]
error=DF_Delta_rango['Field(G)'].max()

#Ajuste polinomio para calibracion
calibracion_poly=np.polyfit(X_train[:,0],Y_train[:,0],grado_poly)

#Error calibracion polinomio
p=np.poly1d(calibracion_poly)
Delta_poly=Y-p(X_test)
DF_Delta_poly=pd.DataFrame()
DF_Delta_poly['Voltage(V)']=pd.DataFrame(X)
DF_Delta_poly['Field(G)']=pd.DataFrame(Delta_poly)
DF_Delta_poly_rango=DF_Delta_poly[(DF_Delta_poly['Voltage(V)']<=rango_ajuste) & (DF_Delta_poly['Voltage(V)']>=-rango_ajuste)]
error_poly=DF_Delta_poly_rango['Field(G)'].max()

#Mostrar calibracion
pendiente=calibracion_lineal.coef_
ordenada=calibracion_lineal.intercept_
print('Ajuste lineal')
print("%.4f" % float(pendiente),'x +',"%.0f"% float(ordenada),)
print('Error en el rango seleccionado:',"%.0f"% error,'Oe')
print()
print('Polinomio:')
print(np.poly1d(calibracion_poly))
print('Error polinomio:', "%.0f"% float(error_poly), 'Oe')
print()
print('Campo maximo para',distancia_nucleos,'cm:', float(max(Y)),'Oe')
AjusteLineal={'Pendiente (Oe/V)':[float(pendiente)], 'Ordenada (Oe)':[float(ordenada)]}

#Guardar calibracion Lineal
DF_AjusteLineal=pd.DataFrame(AjusteLineal)
if control==1:
    archivo_calibracion_lineal=directorio_medicion+'/'+'CalibracionLineal_'+archivo_medicion
else:
    archivo_calibracion_lineal=directorio_medicion+'/'+'CalibracionLineal_'+'d='+str(distancia_nucleos)+'cm_'+'hasta'+str(rango_ajuste)+'V.dat'
DF_AjusteLineal.to_csv(archivo_calibracion_lineal)

#Guardar calibracion Polinomio
Polinomio=pd.DataFrame(calibracion_poly)
if control==1:
    archivo_calibracion_polinomio=directorio_medicion+'/'+'CalibracionPolinomio_'+archivo_medicion
else:
    archivo_calibracion_polinomio=directorio_medicion+'/'+'CalibracionPolinomio_'+'d='+str(distancia_nucleos)+'cm_'+'hasta'+str(rango_ajuste)+'V.dat'
Polinomio.to_csv(archivo_calibracion_polinomio)

#Aplico calibraciones a la medicion
if control==1:
    X_med=medicion['Voltaje(V)'].to_numpy().reshape(-1,1)
    Y_med_lineal=calibracion_lineal.predict(X_med)
    Y_med_poly=p(X_med)                     

#Guardo archivos calibrados 
                     #polinomio
if control==1:
    DF_HvsR_poly=pd.DataFrame()
    DF_HvsR_poly['Voltaje(V)']=pd.DataFrame(X_med)
    DF_HvsR_poly['Campo(Oe)']=pd.DataFrame(Y_med_poly)
    DF_HvsR_poly['Resistencia(Ohm)']=pd.DataFrame(medicion['Resistencia(Ohm)'])
    salida_poly=directorio_medicion+'/'+'V_H_R_poly'+archivo_medicion
    DF_HvsR_poly.to_csv(salida_poly)
                     #lineal
    DF_HvsR_lineal=pd.DataFrame()
    DF_HvsR_lineal['Voltaje(V)']=pd.DataFrame(X_med)
    DF_HvsR_lineal['Campo(Oe)']=pd.DataFrame(Y_med_lineal)
    DF_HvsR_lineal['Resistencia(Ohm)']=pd.DataFrame(medicion['Resistencia(Ohm)'])
    salida_lineal=directorio_medicion+'/'+'V_H_R_lineal'+archivo_medicion
    DF_HvsR_poly.to_csv(salida_lineal)

plt.figure('H vs V referencia')
plt.plot(X_test,prueba['Field(G)'].to_numpy().reshape(-1,1), label='calibracion')
plt.plot(X_test,calibracion_lineal.predict(X_test), label='lineal')
plt.plot(X_test,np.polyval(calibracion_poly,X_test),'--', label='polinomio')
plt.legend()

plt.figure('Error de la calibracion')
p=np.poly1d(calibracion_poly)
plt.plot(X,Y-p(X_test),'--', label='polinomio')
plt.plot(X,Y-calibracion_lineal.predict(X_test), label='lineal')
plt.axvline(rango_ajuste)
plt.axvline(-rango_ajuste)
plt.legend()

plt.figure('H vs R')
plt.plot(Y_med_poly,medicion['Resistencia(Ohm)'])
plt.ylabel('Resistencia (Ohm)')
plt.xlabel('Campo (Oe)')
plt.show()