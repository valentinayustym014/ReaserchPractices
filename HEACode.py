# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# HEA

import pandas as pd
import numpy as np
import time
import random

df = pd.read_excel('Datos.xlsx', "I20", header= None)

# Cantidad
numVar = df.iloc[0][0]
numRes = df.iloc[0][1]
numObj = df.iloc[0][2]


# Data Frames 

dfRes  = df.iloc[1:int(numRes)+1]
dfResA = dfRes.iloc[:,:-1]
dfResB = dfRes.iloc[:,-1]

dfObj = df.iloc[int(numRes)+1:]
dfObj2 = dfObj.iloc[:,:-1]
dfObj1 = dfObj2.copy()

# Identified

x = np.zeros(int(numVar))

a = dfRes.iloc[:,:-1].copy()

b =dfResB.copy()

CdfResB= dfResB.copy()

ob = dfObj1.copy()
  
# Initial solution obtained from constructive method 

dfconst = pd.read_excel('AA2 - Metodo 1 GRASP .xlsx', "I20", header= None)   

NumSolDisponible = dfconst.iloc[0][0]
    

def dominancia(a, b): #Dominance-test 

    va = np.dot(dfObj2, a)
        
    vb = np.dot(dfObj2, b)
        
    cont = 0 
    noagregar = 0
    
    if np.any(va >= vb) == True:
        cont += 1
    if np.all(va < vb) == True: 
        noagregar += 1
    if cont>=1 and noagregar== 0: # Osea A domina a B 
        return True
    else: 
        return False      
    
    
def difference(sola, solb): 
    
    diferencia = 0 
    
    for i in range(len(sola)):
        if sola[i]-solb[i] != 0: 
            diferencia +=1
            
    return diferencia
    

def seleccion(soluciones):
    
    if len(soluciones) == 1: 
        
        return soluciones[0], soluciones[0]
    
    if len(soluciones) == 2:
        
        randoms = random.sample(range(0, int(len(soluciones))), 2)
        rand1 = randoms[0]
        rand2 = randoms[1]
        return soluciones[rand1], soluciones[rand2]
    
    else: 
        randoms = random.sample(range(0, int(len(soluciones))), 3)
        
        rand1 = randoms[0]
        rand2 = randoms[1]
        rand3 = randoms[2]
        
        randoms = [rand1, rand2, rand3]

        indexMaxi = 0
        maxi = 0 
        diff = []
    
        for i in range(3): 
            for j in range(3): 
                diff.append(difference(soluciones[randoms[i]], soluciones[randoms[j]]))
        
        maxi = max(diff)
        indexMaxi = diff.index(maxi)    
                
        if indexMaxi ==1: 
            return soluciones[randoms[0]], soluciones[randoms[1]]
        if indexMaxi ==2: 
            return soluciones[randoms[0]], soluciones[randoms[2]]
        if indexMaxi ==5: 
            return soluciones[randoms[1]], soluciones[randoms[2]]
    

def crossOver(solucion1, solucion2):  # Cross by point

    randomPercentage = int(int(numVar)*(random.randint(0,100)/100))
    
    parte1 = solucion1[0:randomPercentage]
    parte2 = solucion2[randomPercentage:]
    
    parte3 = solucion2[0:randomPercentage]
    parte4 = solucion1[randomPercentage:]
    
    crossO = np.concatenate([parte1,parte2])
    cross1 = np.concatenate([parte3,parte4])
    
    return crossO , cross1
    
        
def mutacion(solucion_actual):  # Randomly modify random position 
    
    solucionOriginal = solucion_actual.copy() 
    solucionAlterada = []
    
    randomValue = random.randint(0, int(numVar)-1)

    if solucionOriginal[randomValue] == 1: 
        solucion_actual[randomValue] = 0
    else: 
        solucion_actual[randomValue] = 1
            
    solucionAlterada = solucion_actual   

    if dominancia(solucionAlterada, solucionOriginal) == True:
        return solucionAlterada
    else: 
        return solucionOriginal
    
    
def NoDom(valorZ1, valorZ2, valorZ3): #Pareto Frontier
    
    Comparacion = pd.DataFrame({"Z1":valorZ1, "Z2":valorZ2, "Z3": valorZ3})

    Comparacion = Comparacion.sort_values("Z1", ascending=False)

    elegidos = pd.DataFrame(Comparacion.iloc[0]).T

    for elem in elegidos.iloc[0]:
        for i in range(Comparacion.shape[0]):
            cont = 0 
            noagregar = 0
            for numElegidos in range(len(elegidos)): # Agregando elegidos
                if np.any(Comparacion.iloc[i] > elegidos.iloc[numElegidos]) == True: # Si le gana en todos los valores bye
                    cont += 1 
                if np.all(Comparacion.iloc[i] <= elegidos.iloc[numElegidos]) == True:
                    noagregar  +=1
            if cont> 0 and noagregar == 0:
                elegidos = elegidos.append(Comparacion.iloc[i])
            
                   
    elegidos = elegidos.drop_duplicates()

    SeleccionadosPF = list(elegidos.index)
    
    return SeleccionadosPF


def PoblacionZ1Rec(SeleccionadosPF, poblacion, SolucionesZ, varIguala1, recUtilizados): 
    
    print(poblacion)
    print(SeleccionadosPF)
    
    noRepetidasPareto = []
    noRepetidasZ = []
    noRepetidas1 = []
    noRepetidasRec = []
    
    for i in SeleccionadosPF: 
        noRepetidasPareto.append(poblacion[i])
        noRepetidasZ.append(SolucionesZ[i])
        noRepetidas1.append(varIguala1[i])
        noRepetidasRec.append(recUtilizados[i])
        
    return noRepetidasPareto, noRepetidasZ, noRepetidas1, noRepetidasRec

start = time.time()
elapsed = time.time() - start

paretoFront = []
varIguala1 = []
recUtilizados = []

SolucionesZ = []

valorZ1 = []
valorZ2 = []
valorZ3 = []

numberOfChildren = 10
conjuntoSoluciones = []  
probabilidadMutacion = 0.6
adjuntados = [] 


indicesSeleccionadas = []
Rest = []
Zval = []

for i in range(NumSolDisponible):
    indicesSeleccionadas.append((dfconst.iloc[i+1][1:dfconst.iloc[i+1][0]+1].values).astype(int))
    Rest.append((dfconst.iloc[i+1][dfconst.iloc[i+1][0]+1: dfconst.iloc[i+1][0]+1 + int(numRes)].values).astype(int))
    Zval.append((dfconst.iloc[i+1][dfconst.iloc[i+1][0]+1+int(numRes): dfconst.iloc[i+1][0]+1+int(numRes)+3].values).astype(int))
        
solucion_inicial = np.zeros(int(numVar))
poblacioninicial = []

for i in range(len(indicesSeleccionadas)): 
    solucion_inicial = np.zeros(int(numVar))
    solucion_inicial[indicesSeleccionadas[i]-1] = 1
    poblacioninicial.append(solucion_inicial)
    
    
for i in range(len(poblacioninicial)):
    valorZ1.append(np.dot(dfObj2,poblacioninicial[i])[0])
    valorZ2.append(np.dot(dfObj2,poblacioninicial[i])[1])
    valorZ3.append(np.dot(dfObj2,poblacioninicial[i])[2])
    recUtilizados.append(list(np.dot(a, poblacioninicial[i])))
    varIguala1.append(poblacioninicial[i].sum())
    SolucionesZ.append(list(np.dot(dfObj2,poblacioninicial[i])))
            
    
poblacion = poblacioninicial.copy()


while elapsed < 300: 
    for i in range(numberOfChildren): 
        [seleccionado1, seleccionado2] = seleccion(poblacion)
        [cross1, cross2] = crossOver(seleccionado1, seleccionado2)
        randomVal = random.random()
        if randomVal < probabilidadMutacion: 
            cross1 = mutacion(cross1)
            cross2 = mutacion(cross2)
        
        cross1a = cross1.copy()
        cross2a = cross2.copy()
        
        cross1 = cross1.tolist()
        cross2 = cross2.tolist()
        
        adjuntados.append(cross1a)
        adjuntados.append(cross2a)
        
        if np.sum(np.dot(a, cross1)<=b) >= np.sum(np.dot(a, seleccionado2)<=b):
            varIguala1.append(cross1a.sum())
            SolucionesZ.append(list(np.dot(dfObj2,cross1)))
            valorZ1.append(np.dot(dfObj2,cross1)[0])
            valorZ2.append(np.dot(dfObj2,cross1)[1])
            valorZ3.append(np.dot(dfObj2,cross1)[2])
            recUtilizados.append(list(np.dot(a, cross1)))
            
        
        if np.sum(np.dot(a, cross2)<=b) >= np.sum(np.dot(a, seleccionado2)<=b): # Numero de restricciones violadas
            varIguala1.append(cross2a.sum())
            SolucionesZ.append(list(np.dot(dfObj2,cross2)))
            valorZ1.append(np.dot(dfObj2,cross2)[0])
            valorZ2.append(np.dot(dfObj2,cross2)[1])
            valorZ3.append(np.dot(dfObj2,cross2)[2])
            recUtilizados.append(list(np.dot(a, cross2)))
            
        
    aa = np.vstack(adjuntados)
    elapsed = time.time() - start
    for i in range(len(aa)): 
        poblacion.append(aa[i]) 

    indices = []
    print(indices)
    indices = NoDom(valorZ1, valorZ2, valorZ3)
    print(indices)
    
    [noRepetidasPareto, noRepetidasZ, noRepetidas1, noRepetidasRec] = PoblacionZ1Rec(indices, poblacion, SolucionesZ, varIguala1, recUtilizados)         # Para cada generacion se actualiza la poblacion
        
    
    poblacion = noRepetidasPareto
    varIguala1 = noRepetidas1
    SolucionesZ = noRepetidasZ
    
    print(SolucionesZ)
    
    valorZ1 = []
    valorZ2 = []
    valorZ3 = []
    
    for i in range(len(SolucionesZ)): 
        valorZ1.append(noRepetidasZ[i][0])
        valorZ2.append(noRepetidasZ[i][1])
        valorZ3.append(noRepetidasZ[i][2])
        
    recUtilizados = noRepetidasRec
    
    adjuntados = []                      
    
nuevopob = []
nuevo = []

poblacion = [list(item) for item in poblacion]

print(poblacion)

Comparacion = pd.DataFrame({"Z1":valorZ1, "Z2":valorZ2, "Z3": valorZ3})

Comparacion = Comparacion.sort_values("Z1", ascending=False)

elegidos = pd.DataFrame(Comparacion.iloc[0]).T

for elem in elegidos.iloc[0]:
    for i in range(Comparacion.shape[0]):
        cont = 0 
        noagregar = 0
        for numElegidos in range(len(elegidos)): # Agregando elegidos
            if np.any(Comparacion.iloc[i] > elegidos.iloc[numElegidos]) == True: # Si le gana en todos los valores bye
                cont += 1 
            if np.all(Comparacion.iloc[i] <= elegidos.iloc[numElegidos]) == True:
                noagregar  +=1
        if cont> 0 and noagregar == 0:
            elegidos = elegidos.append(Comparacion.iloc[i])
            
                   
elegidos = elegidos.drop_duplicates()

SeleccionadosPF = list(elegidos.index)

noRepetidasPareto = []
noRepetidasZ = []
noRepetidas1 = []
noRepetidasRec = []

for i in range(len(SeleccionadosPF)): 
    noRepetidasPareto.append(poblacion[i])
    noRepetidasZ.append(SolucionesZ[i])
    noRepetidas1.append(varIguala1[i])
    noRepetidasRec.append(recUtilizados[i])
        
numSol = len(noRepetidasPareto)        

registro1 =[]

indicesX= []
for i in range(numSol):
    indicesX = []
    for j in range(int(numVar)):
        if noRepetidasPareto[i][j] == 1:
            indicesX.append(j+1)
    registro1.append(indicesX)
    

listSol = []

listSol.append([numSol])

def flatten(iterable):
    try:
        for item in iterable:
            yield from flatten(item)
    except TypeError:
        yield iterable

for i in range(numSol):
    agregar = [noRepetidas1[i], registro1[i], noRepetidasRec[i], noRepetidasZ[i]]
    entonces = list(flatten(agregar))
    listSol.append(entonces)
    
listSol.append(["Tiempo total de algortimo"])
listSol.append(["300"])

import xlsxwriter 

new_list = listSol

with xlsxwriter.Workbook('Entrega4.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for row_num, data in enumerate(new_list):
        worksheet.write_row(row_num, 0, data)