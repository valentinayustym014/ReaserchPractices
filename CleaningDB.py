#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: valentinayusty
"""

# Cleaning the DB and using Mahalanobis Distance for Outlier Detection
 
import pandas as pd
import numpy as np
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
 
# Inicialization DF
 
df = pd.read_excel('BaseP1.xlsx', sheet_name='Exportar Hoja de Trabajo')
 
# Identify and remove columns that will not be used 

h=['SOLICITUD', 'SOLFECHAINICIO', 'FECHA_REFER', 'CONCESIONARIO_SOL',
 'ZONA_SOL', 'REFERENCIA_MOTO_SOL', 'CILINDRAJE_MOTO_SOL',
 'POTENCIA_MOTO_SOL', 'CAJA_MOTO_SOL', 'COMBUSTIBLE_SOTO_SOL',
 'VALOR_COMERCIAL_SOL', 'MARCA_ORIGINAL_SOL',
 'SCORE_RAD', 'HUELLAOTR5_RAD', 'HUELLA5_RAD', 
 'TIPOATENCION_RAD',
 'NUEVO TIPO DE ATENCIÓN',
 'DISPONIBLE_RAD', 'EVALUAPROP_RAD', 'EVALUACOD_RAD', 'EVALUAINI_RAD',
 'EVALUAGM_RAD', 'EVALUAHUELLA_RAD', 'HABITOPAGO_RAD', 'ACIERTA_RAD',
 'MARCAO_RAD', 'MARCAV_RAD', 'INICIAL_RAD', 'PROPOSITOVEH_RAD',
 'ZONA_RAD', 'VALORCOMER_RAD', 'SOLICITADO_RAD', 
 'CONCESIONARIO_RAD', 'CIUDAD_CONCESIONARIO_RAD',
 'DPT_CONCESIONARIO_RAD', 'TIPO_ATENCION', 'CIUDAD_RESIDENCIA',
 'PROFESION', 'SECTOR_ECONOMICO', 'indicador_retanqueo',
 'indicador_rrs', 'HUELLA_SUFI_', 'HUELLA_OTROS_', 'producto_financiero',
 'GASTOS','TIPOCLI_RAD']
 
df = df.drop(h, axis=1)
df=df.drop('SOLNUMERO_RAD', axis=1)
 
#Remove individuals whose monthly payment is greater than 1,000,000 COP (Outliers) (indexes)

df = df.drop([8712,12350, 41322, 89256])
 
#Remove those records that have null values
 
df.dropna(subset=['CAPACIDAD_RAD', 'ANTIGUEDAD_RAD', 'ESTADOCIV_RAD',
 'GENERO_RAD', 'ESTUDIO_RAD', 'PERSONASCARGO_RAD', 'SUBTIPOCLI_RAD',
 'VIVIENDA_RAD', 'EDAD_RAD', 'CUOTASINF_RAD', 'ESTADO',
 'PLAZO', 'INGRESOS', 'EGRESOS','ESTADO_ANTERIOR'], inplace=True)
 
# Convert String Datatype into Binary Datatype

df['ESTADO'] = df['ESTADO'].replace(['Anulada', "Negada", "Aprobada", "Comité Crédito", "Cotizado", "Aprobada Sin Desembolso", "Aplazada por Estudio"], np.nan)
 
tobereplaced= ['Actualización por Visita', "Aplazada por Estudio", "Aplazada por Referencias", 
 "Comité Crédito", "Cotizado", "Desistida", "En Cambio de Producto", "Negada", 
 "Pendiente Codeudor", "Radicada", "Referencias Pendientes", "Suspendida", 
 "Suspendida otro Aliado", "Visita Pendiente", "Anulada"]
 
df['ESTADO_ANTERIOR'] = df['ESTADO_ANTERIOR'].replace(tobereplaced, np.nan)
 
df.dropna(subset=['CAPACIDAD_RAD', 'ANTIGUEDAD_RAD', 'ESTADOCIV_RAD',
 'GENERO_RAD', 'ESTUDIO_RAD', 'PERSONASCARGO_RAD', 'SUBTIPOCLI_RAD',
 'VIVIENDA_RAD', 'EDAD_RAD', 'CUOTASINF_RAD', 'ESTADO',
 'PLAZO', 'INGRESOS', 'EGRESOS','ESTADO_ANTERIOR'], inplace=True)
 
df['ESTADO'] = df['ESTADO'].replace(['Desistida'], 0)
df['ESTADO'] = df['ESTADO'].replace(['Aprobada Con Desembolso'], 1)
 
df['ESTADOCIV_RAD'] = df['ESTADOCIV_RAD'].replace(["CASA", "UNLB"], 1)
df['ESTADOCIV_RAD'] = df['ESTADOCIV_RAD'].replace(["DIVO", "SOLT", "VIUD"], 0)
df['ESTADOCIV_RAD'] = df['ESTADOCIV_RAD'].replace(["CASA", "UNLB"], 1)

df['SUBTIPOCLI_RAD'] = df['SUBTIPOCLI_RAD'].replace(["INFO"], 0)
df['SUBTIPOCLI_RAD'] = df['SUBTIPOCLI_RAD'].replace(["FORM"], 1)

df['GENERO_RAD'] = df['GENERO_RAD'].replace(["F"], 0)
df['GENERO_RAD'] = df['GENERO_RAD'].replace(["M"], 1)

df['ESTUDIO_RAD'] = df["ESTUDIO_RAD"].replace(["NING", "POST", "PRIM", "SECU"], 0)
df['ESTUDIO_RAD'] = df['ESTUDIO_RAD'].replace(["TECO", "UNDA"], 1)
 
df["CUOTASINF_RAD"] = df['CUOTASINF_RAD'].astype(str)
df['CUOTASINF_RAD']=df['CUOTASINF_RAD'].str.replace(',','.')
df['CUOTASINF_RAD'] = df['CUOTASINF_RAD'].astype(float)
 
# Organize DF
 
df=df.drop(["ESTADO_ANTERIOR", "CAPACIDAD_RAD"], axis=1)
cols = df.columns.tolist()
cols.insert(14, cols.pop(cols.index('ESTADO')))
df = df.reindex(columns= cols)
 
# Calculate Outliers
 
def createCovariance(data):
 return pd.DataFrame.cov(data)
 
def mahal(data):
 
 mahalData = []
 meanData = np.mean(data, axis = 0)
 
 for i in range(data.shape[0]):
     primer = data.iloc[i] - meanData
     primerT = np.transpose(primer)
     S = createCovariance(data)
     SInv = np.linalg.inv(S)
 
     mahalVal = np.dot(np.dot(primer, SInv), primerT)
     mahalData.append(mahalVal)  
     return np.asarray(mahalData)
 
continiousvar=['ANTIGUEDAD_RAD', 'PERSONASCARGO_RAD', 'EDAD_RAD',
 'CUOTASINF_RAD', 'INGRESOS', 'EGRESOS']
 
dfcont=df[continiousvar]
 
d=mahal(dfcont)
 
percentileMahal = chi2.ppf(0.95 , np.shape(dfcont)[1])

distance=[d[i]>percentileMahal for i in range(np.shape(d)[0])]
 
dfcont["Mahalanobis Distance"]=distance
 
registrosAtipicos= dfcont[dfcont["Mahalanobis Distance"]==True].index
 
dfcont=dfcont.drop(dfcont[dfcont["Mahalanobis Distance"]==True].index)
 
dfcont= dfcont.drop(["Mahalanobis Distance"], axis=1)
 
nuevodf= dfcont.merge(df,"inner")