# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 03:39:00 2019

@author: pablo
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#falta imbalancedlearn

'''
Lectura de datos
'''

#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('nepal_earthquake_tra_original.csv')
data_y = pd.read_csv('nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('nepal_earthquake_tst.csv')

#se quitan las columnas que no se usan
data_x.drop(labels=['building_id'], axis=1,inplace = True)
data_x_tst.drop(labels=['building_id'], axis=1,inplace = True)
data_y.drop(labels=['building_id'], axis=1,inplace = True)
    
data_aux = pd.read_csv('nepal_earthquake_tra.csv')

'''
Se convierten las variables categóricas a variables numéricas (ordinales)
'''
from sklearn.preprocessing import LabelEncoder
mask = data_x.isnull()
data_x_tmp = data_x.fillna(9999)
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
data_x_nan = data_x_tmp.where(~mask, data_x)

mask = data_x_tst.isnull() #máscara para luego recuperar los NaN
data_x_tmp = data_x_tst.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst) #se recuperan los NaN

X = data_x_nan.values
X_tst = data_x_tst_nan.values

y = np.ravel(data_y.values)




#---------------------------------------------------------------------------------------
#                              ANALISIS EXPLORATORIO DE DATOS
#---------------------------------------------------------------------------------------

















#---------------------------------------------------------------------------------------
#                               DESBALANCEO DE CLASES
#---------------------------------------------------------------------------------------

#import imbalanced_learn
from imblearn.combine import SMOTETomek
'''
def plot_2d_space(X, y, label='damage_grade'):   
    colors = ['#1F77B4', '#FF7F0E'],#, '#111111']
    markers = ['o', 's']#, 'X']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 1],
            X[y==l, 2],
            #X[y==l, 3],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

smt = SMOTETomek(sampling_strategy='auto')
X_smt, y_smt = smt.fit_sample(X, y)
plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
'''

from collections import Counter
'''
from imblearn.combine import SMOTEENN
print("\n------ SMOTE-ENN -----\n")
smote_enn = SMOTEENN(random_state = 123456, n_jobs = -1 )
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
'''

import smote_variants as sv
print("\n------ SMOTE-Polynom_Fit -----\n")
#oversampler = sv.SMOTE_IPF(proportion=1.0, n_jobs = -1, random_state=123456)
oversampler = sv.polynom_fit_SMOTE(random_state = 123456)
X_samp, y_samp = oversampler.sample(X, y)


oversampler2 = sv.ProWSyn(n_jobs = -1, random_state = 123456)
X_samp2, y_samp2 = oversampler2.sample(X, y)

print(sorted(Counter(y).items()))
#print(sorted(Counter(y_resampled).items()))
print(sorted(Counter(y_samp).items())) # polynom fit smote
print(sorted(Counter(y_samp2).items())) #prowsyn

# Visualizacion del espacio tras el sobremuestreo
'''
n_class_0 = data_aux[data_aux['damage_grade']==1].shape[0]# Getting the no. of instances with label 1
n_class_1 = data_aux[data_aux['damage_grade']==2].shape[0]# Bar Visualization of Class Distribution
n_class_2 = data_aux[data_aux['damage_grade']==3].shape[0]# Bar Visualization of Class Distribution

y = np.array([n_class_0, n_class_1, n_class_2])
plt.bar(X, y)
plt.xlabel('Labels/Classes')
plt.ylabel('Number of Instances')
plt.title('Distribution of Labels/Classes in the Dataset')
'''


# falta grafica de barras que saque la cantidad de instancias de cada clase

'''
# Grafica que muestra la importancia de cada variable dentro del dataset
model = xgb.XGBClassifier(n_estimators = 200,n_jobs=-1)
model.fit(X, y)
# plot
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()

#scatterplot
import seaborn as sns
sns.set()
cols = ['height_percentage', 'land_surface_condition', 'position', 'plan_configuration', 'has_superstructure_stone_flag', 'has_superstructue_cement_mortar_stone']
sns.pairplot(X[cols], size = 2.5)
plt.show()
'''

# -----------------------------------------------------------------------------------------------------
#                                           RUIDO
# -----------------------------------------------------------------------------------------------------
'''
import smote_variants as sv
print("\n------ Ruido-----\n")
oversampler = sv.SMOTE_IPF(proportion=1.0, n_jobs = -1, random_state=123456)
X_samp, y_samp = oversampler.sample(X, y)
'''


# -----------------------------------------------------------------------------------------------------
#                                          VALORES PERDIDOS
# -----------------------------------------------------------------------------------------------------

print ("---- Valores perdidos----")
from missingpy import MissForest
imputer = MissForest(n_estimators = 1000, class_weight={1:6.5,2:1.5,3:2}, warm_start = True, n_jobs = -1, random_state = 123456)

X_imputed = imputer.fit_transform(X_samp)
#X_imputed = imputer.fit_transform(X)


# -----------------------------------------------------------------------------------------------------
#                                          SELECCION DE CARACTERÍSTICAS
# -----------------------------------------------------------------------------------------------------



'''
Seleccion de caracterÃ­sticas (Feature Selection)
Realizamos una selecciÃ³n de caracterÃ­sticas usando Random Forest como estimador.
Configuramos Random Forest
'''

'''
print("------ SelecciÃ³n de caracterÃ­sticas...")
from boruta import BorutaPy
rf = RandomForestClassifier(n_estimators = 500, class_weight={1:6.5,2:1.5,3:2}, n_jobs=-1) # , class_weight={1:6.5,2:1.5,3:2})
'''


'''
Configuramos BorutaPy para la selecciÃ³n de caracterÃ­sticas en funciÃ³n de la configuraciÃ³n hecha
para Random Forest
'''


'''
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=123456) #verbose=0, max_iter = 9
#Lo aplicamos sobre nuestros datos.
feat_selector.fit(X, y)
#feat_selector.fit(X_samp, y_samp)

print("\nCaracterÃ­sticas seleccionadas:")
#print(selected_features)
#Comprobar el ranking de caracterÃ­sticas.
print("\nRanking de caracterÃ­sticas:")
print(feat_selector.ranking_)
#Aplicamos transform() a X para filtrar las caracterÃ­sticas y dejar solo las seleccionadas.
X_filtered = feat_selector.transform(X)

print("\nNÃºmero de caracterÃ­sticas inicial: {:d}, despues de la seleccion: {:d}\n".format(X.shape[1],X_filtered.shape[1]))
'''


'''
from feature_selector import FeatureSelector
# Features are in train and labels are in train_labels
fs = FeatureSelector(data = X, labels = y)

fs.identify_all(selection_params = {'missing_threshold': 0.6,    
                                    'correlation_threshold': 0.98, 
                                    'task': 'classification',    
                                    'eval_metric': 'auc', 
                                    'cumulative_importance': 0.99})
'''

'''
import smote_variants as sv
print("\n------ Ruido-----\n")
oversampler = sv.SMOTE_IPF(proportion=1.0, n_jobs = -1, random_state=123456)
X_samp, y_samp = oversampler.sample(X, y)
#☺X_samp_test, y_samp = oversampler.sample(X_tst, y)
'''


#------------------------------------------------------------------------
#Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla

skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=123456)

from sklearn.metrics import f1_score

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("F1 score (val): {:.4f}, tiempo: {:6.2f} segundos".format(f1_score(y[test],y_pred,average='micro') , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all
#------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------
#                                           PROCESAMIENTO
# -----------------------------------------------------------------------------------------------------


#print("------ LightGBM sobre las caracteri­sticas seleccionadas...")
'''
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=1000,n_jobs=-1)
lgbm, y_test_lgbm = validacion_cruzada(lgbm,X_samp,y_samp,skf)
'''

rf = RandomForestClassifier(n_estimators = 1000, class_weight={1:6.5,2:1.5,3:2}, n_jobs=-1 )
rf, y_test_rf = validacion_cruzada(rf,X,y,skf)

#clf = xgbclf
#clf = lgbm
clf = rf
clf = clf.fit(X_samp,y_samp)

'''
print('Plotting feature importances...')
# Plot model’s feature importances.
ax = lgb.plot_importance(lgbm, title='Feature importance', xlabel='Feature importance', ylabel='Features', importance_type='split')
plt.show()
'''

y_pred_tra = clf.predict(X_samp)
print("F1 score (tra): {:.4f}".format(f1_score(y_samp,y_pred_tra,average='micro')))

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission.csv", index=False)

















