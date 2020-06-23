# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Noviembre/2019
Contenido:
    Uso simple de XGB y LightGBM para competir en DrivenData:
       https://www.drivendata.org/competitions/57/nepal-earthquake/
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
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
import smote_variants as sv

'''
lectura de datos
'''

#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('../nepal_earthquake_tra_original.csv')

# falta mejorar teniendo en cuenta la prediccion de otro modelo Imputamos valores perdidos 
# de momento con la media que es una manera tosca de hacerlo
#for col in data_x:
#   data_x[col].fillna(data_x[col].mean(), inplace=True)  #RELLENA VALORES PERDIDOS#



data_y = pd.read_csv('../nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('../nepal_earthquake_tst.csv')

#se quitan las columnas que no se usan
data_x.drop(labels=['building_id'], axis=1,inplace = True)
data_x_tst.drop(labels=['building_id'], axis=1,inplace = True)
data_y.drop(labels=['building_id'], axis=1,inplace = True)
    


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


'''
for col in data_x:
   data_x[col].fillna(data_x[col].mean(), inplace=True)  #RELLENA VALORES PERDIDOS#
'''

'''
from sklearn.impute import SimpleImputer
imp = SimpleImputer ( strategy = "most_frequent" )
imp.fit_transform ( data_x )
# data_x = data_x.replace(np.NaN, 0)
'''

#oversampler = sv.SMOTE_IPF()
'''
oversampler = sv.polynom_fit_SMOTE()
X_samp, y_samp = oversampler.sample(data_x, data_y)
'''

'''
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(data_x, data_y)
'''

'''
data_x1 = pd.read_csv('nepal_earthquake_tra.csv')
print("------ Y en la clase")
print(data_x1['damage_grade'].value_counts(),"\n")
'''


#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)

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

'''
falta revisar/borrar...

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []
    y_prob_all = []

    for train, test in cv.split(X, y):
        modelo = modelo.fit(X[train],y[train])
        y_pred = modelo.predict(X[test])
        y_prob = modelo.predict_proba(X[test])[:,1] #la segunda columna es la clase positiva '1' en bank-marketing
        y_test_bin = y[test]
        #y_test_bin = le.fit_transform(y[test]) #se convierte a binario para AUC: 'yes' -> 1 (clase positiva) y 'no' -> 0 en bank-marketing
        
        print("Accuracy: {:6.2f}%, F1-score: {:.4f}, G-mean: {:.4f}, AUC: {:.4f}".format(accuracy_score(y[test],y_pred)*100 , f1_score(y[test],y_pred,average='micro'), geometric_mean_score(y[test],y_pred,average='micro'), roc_auc_score(y_test_bin,y_prob)))
        y_test_all = numpy.concatenate([y_test_all,y_test_bin])
        y_prob_all = numpy.concatenate([y_prob_all,y_prob])

    print("")

    return modelo, y_test_all, y_prob_all

'''



'''
print("------ XGB...")
xgbclf = xgb.XGBClassifier(n_estimators = 200,n_jobs=2)
xgbclf, y_test_xgbclf = validacion_cruzada(xgbclf,X,y,skf)
#'''


print("------ LightGBM...")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=200,n_jobs=5)
lgbm, y_test_lgbm = validacion_cruzada(lgbm,X,y,skf)

'''
def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary',
             sample_weight=None, zero_division="warn"):
'''
    
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
print("------ Grid Search...")
params_lgbm = {'feature_fraction':[i/10.0 for i in range(3,6)], 'learning_rate':[0.05,0.1],
               'num_leaves':[30,50], 'n_estimators':[200]}
puntuacion = f1_score(y,y_test_lgbm,average='micro')
grid = GridSearchCV(lgbm, params_lgbm, cv=3, n_jobs=1, verbose=1, scoring=make_scorer(puntuacion))
#grid.fit(X,y_bin)
grid.fit(X,y)
print("Mejores parÃ¡metros:")
print(grid.best_params_)
print("\n------ LightGBM con los mejores parÃ¡metros de GridSearch...")
#gs, y_test_gs, y_prob_gs = validacion_cruzada(grid.best_estimator_,X,y,skf)
grid_model, y_test_grid = validacion_cruzada(grid.best_estimator_,X,y,skf)

'''
print("---------Random Forest-----")
rf = RandomForestClassifier(n_estimators = 500, class_weight={1:6.5,2:1.5,3:2}, n_jobs=6 )
rf, y_test_lgbm = validacion_cruzada(rf,X,y,skf)
'''


#clf = xgbclf
clf = lgbm
#clf = rf
clf = clf.fit(X,y)

y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('../nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission.csv", index=False)
