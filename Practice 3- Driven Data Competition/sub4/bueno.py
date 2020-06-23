# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 16:51:46 2019

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
import smote_variants as sv

'''
lectura de datos
'''

#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('nepal_earthquake_tra_original.csv')

# falta mejorar teniendo en cuenta la prediccion de otro modelo Imputamos valores perdidos 
# de momento con la media que es una manera tosca de hacerlo
#for col in data_x:
#   data_x[col].fillna(data_x[col].mean(), inplace=True)  #RELLENA VALORES PERDIDOS#



data_y = pd.read_csv('nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('nepal_earthquake_tst.csv')

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
print("------ LightGBM...")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=200,n_jobs=-1)
lgbm, y_test_lgbm = validacion_cruzada(lgbm,X,y,skf)
'''


print("---------Random Forest-----")
rf = RandomForestClassifier(n_estimators = 500, class_weight={1:6.5,2:1.5,3:2}, n_jobs=6 )
rf, y_test_lgbm = validacion_cruzada(rf,X,y,skf)



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS
print("------ Grid Search...")
params_lgbm = {'feature_fraction':[i/10.0 for i in range(3,6)], 'learning_rate':[0.05,0.1],
               'num_leaves':[30,50], 'n_estimators':[200]}

cw1 = {1:6.5,2:1.5,3:2}
cw2 = {1:8,2:1.3,3:1.8}
cw3 = {1:7,2:1.3,3:2}
params_rf = {'n_estimators':[500, 1000, 1100], 'class_weight' :[cw1, cw2, cw3], 'n_jobs':[-1]}

puntuacion = f1_score(y,y_test_lgbm,average='micro')
#grid = GridSearchCV(lgbm, params_lgbm, cv=3, n_jobs=1, verbose=1, scoring=make_scorer(puntuacion))

'''
print("Scorers validos:")
print(sorted(SCORERS.keys()))
'''

#grid = GridSearchCV(lgbm, params_lgbm, cv=3, n_jobs=1, verbose=1, scoring=sorted(SCORERS.keys()))
#grid = GridSearchCV(lgbm, params_lgbm, cv=3, n_jobs=-1, verbose=1, scoring='f1_micro')
grid = GridSearchCV(rf, params_rf, cv=3, n_jobs=-1, verbose=1, scoring='f1_micro')



grid.fit(X,y)

'''
print("Mejores parÃ¡metros:")
print(grid.best_params_)
'''

print("\n------ LightGBM con los mejores parÃ¡metros de GridSearch...")
#gs, y_test_gs, y_prob_gs = validacion_cruzada(grid.best_estimator_,X,y,skf)
grid_model, y_test_grid = validacion_cruzada(grid.best_estimator_,X,y,skf)






#clf = xgbclf
#clf = lgbm
#clf = rf
#clf = grid
clf = grid_model
clf = clf.fit(X,y)

y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission.csv", index=False)


