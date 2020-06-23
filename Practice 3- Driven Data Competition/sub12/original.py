# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:22:22 2019

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

'''
lectura de datos
'''
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('nepal_earthquake_tra_original.csv')
data_y = pd.read_csv('nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('nepal_earthquake_tst.csv')
#se quitan las columnas que no se usan
data_x.drop(labels=['building_id'], axis=1,inplace = True)
data_x_tst.drop(labels=['building_id'], axis=1,inplace = True)
data_y.drop(labels=['building_id'], axis=1,inplace = True)

#he quitado area-percentage y ,'has_secondary_use_rental'
data_x.drop(labels=['has_secondary_use_hotel'], axis=1, inplace = True)
data_x.drop(labels=['has_secondary_use_rental'], axis=1, inplace = True)
data_x.drop(labels=['has_superstructure_rc_non_engineered'], axis=1, inplace = True)
data_x.drop(labels=['has_superstructure_rc_engineered'], axis=1, inplace = True)
data_x.drop(labels=['has_superstructure_cement_mortar_brick'], axis=1, inplace = True)
data_x.drop(labels=['has_superstructure_cement_mortar_stone'], axis=1, inplace = True)
#data_x.drop(labels=['area_percentage'], axis=1, inplace = True)


data_x_tst.drop(labels=['has_secondary_use_hotel'], axis=1, inplace = True)
data_x_tst.drop(labels=['has_secondary_use_rental'], axis=1, inplace = True)
data_x_tst.drop(labels=['has_superstructure_rc_non_engineered'], axis=1, inplace = True)
data_x_tst.drop(labels=['has_superstructure_rc_engineered'], axis=1, inplace = True)
data_x_tst.drop(labels=['has_superstructure_cement_mortar_brick'], axis=1, inplace = True)
data_x_tst.drop(labels=['has_superstructure_cement_mortar_stone'], axis=1, inplace = True)
#data_x_tst.drop(labels=['area_percentage'], axis=1, inplace = True)

'''
data_x.drop(labels=[''], axis=1, inplace = True)
data_x.drop(labels=[''], axis=1, inplace = True)
data_x.drop(labels=[''], axis=1, inplace = True)
data_x.drop(labels=[''], axis=1, inplace = True)
data_x.drop(labels=[''], axis=1, inplace = True)
'''




'''
data_x_tst.drop(labels=['has_superstructure_cement_mortar_brick','has_superstructure_rc_non_engineered','has_superstructure_rc_engineered','has_secondary_use_hotel'], axis=1,inplace = True)
data_y.drop(labels=['has_superstructure_cement_mortar_brick','has_superstructure_rc_non_engineered','has_superstructure_rc_engineered','has_secondary_use_hotel'], axis=1,inplace = True)
'''
#['has_superstructure_cement_mortar_brick','has_superstructure_rc_non_engineered','has_superstructure_rc_engineered','has_secondary_use_hotel']

# Antes de eliminar esas variables, comprobamos su importancia mediante el analisis exploratorio
# que falta en este archivo

# 'has_superstructure_other','legal_ownership_status','has_superstructure_rc_engineered','count_families',
#'has_superstructure_rc_non_engineered','has_secondary_use','has_superstructure_bamboo', 'has_secondary_use_hotel'



'''
Se convierten las variables categÃ³ricas a variables numÃ©ricas (ordinales)
'''

'''
mask = data_x.isnull()
data_x_tmp = data_x.fillna(9999)
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
data_x_nan = data_x_tmp.where(~mask, data_x)

mask = data_x_tst.isnull() #mÃ¡scara para luego recuperar los NaN
data_x_tmp = data_x_tst.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categÃ³ricas en numÃ©ricas
data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst) #se recuperan los NaN

X = data_x_nan.values
X_tst = data_x_tst_nan.values
'''

y = np.ravel(data_y.values)



from sklearn.preprocessing import LabelEncoder
def categorical_numer(training, test, y_training):
    # Se extraen las categorias
    columnas_categoricas = list(training.select_dtypes('object').astype(str))
    variables_categoricas = training[columnas_categoricas]
    
    # Eliminamos estas variables
    training = training.drop(columns = columnas_categoricas)
    #Aplicamos label encoder
    training_cat = variables_categoricas.apply(preprocessing.LabelEncoder().fit_transform)
    #juntamos el conjunto de train
    training_procesado = pd.concat((training, training_cat), axis=1, join='outer', ignore_index=False,
                                   levels=None, names=None, verify_integrity=False, copy=True)
    
    # Repeticion del proceso para el conjunto de test
    
    # Se extraen las categorias
    columnas_categoricas = list(test.select_dtypes('object').astype(str))
    variables_categoricas = test[columnas_categoricas]
    
    # Eliminamos estas variables
    test = test.drop(columns = columnas_categoricas)
    #Aplicamos label encoder
    test_cat = variables_categoricas.apply(preprocessing.LabelEncoder().fit_transform)
    #juntamos el conjunto de train
    test_procesado = pd.concat((test, test_cat), axis=1, join='outer', ignore_index=False,
                                   levels=None, names=None, verify_integrity=False, copy=True)
        
    return training_procesado, test_procesado

print("Preprocesando los datos...\n")
train_procesado, test_procesado = categorical_numer(data_x, data_x_tst, y)
X = train_procesado.values
X_tst = test_procesado.values

'''
# Comprobando la distribución de la variable de clasificación
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot('damage_grade', data = data_y)
plt.rcParams["figure.figsize"] = (30,30)
plt.show()
'''

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
rng = np.random.RandomState(0)
#X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
'''
print("Aplicando QuantileTransformer")
qt = QuantileTransformer( random_state=123456)
qt.fit_transform(X)
#qt.fit_transform(X_tst)
'''

print("Aplicando Standard Scaler")
qt = StandardScaler()
qt.fit_transform(X)


import smote_variants as sv
#print("\n------ SMOTE-Polynom_Fit -----\n")
#oversampler = sv.SMOTE_IPF(proportion=1.0, n_jobs = -1, random_state=123456)
oversampler = sv.polynom_fit_SMOTE(random_state = 123456)
X_samp, y_samp = oversampler.sample(X, y)



#------------------------------------------------------------------------
'''
ValidaciÃ³n cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
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
lgbm = lgb.LGBMClassifier(objective='multiclass',n_estimators=200, n_jobs=-1)#, num_threads=2)
lgbm, y_test_lgbm = validacion_cruzada(lgbm,X_samp,y_samp,skf)

print("------ Importancia de las caracterÃ­sticas...")
importances = list(zip(lgbm.feature_importances_, data_x.columns))
importances.sort()
pd.DataFrame(importances, index=[x for (_,x) in importances]).plot(kind='barh',legend=False)
'''


'''
print("Seleccion de caracteristicas")
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=5, n_estimators=200, n_jobs=-1) #warm_start=True, class_weight={1:5, 2:1.2, 3:1.7},

feature_selector = BorutaPy(rf, n_estimators=200, verbose=0, max_iter=5, random_state=123456)
feature_selector.fit(X, y)
print("Ranking de caracteristicas: ", feature_selector.ranking_)
X_filtered = feature_selector.transform(X)

print("Numero de caractericas inicial: {:d}, despues de la seleccion {:d}".format(X.shape[1], X_filtered.shape[1]))
'''


'''
print("------ XGB...")
xgbclf = xgb.XGBClassifier(objective='regression_l1', verbose=0, tree_method='hist', class_weight={1:5,2:1.2,3:1.7}, n_estimators=500,n_jobs=-1)#n_estimators = 200,n_jobs=2,)
xgbclf, y_test_xgbclf = validacion_cruzada(xgbclf,X_samp,y_samp,skf)


'''
print("------ LightGBM...")
#lgbm = lgb.LGBMClassifier(objective='regression_l1', class_weight={1:5,2:1.2,3:1.7}, n_estimators=500,n_jobs=-1) #boosting_type= 'rf', 
# = lgb.LGBMClassifier(objective='multiclass', class_weight={1:5,2:1.2,3:1.7}, n_estimators=500,n_jobs=-1) #boosting_type= 'rf', 
lgbm = lgb.LGBMClassifier(objective='regresion_l1', num_leaves=130, n_estimators=1700,n_jobs=-1) #boosting_type= 'rf', 
lgbm, y_test_lgbm = validacion_cruzada(lgbm,X_samp,y_samp,skf)


from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

print("------ Grid Search...")
params_lgbm = {'feature_fraction':[i/10.0 for i in range(3,6)], 'learning_rate':[0.1, 0.2],
               'num_leaves':[100, 130], 'n_estimators':[700,800]}

grid = GridSearchCV(lgbm, params_lgbm, cv=3, n_jobs=-1, verbose=1, scoring='f1_micro') #make_scorer(f1_score))

grid.fit(X_samp,y_samp)

print("Mejores parÃ¡metros:")
print(grid.best_params_)
print("\n------ LightGBM con los mejores parÃ¡metros de GridSearch...")
gs, y_test_gs = validacion_cruzada(grid.best_estimator_,X_samp,y_samp,skf)

print("Mejores parÃ¡metros:")
print(grid.best_params_)


'''
print("AdaBoost...")
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators = 300,  algorithm='SAMME.R', random_state=123456) #learning_rate = 1.5,
ada, y_test_rf = validacion_cruzada(ada,X_samp,y_samp,skf)
'''

#clf = xgbclf
#clf = lgbm
clf = gs
clf = clf.fit(X_samp,y_samp)

y_pred_tra = clf.predict(X_samp)
print("F1 score (tra): {:.4f}".format(f1_score(y_samp,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission.csv", index=False)



'''
------ LightGBM...
F1 score (val): 0.8171, tiempo: 131.72 segundos
F1 score (val): 0.8167, tiempo: 429.37 segundos
F1 score (val): 0.8202, tiempo: 393.54 segundos
F1 score (val): 0.8196, tiempo: 190.53 segundos
F1 score (val): 0.8196, tiempo:  41.33 segundos

F1 score (tra): 0.8309

'''

'''
Aplicando Standard Scaler
2019-12-31 16:18:40,783:INFO:polynom_fit_SMOTE: Running sampling via ('polynom_fit_SMOTE', "{'proportion': 1.0, 'topology': 'star', 'random_state': 123456}")
------ LightGBM...
F1 score (val): 0.8267, tiempo:  32.44 segundos
F1 score (val): 0.8258, tiempo:  33.19 segundos
F1 score (val): 0.8269, tiempo:  36.05 segundos
F1 score (val): 0.8285, tiempo:  57.38 segundos
F1 score (val): 0.8279, tiempo:  38.59 segundos

------ Grid Search...
Fitting 3 folds for each of 36 candidates, totalling 108 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed: 12.9min
[Parallel(n_jobs=-1)]: Done 108 out of 108 | elapsed: 41.5min finished
Mejores parÃ¡metros:
{'feature_fraction': 0.5, 'learning_rate': 0.1, 'n_estimators': 600, 'num_leaves': 130}

------ LightGBM con los mejores parÃ¡metros de GridSearch...
F1 score (val): 0.8266, tiempo:  39.74 segundos
F1 score (val): 0.8259, tiempo:  40.12 segundos
F1 score (val): 0.8285, tiempo:  39.83 segundos
F1 score (val): 0.8282, tiempo:  39.35 segundos
F1 score (val): 0.8282, tiempo:  39.59 segundos
'''
