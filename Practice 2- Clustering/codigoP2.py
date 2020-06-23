# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Noviembre/2019
Contenido:
    Ejemplo de uso de clustering en Python
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
# El siguiente es uno de los algoritmos de clustering mas rapidos existente
from sklearn.mixture import GaussianMixture

from sklearn.cluster import AgglomerativeClustering
#falta. los 2 siguientes son para seguir el ejemplo
# de la web de dbscan
from sklearn import metrics
from sklearn import preprocessing
from math import floor
import seaborn as sns
from scipy.cluster import hierarchy 

'''
for col in censo:
   missing_count = sum(pd.isnull(censo[col]))
   if missing_count > 0:
      print(col,missing_count)
#'''

#Se pueden reemplazar los valores desconocidos por un número
#censo = censo.replace(np.NaN,0)

#O imputar, por ejemplo con la media      
#for col in censo:
#   censo[col].fillna(censo[col].mean(), inplace=True)  #RELLENA VALORES PERDIDOS#


def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())


def LecturaDatos(n):
    datos = pd.read_csv(n)
    
    for col in datos:
        datos[col].fillna(datos[col].mean(), inplace=True)    
    return datos
    pass

# Filtro de datos para quitar los posibles outliers
def Outliers(datos):    
    Q1 = datos.quantile(0.25)
    Q3 = datos.quantile(0.75)
    iqr = Q3 - Q1
    
    datos = datos[~((datos < (Q1-1.5*iqr)) | ( datos > (Q3+1.5*iqr))).any(axis=1)]
    X_normal = datos.apply(norm_to_zero_one)
    
    #falta borrar siguiente linea    
    print ('tam del x normal dentro del metodo outliers: ', X_normal.size)
    return datos, X_normal
    pass

# Grafico de barras con el tamaño de cada cluster
def GraficoTamClusters(clusters):
    size=clusters['cluster'].value_counts()    
    nombres = ('MeanShift','DBSCAN', 'Kmeans', 'Ward', 'GMM')
    y_pos = np.arange(len(nombres))
    datos = []
    
    for num,i in size.iteritems():
       datos.append(100*i/len(clusters)) 
       print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
    
    for i in range(len(datos)):
        plt.bar(y_pos, datos[i], align='center')
        pass    
    
    plt.xticks(y_pos, nombres)
    plt.ylabel("Tamaño de cada cluster")
    #plt.xlabel("Nombre de cada cluster")
    #plt.title("Tamaño de cada cluster")
    plt.show()
   

def EjecucionAlgoritmos(algorithms, X, X_normal):
    for name, alg in algorithms:
        print('----- Ejecutando '+name ,end='')                        #EJECUTAR ALGORITMO
        t = time.time()
        cluster_predict = alg.fit_predict(X_normal) 
        tiempo = time.time() - t
        print(": {:.2f} segundos, ".format(tiempo), end='')
        metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
        print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
        
        #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, 
        #digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
        if len(X) > 10000:
           muestra_silhoutte = 0.2
        else:
           muestra_silhoutte = 1.0
           
        metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
        print("Silhouette Coefficient: {:.5f}".format(metric_SC))
                                                                #FIN EJECUTAR ALGORITMO
                                                                #se convierte la asignación de clusters a DataFrame
        clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
       
        GraficoTamClusters(clusters)
        
        if(name == 'Kmeans'): 
            print('kmeans....\n')
            centers = pd.DataFrame(alg.cluster_centers_,columns=list(X))
            centers_desnormal = centers.copy()
            
            #se convierten los centros a los rangos originales antes de normalizar
            for var in list(centers):
                centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())
            
            sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
        pass
        
        X_kmeans = pd.concat([X, clusters], axis=1)
        sns.set()
        variables = list(X_kmeans)
        variables.remove('cluster')
        sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
        sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
        sns_plot.savefig("kmeans.png")
        print("")
        
        
        #---------------------------------------------------------------
        # PINTAR DENDOGRAMA Y CLUSTERMAP
        #---------------------------------------------------------------
        
        X_cluster = pd.concat([X, clusters], axis=1)

        #Filtro quitando los elementos (outliers) que caen en clusters muy pequeños en el jerárquico
        min_size = 5    # tamaño minimo que exijo al cluster
        
        # el tamaño de los grupos tiene que ser mayor que 5
        X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
        
        X_filtrado = X_filtrado.drop('cluster', 1)
        
        #Normalizo el conjunto filtrado
        X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')
        
        #Saco el dendrograma usando scipy, que realmente vuelve a ejecutar el clustering jerárquico
        linkage_array = hierarchy.ward(X_filtrado_normal)
        plt.figure(1)
        plt.clf()
        hierarchy.dendrogram(linkage_array,orientation='left') #lo pongo en horizontal para compararlo con el generado por seaborn
        
        #Ahora lo saco usando seaborn (que a su vez usa scipy) para incluir un heatmap
        X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X_filtrado.index,columns=usadas)
        # vuelve a hacer el clustering+
        sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
        print('pintando kde')
        sns.kdeplot(data = X)
        
        
    #return clusters
    pass


def PintarScatter(X, clusters, nombreAlgoritmo):
    print("\n---------- Preparando el scatter matrix...")
    #se añade la asignación de clusters como columna a X
    X_kmeans = pd.concat([X, clusters], axis=1)
    sns.set()
    variables = list(X_kmeans)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig(nombreAlgoritmo +".png")
    print("")

    
def ClustersSize(clusters):
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        #datos.append(100*i/len(clusters)) 
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
    

# Definicion de algoritmos a usar. falta revisar parametros pa ver los mejores
k_means = KMeans(init='k-means++', n_clusters=5, n_init=5)      #INSTANCIACION
MS = MeanShift()
ward = AgglomerativeClustering(n_clusters=5, linkage='ward')
db = DBSCAN(eps = 0.3, min_samples=50)
gmm = GaussianMixture(n_components=3,covariance_type='full', max_iter=20, random_state=0)
algoritmos = (('MeanShift',MS),('DBSCAN',db),('Kmeans',k_means),('Ward', ward), ('GMM',gmm))




''' REVISAR
centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
hm.set_ylim(len(centers),0)
'''



#-----------------------------------------------------------------------       
#                   INICIO DEL PROGRAMA
#-----------------------------------------------------------------------


print('INICIO\n')
censo = LecturaDatos('mujeres_fecundidad_INE_2018.csv')
algoritmos = (('MeanShift',MS),('DBSCAN',db),('Kmeans',k_means),('Ward', ward), ('GMM',gmm))


#-----------------------------------------------------------------------       
#                   CASO DE ESTUDIO 1
#-----------------------------------------------------------------------

#mujeres de mas de 40, españolas, que no tienen trabajo ni ella ni la pareja y tienen dificultad
subset = censo.loc[(censo['EDAD']>40) & (censo['EMPPAREJA']==4) & (censo['NAC']==1) & (censo['DIFICULTAD']==1)]# & (censo['EMPPAREJA']==4) (censo['NHIJOSDESEO']==2)]  #FILTROS
usadas = ['RELIGION', 'NHOGAR', 'NTRABA', 'TEMPRELA', 'NDESEOHIJO']
X1 = subset[usadas]

X1, X_normalizada = Outliers(X1)

clusters = EjecucionAlgoritmos(algoritmos, X1, X_normalizada)    


#-----------------------------------------------------------------------       
#                   CASO DE ESTUDIO 2
#-----------------------------------------------------------------------

'''
subset = censo.loc[censo['EDAD']>29]
subset = subset.loc[subset['ESTUCON']>0]
subset = subset.loc[subset['TENEN']<4]
subset = subset.loc[subset['TIPONUC']==2]
subset = subset.loc[subset['SITU']==1]
#REGVI < 4
'''
'''
# mas de 35 años,  y que jornada es completa
subset = censo.loc[(censo['REGVI']==1) & (censo['JORNADA']<3)] # & (censo['HIJOSCONEXP1']==1)]

usadas = ['EDAD', 'NHIJOSPAR', 'NTRABA', 'JORNADA', 'NDESEOHIJO']
X1 = subset[usadas]

X1, X_normalizada = Outliers(X1)

clusters = EjecucionAlgoritmos(algoritmos, X1, X_normalizada)    
'''

#-----------------------------------------------------------------------       
#                   CASO DE ESTUDIO 3
#-----------------------------------------------------------------------


#----------------------------------------------------------------------------
#                   EJEMPLO EJECUCION PARA UN ALGORITMO CONCRETO
#----------------------------------------------------------------------------


