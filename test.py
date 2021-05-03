'''

Source: https://prograide.com/pregunta/1567/-commentaires-multilignes-en-python-
import sklearn
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn import linear_model
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn import cluster
#lecture du fichier "fromage.txt" avec la fonction read_table de pandas
#header = 0, la première ligne correspond à l'entête (intitulé des champs)
filepath =  '/content/drive/MyDrive/data/fromage.txt'
fromage = pd.read_table(filepath,sep="\t",header=0,index_col=0)
print("Head\n",fromage.head())
#afficher les dimensions de la table  (fonction shape) ainsi que leur types (propriété dtype)
print("Dim\n",fromage.shape)
#afficher la liste des colonnes et leurs types avec la propriété dtype
print("Type of columns\n",fromage.dtypes)
#afficher les 4 premières lignes de la table
print(fromage[0:4])

#importation de la librarire

#Affichage des statistiques descriptives
print(fromage.describe())
#affichage du graphique de corrélation
scatter_matrix(fromage,figsize=(9,9))
#Nous pouvons remarquer dans la matrice de corrélation que les calories sont corrélé positivement avec les lipides, avec le cholestérole et le magnesiums
#les proteines sont corrélé positivement avec le cholesteroles et les lipides
#les lipides et le cholesteroles sont aussi corrélé
#magnesium avec

#k-means sur les données des fromages
#importer les librarires cluster et pyplot (as plt)

from matplotlib import pyplot
from sklearn import datasets
import pandas as pd
#création d'un clusetr Kmean avec un nombre de clusters égal à 4
kmeans = cluster.KMeans(n_clusters=4)

#Apprentissage (segmetation) (fonction fit)


kmeans.fit(fromage)

#Prédiction et enregistrement des labels
labels= kmeans.predict(fromage)

#Affichage des labels
print(labels)
#enregistrement des centres des clusters

plt.scatter(centresKM[:,0],centresKM[:,1], s=60,marker='^',linewidth=2)

from sklearn.cluster import AgglomerativeClustering
#création d'une  CHA
agg= AgglomerativeClustering(n_clusters=3)

#Apprentissage (segmetation)

#Prédiction et enregistrement des labels (fonction fit_predict)
labelsCHA = ....
print(labelsCHA)
**/
#Affichage du dendrogramme de la méthode hiérarchique
#importer la librarairie scipy.cluster.hierarchy

#générer le dendrogramme des méthodes hiérarchique

#librairie pour évaluation des partitions

# ffichage de la fonction

#correspondance avec les groupes de kmeans

'''
'''


def f(a, list=[]):
    for i in range(a):
        list.append(i * i)
    print(list)


f(3)
f(2, [1, 2, 3])
f(2,[])

list = ['1','2','3','4','5']
print (list[:12])
'''
