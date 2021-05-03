import numpy as np
from sklearn.svm import SVC
from geneticalgorithm import geneticalgorithm as ga
from sklearn import datasets as ds
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics

names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","target"]

df = pd.read_csv('kddcup99_csv.csv', names=names)
print(df.shape)
print(df.columns)
#afficher la liste des colonnes et leurs types avec la fonction dtypes
print(df.dtypes)
data = df.to_numpy()
Y = data[:,39]
X = data[:,0:39]

x_app,x_test,y_app,y_test = train_test_split(X, Y, test_size=300,random_state=0)
print(x_app.shape, x_test.shape, y_app.shape, y_test.shape)
svm = SVC()
#exécution de l'instance sur les données d'apprentissage
#c.à d . construction du modèle prédictif
svm.fit(x_app,y_app)
print("model score: %.3f" % svm.score(x_test, y_test))
#appliquer le modèle sur les données test
predictionSVM = svm.predict(x_test)
#création et affichage de la matrice de confusion
cm = metrics.confusion_matrix(y_test, predictionSVM)
print(cm)
