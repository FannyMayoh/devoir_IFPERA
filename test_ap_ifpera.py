import streamlit as slt
slt.header("Application de prédiction de fev/fvc")
slt.subheader("Cette application permet de prédire le fev/fvc, dans un intervalle de confiance de 90% en fonction de l'age et de la taille")
slt.markdown("***Application developpée par Dr Fanny dans le cadre de sa formation en data science***")

import pandas as pd

import numpy as np

df= pd.read_csv('https://raw.githubusercontent.com/pefura/IFPERA/main/Cameroon_lung_function.csv', sep= ';')
df = df.copy()

dataset= df.loc[df['sex']== 2, ['age', 'height', 'fev', 'fvc']]
dataset['fev/fvc']=dataset.fev/dataset.fvc
dataset_c= dataset.drop([1127, 1561, 1628, 1741], axis=0)
data= dataset_c.drop(columns =['fev', 'fvc'])


y = data['fev/fvc']
X = data.drop(columns =['fev/fvc'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,
                                                      random_state=0 )

#Gradient boosting model
from sklearn.ensemble import GradientBoostingRegressor
GB_model_final= GradientBoostingRegressor(max_depth=2, n_estimators=225, random_state=0)

## Fonction de calcul
age = slt.number_input (label= 'Age en années')
height = slt.number_input (label= 'Taille en cm')
def prediction_fevfvc (age, height):
    fit_mean = GradientBoostingRegressor(loss="squared_error", random_state=0, max_depth=2, n_estimators=225,)
    fit_mean.fit(X_train, y_train)
    fit_LLN= GradientBoostingRegressor(loss="quantile", alpha=0.05, random_state=0, max_depth=2, n_estimators=225,)
    fit_LLN.fit(X_train, y_train)
    fit_ULN= GradientBoostingRegressor(loss="quantile", alpha=0.95, random_state=0, max_depth=2, n_estimators=225,)
    fit_ULN.fit(X_train, y_train)
    var = {'age':[age],
        'height':[height]}
    X1 = pd.DataFrame (var)
    pred_mean = fit_mean.predict(X1)
    LLN = fit_LLN.predict(X1)
    ULN = fit_ULN.predict(X1)
    table = pd.DataFrame([LLN [0],pred_mean[0], ULN[0]]).T
    table.columns = ["LLN", "mean", "ULN"]
    return table

slt.markdown("Valeur moyenne prédite de fev/fvc")

fevfvc= slt.write(prediction_fevfvc (age, height))

slt.text ("LLN: limite inférieure de l'IC à 90%")

slt.text ("ULN: limite supérieure de l'IC à 90%")




