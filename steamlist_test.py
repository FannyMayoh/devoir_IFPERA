import streamlit as st
import pandas as pd
import numpy as np

#Finitions de l'application
st.title('Application de prédiction de fev/fvc')
st.markdown("***Application developpée par Dr Fanny dans le cadre de sa formation en data science***")
st.subheader("Cette application permet de prédire le fev/fvc, dans un intervalle de confiance de 90% en fonction de l'age et de la taille")

#Données utiles
df= pd.read_csv('https://raw.githubusercontent.com/pefura/IFPERA/main/Cameroon_lung_function.csv', sep= ';')
df = df.copy()

data= df.loc[df['sex']== 2, ['age', 'height', 'fev', 'fvc']]
data['fev/fvc']=data.fev/data.fvc

y = data['fev/fvc']
X = data.drop(columns =['fev/fvc', 'fev', 'fvc'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,
                                                      random_state=0 )

#Gradient boosting model
from sklearn.ensemble import GradientBoostingRegressor
GB_model_final= GradientBoostingRegressor(max_depth=2, n_estimators=225, random_state=0)

## Fonction de calcul
age = st.number_input (label= 'Age en années')
height = st.number_input (label= 'Taille en cm')


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

fevfvc = prediction_fevfvc (age, height)

st.markdown("Valeur moyenne prédite de fev/fvc")
fevfec= st.write(fevfvc)
st.text ("LLN: limite inférieure de l'IC à 90%")
st.text ("ULN: limite supérieure de l'IC à 90%")

            