import numpy as np
import pickle 
import pandas as pd
import streamlit as st
model = pickle.load(open('models/RandomForest1/RandomForest1.sav', 'rb'))
def predict(df):
    results=model.predict(df)
    if results==0:
        return "Your Income is less than 50K"
    else:
        return "Your Income is Greater than 50K"

def main():
    st.title("Income Predictor")
    html_temp="""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Income Prediction</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age=st.text_input("Age","Type here")
    workclass=st.text_input("workclass","Type here")
    education=st.text_input("education","Type here")
    marital_status=st.text_input("martial_status","Type here")
    occupation=st.text_input("Occupation","Type Here")
    relationship=st.text_input("relationship","Type here")
    race=st.text_input("race","Type here")
    sex=st.text_input("sex","Type here")
    capital_gain=st.text_input("capital-gain","Type here")
    capital_loss=st.text_input("capital-loss","Type here")
    hours_per_week=st.text_input("hours_per_week","Type here")
    country=st.text_input("country","Type here")
    result=""
    if st.button("Predict"):
        result=predict([[age,workclass,education,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,country]])
    st.success('Verdict: {}'.format(result))
if __name__=="__main__":
    main()