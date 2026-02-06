#streamlit webapp
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pickle
from tensorflow.keras.models import load_model


#Load the trained model
model=tf.keras.models.load_model("model.h5")

#Load the encoder and scalars
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

#Streamlit app so that we dont have to worry about html
st.title("Customer Churn Prediction")

#User input
#geography = st.selectbox('Geography', list(onehot_encoder_geo.categories[0]))
geo_options = list(onehot_encoder_geo.categories_[0])  
geography = st.selectbox("Geography", geo_options)
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age= st.slider('Age',18,100)
balance=st.number_input('Balance',min_value=0.0)
credit_score=st.number_input('Credit Score',min_value=0.0)
EstimatedSalary=st.number_input('Estimated Salary',min_value=0.0)
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
IsActiveMember=st.selectbox('Is Active Member',[0,1])

#Prepare input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})




# One-hot encode Geography (match training format)
geo_encoded = onehot_encoder_geo.transform(
    pd.DataFrame({'Geography': [geography]})
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine geography with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scale the input data
input_scaled=scaler.transform(input_data)

#Predict Churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')


if prediction_proba >= 0.5:
    st.write("Customer will churn")
else:
    st.write("Customer will NOT churn")
