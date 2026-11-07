import streamlit as st
import pandas as pd
import pickle

st.header('Social Media Impact Application')

with open('socialmedia_classify.pkl','rb') as f:
    model=pickle.load(f)


df=pd.read_csv('Social_media_impact_on_life.csv')
df.columns=df.columns.str.strip()

with st.sidebar:
    st.image("parami.jpg") 
    st.text("Student Name: Zay Yan Myint")
    st.text("Student ID: PUIS20240081")
    st.text("Class Name: Into to Machine Learning")
    st.write("Class of 2028")

col1,col2=st.columns(2)

with col1:
    Age= st.number_input(
            "Enter your age",
             max_value=int(df['Age'].max()),
             min_value=int(df['Age'].min()),
             value=20
        )
    Gender=st.selectbox('Gender', df['Gender'].unique())
    Academic_Level=st.selectbox('Academic Level', df['Academic_Level'].unique())
    Avg_Daily_Usage_Hours = st.slider(
            "Average Daily Usage Hours",
            max_value=float(df['Avg_Daily_Usage_Hours'].max()),
            min_value=float(df['Avg_Daily_Usage_Hours'].min()),
            value=5.0)


with col2:
    Most_Used_Platform = st.selectbox( 'Most Used Platform', df['Most_Used_Platform'].unique())
    Affects_Academic_Performance = st.selectbox(
        'Does Social Media Affect Academic Performance?',
        df['Affects_Academic_Performance'].unique())
    Sleep_Hours_Per_Night = st.number_input(
        "Sleep Hours Per Night",
        max_value=float(df['Sleep_Hours_Per_Night'].max()),
        min_value=float(df['Sleep_Hours_Per_Night'].min()),
        value=4.0
    )
    Mental_Health_Score = st.slider(
        "Mental Health Score",
        max_value=float(df['Mental_Health_Score'].max()),
        min_value=float(df['Mental_Health_Score'].min()),
        value=5.0
    )

predict_button = st.button("Predict Social Media Impact")

if predict_button:
    input_data = pd.DataFrame({
        'Age': [Age],
        'Gender': [Gender],
        'Academic_Level': [Academic_Level],
        'Most_Used_Platform': [Most_Used_Platform],
        'Affects_Academic_Performance': [Affects_Academic_Performance],
        'Avg_Daily_Usage_Hours': [Avg_Daily_Usage_Hours],
        'Sleep_Hours_Per_Night': [Sleep_Hours_Per_Night],
        'Mental_Health_Score': [Mental_Health_Score]
    })

    result = model.predict(input_data)[0]

    if result == 0:
        st.write("Your social media impact is Negative.")
    elif result == 1:
        st.write("Your social media impact is Neutral.")
    elif result == 2:
        st.write("Your social media impact is Positive.")


with st.expander("Dataset"):
    st.table(df)