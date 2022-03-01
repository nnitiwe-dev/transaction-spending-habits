## streamlit
import streamlit as st 
## for data manipulation
import pandas as pd 
import re 
## For model loading 
import time
from sklearn import feature_extraction
import joblib
## for text input and output from the web app
from PIL import Image   
from io import StringIO  




def load_model():
#declare global variables
    global nlp_model
    global vectorizer

    nlp_model=joblib.load("clf-nlp-model.pickle")
    vectorizer=joblib.load("clf-nlp-vectorizer.pickle")


def predict(raw_text):
    print("Raw Narration = ", raw_text)  ## tweet
    raw_text = re.sub(r'[^\w\s]', '', str(raw_text).lower().strip())
    X_test = vectorizer.transform([raw_text])

    predicted=nlp_model.predict(X_test)

    #load labels
    labels_df=pd.read_csv("label.csv")
    labels_df=labels_df.replace(24,0)
    result_labels=labels_df.sort_values('indd', axis=0, ascending=True)['label'].tolist()
    print(result_labels,predicted)
    result=result_labels[predicted[0]]

    print(result)
    return(result)


def run():
    st.sidebar.info('You can either enter the Narration text online in the textbox or upload a txt file')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Txt file"))
    st.title("Predicting Transaction Spend Category")
    st.header('This app is created to predict the type of Transaction using its Narration')


    if add_selectbox == "Online":
        text1 = st.text_area('Enter text')
        output = ""
        if st.button("Predict"):
            output = predict(text1)
            #output = str(output[0]) # since its a list, get the 1st item
            st.success(f"The Narration text is {output}")
            st.balloons()
        elif add_selectbox == "Txt file":
        	output = ""
        	file_buffer = st.file_uploader("Upload text file for new item", type=["txt"])
        	if st.button("Predict"):
        		text_narration = file_buffer.read()

        # in the latest stream-lit version ie. 68, we need to explicitly convert bytes to text
        st_version = st.__version__ # eg 0.67.0
        versions = st_version.split('.')
        if int(versions[1]) > 67:
        	text_narration = text_narration.decode('utf-8')
        	print(text_narration)
        	output = predict(text_narration)
        	#output = str(output[0])
        	st.success(f"The Narration text is {output}")
        	st.balloons()


if __name__ == "__main__":
	load_model()
	run()