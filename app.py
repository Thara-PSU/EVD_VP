import streamlit as st
import pandas as pd
import numpy as np
import pickle
#from sklearn.ensemble import RandomForestClassifier

def main():
    from PIL import Image
    #image_hospital = Image.open('Neuro1.png')
    image_ban = Image.open('Neuro1.png')
    st.image(image_ban, use_column_width=False)
    #st.sidebar.image(image_hospital)
if __name__ == '__main__':
    main()
 

st.write("""
# Machine learning for predicting VP shunt operation in hydrocephalus patient who undergone EVD (for unseen data)

""")
st.write ("Tunthanathip et al.")

#st.write("""
### Performances of various algorithms from the training dataset [Link](https://pedtbi-train.herokuapp.com/)
#""")

#st.write ("""
### Labels of input features
#1.GCSer (Glasgow Coma Scale score at ER): range 3-15

#2.Hypotension (History of hypotension episode): 0=no , 1=yes

#3.pupilBE (pupillary light refelx at ER): 0=fixed both eyes, 1= fixed one eye, 2=React both eyes

#4.SAH (Subarachnoid hemorrhage on CT of the brain): 0=no, 1=yes

#""")


st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/Thara-PSU/EVD_VP/blob/main/example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if  uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        A = st.sidebar.slider('A', 10.0, 80.0, 45.5)
        B = st.sidebar.slider('B', 10.0, 80.0, 45.5)
        C = st.sidebar.slider('C', 10.0, 80.0, 45.5)
        D = st.sidebar.slider('D', 10.0, 80.0, 45.5)
        Age = st.sidebar.slider('Age (year)', 0, 100, 1)
        HEADACHE = st.sidebar.slider('HEADACHE(0=no, 1=yes)', 0, 1, 0)
        Brain_tumor = st.sidebar.slider('Brain_tumor (0=no, 1=yes)', 0, 1, 0)
        Aneurysm = st.sidebar.slider('Ruptured aneurysm (0=no, 1=yes)', 0, 1, 0)
        Stroke = st.sidebar.slider('Hemorrhagic stroke (0=no, 1=yes)', 0, 1, 0)
        data = {'A': A,
                'B': B,
                'C': C,
                'D': D,
                'Age': Age,
                'HEADACHE': HEADACHE,
                'Brain_tumor': Brain_tumor,
                'Aneurysm': Aneurysm,
                'Stroke': Stroke,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
GBM_raw = pd.read_csv('train.2020.csv')
GBM = GBM_raw.drop(columns=['FINAL_VP'])
df = pd.concat([input_df,GBM],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['HEADACHE','Brain_tumor','Aneurysm','Stroke']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)


# Reads in saved classification model
#f = open('evd_vp/vp_lr_clf.pkl', 'rb')
#load_clf  = pickle.load(f)
#f.close()
load_clf = pickle.load(open('vp_lr_clf.pkl', 'rb'))

#pickle_file = open("evd_vp\\vp_lr_clf", "rb")
#load_clf  = pickle.load(pickle_file)

#with open("evd_vp/vp_lr_clf.pkl", 'rb') as pfile:  
#    load_clf=pickle.load(pfile)

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.write("""# Prediction Probability""")
#st.subheader('Prediction Probability')
st.write(prediction_proba)


st.subheader('Class Labels and their corresponding index number')

label_name = np.array(['Successful EVD removal','VP_shunt'])
st.write(label_name)
# labels -dictionary
names ={0:'Successful EVD removal',
1: 'VP shunt'}

#st.write("""# Prediction""")
#st.subheader('Prediction')
#two_year_survival = np.array(['Negative','Positive'])
#st.write(two_year_survival[prediction])
st.write("""# Prediction is positive when probability of the class 1 is more than 0.5""")

st.write ("""
### Tunthanathip et al. Prince of Songkla University

""")

#st.markdown( "  [Random forest] (https://ct-pedtbi-test-rf.herokuapp.com/) ")
#st.markdown( "  [Logistic Regression] (https://ct-pedtbi-test-ln.herokuapp.com/) ")
#st.markdown( "  [Neural Network] (https://ct-pedtbi-test-nn.herokuapp.com/) ")
#st.markdown( "  [K-Nearest Neighbor (kNN)] (https://pedtbi-test-knn.herokuapp.com/) ")
#st.markdown( "  [naive Bayes] (https://ct-pedtbi-test-nb.herokuapp.com/) ")
#st.markdown( "  [Support Vector Machines] (https://ct-pedtbi-test-svm.herokuapp.com/) ")
#st.markdown( "  [Gradient Boosting Classifier] (https://pedtbi-test-gbc.herokuapp.com/) ")
#st.markdown( "  [Nomogram] (https://psuneurosx.shinyapps.io/ct-pedtbi-nomogram/) ")

st.write ("""
### [Home](https://sites.google.com/psu.ac.th/psuneurosurgery?pli=1)

""")
