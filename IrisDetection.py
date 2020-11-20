#Importing the necessary libraries, modules and methods
import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

#Caching the iris dataset method so that we load the iris dataset only once
@st.cache
def get_iris_dataset():
    iris_data = datasets.load_iris()
    return iris_data

#Caching the train model method so that we train the model only once
@st.cache
def train_model(features, target):
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(features, target)
    
    return classifier

#Creating User based sliders so as to adjust the feature values according to user input
def get_features(feature_names, features):
    featureList = []
    
    for index, name in enumerate(feature_names):
        value = st.sidebar.slider(name, float(np.min(features[:, index])), 
                float(np.max(features[:, index])), float(np.median(features[:, index])))

        featureList.append(value)
        
    featureList = np.array(featureList).reshape(1, len(feature_names))
        
    return featureList

#Now running the application 
def main():
    #Headers for the Web App
    st.markdown("## Iris Flower Type Detection App using StreamLit")
    st.write("This app is a learning project to get started with Python Streamlit")
    
    #Creating classifier for the Iris dataset
    iris_data = get_iris_dataset()
    features = iris_data.data
    targets = iris_data.target
    feature_names = iris_data.feature_names

    classifier = train_model(features, targets)
    
    #Creating Sliders for User based Feature values
    st.sidebar.markdown("## User Defined Features")
    featureList = get_features(feature_names, features)
    
    #Displaying the User feature values on the Web App
    st.write("User Features: ")
    st.write(featureList)
    
    #Predicting the result from Iris dataset based on User feature values 
    prediction = classifier.predict(featureList)[0]
    predicted_class = iris_data.target_names[prediction]
    
    #Creating subheader to display the predicted result
    st.subheader("Model Predicted Class {} for the given user inputs".format(predicted_class))
    
#Running the Web App
if __name__ == "__main__":
    main()
    
    