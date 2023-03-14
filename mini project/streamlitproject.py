import streamlit as st
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Iris Flower Species Classification")

st.write("""
         This app predicts the species of an iris flower based on its 
         sepal length, sepal width, petal length, and petal width.
         """)

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create a SVM classifier
svc_model = SVC(kernel='linear', C=1, gamma=1)

# Train the classifier on the training data
svc_model.fit(X_train, Y_train)

# Make predictions on the testing data
Y_pred = svc_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(Y_test, Y_pred)

# Display the model's accuracy
st.write("Accuracy:", accuracy)

# Create a function to predict the iris species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    species = svc_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return species[0]

# Create a form to input the flower data
sepal_length = st.slider("Sepal Length", 0.0, 10.0, 5.4, 0.1)
sepal_width = st.slider("Sepal Width", 0.0, 10.0, 3.4, 0.1)
petal_length = st.slider("Petal Length", 0.0, 10.0, 1.3, 0.1)
petal_width = st.slider("Petal Width", 0.0, 10.0, 0.2, 0.1)

# Make a prediction and display the result
species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
st.write("The predicted species is:", iris.target_names[species])
