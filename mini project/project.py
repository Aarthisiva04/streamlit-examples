import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from scipy.special import expit as sigmoid

# Define the neural network model
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        delta3 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, delta3)
        db2 = (1/m) * np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = (1/m) * np.dot(X.T, delta2)
        db1 = (1/m) * np.sum(delta2, axis=0)
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            loss = -np.mean(y * np.log(output))
            if epoch % 100 == 0:
                st.write(f"Epoch {epoch} - Loss: {loss:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Load the MNIST dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df.drop('label', axis=1), train_df['label'], test_size=0.2, random_state=42)

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
test_df = test_df.astype('float32') / 255.0

# Convert the labels to one-hot encoding
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)

# Create an instance of the neural network model
input_size = X_train.shape[1]
hidden_size = 128
output_size = y_train.shape[1]
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
st.write("Training the neural network...")
nn.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Evaluate the model on the validation set
y_pred = nn.predict(X_val)
accuracy = np.mean(y_pred == np.argmax(y_val, axis=1))
st.write(f"Validation set accuracy: {accuracy:.4f}")

# Make predictions on the test set
st.write("Making predictions on the test set...")
test_pred = nn.predict(test_df)

# Convert predictions to DataFrame and save as CSV
test_results = pd.DataFrame({'ImageId': range(1, len(test_pred)+1), 'Label': test_pred})
test_results.to_csv('predictions.csv', index=False)

# Display the predictions in the app
st.write("Digit classification predictions:")
st.table(test_results.head(20))

# Display an example image
st.write("Example image from the test set:")
example_index = st.slider("Select an image index to display", 0, len(test_df)-1)
example_image = np.array(test_df.iloc[example_index]).reshape(28, 28)
st.image(example_image, width=150, caption=f"Label: {test_pred[example_index]}")
