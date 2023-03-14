import streamlit as st
import numpy as np
from tensorflow import keras

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Define a function to preprocess the drawn image
def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = image.astype('float32')
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define the Streamlit app
def app():
    st.title("MNIST Classification App")

    # Allow the user to draw a digit
    canvas = st.sketchpad(height=280, width=280, fill_value=0, key="canvas")

    if st.button("Classify"):
        # Preprocess the drawn image
        image = preprocess_image(canvas.to_image())
        
        # Classify the image using the model
        predictions = model.predict(image)
        predicted_label = np.argmax(predictions[0])
        
        # Display the predicted label
        st.write("The drawn digit is:", predicted_label)

# Run the Streamlit app
if __name__ == '__main__':
    app()
