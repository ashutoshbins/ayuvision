import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

# Configure the gemini-pro model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="skin_condition_model.tflite")
interpreter.allocate_tensors()

# Define the labels for the conditions
labels_dict = {
    0: 'blackhead',
    1: 'acne',
    2: 'dark_circles',
    3: 'dry_skin',
    4: 'hyperpigmentation'
}

# Preprocess image function
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Run inference using TensorFlow Lite model
def make_prediction(img, interpreter):
    img_processed = preprocess_image(img)

    # Get input tensor and set input data
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_processed)

    # Run inference
    interpreter.invoke()

    # Get the output tensor and return the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top_indices = np.argsort(output_data)[-2:][::-1]
    top_conditions = [(labels_dict[idx], output_data[idx]) for idx in top_indices]
    return top_conditions

# Function to generate a warning message using gemini-pro model
def generate_warning_message(top_conditions):
    # Get the top condition (you can use any condition, here I choose the first one)
    condition, probability = top_conditions[0]

    # Create the prompt using the top predicted condition
    prompt = f"Generate a warning type message of less than 100 words for the condition '{condition}' asking the person not to take lightly these conditions and supporting it with medical records if possible. Also, mention how Ayurvedic treatment can help him but don't provide the solution here and add this line at the end  use the Skin Veda app for further assistance"

    # Generate content using gemini-pro model
    response = model.generate_content(f"As an Ayurvedic doctor, give advice for face care: {prompt}")
    generated_content = response.text.strip()
    return generated_content

# Streamlit UI
st.title("Skin Condition Detector")
st.markdown("Upload an image to analyze the skin condition.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image
    image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", channels="BGR", use_column_width=True)

    # Make prediction
    top_conditions = make_prediction(img, interpreter)
    st.subheader("Top Detected Conditions")
    for condition, probability in top_conditions:
        st.write(f"{condition}: {probability:.3f}")
    
    # Generate and display the warning message
    warning_message = generate_warning_message(top_conditions)
    st.markdown(warning_message)
