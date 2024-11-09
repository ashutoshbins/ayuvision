import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

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
        st.write(f"{condition}: {probability:.2f}")
