import streamlit as st
import tensorflow as tf
import numpy as np

# Set Page Configuration
st.set_page_config(page_title="Plant Disease Recognition 🌱", page_icon="🌿", layout="wide")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# Function to update page
def set_page(page_name):
    st.session_state.page = page_name

# Custom CSS for Styling & Centering Buttons
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .navbar {
            display: flex;
            justify-content: center; /* Centering buttons */
            align-items: center;
            gap: 20px;  /* Adjust spacing between buttons */
            padding: 15px;
            background-color: #1E1E1E;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
            animation: fadeIn 1s ease-in-out;
            width: 100%;
        }
        .nav-button {
            font-size: 16px;
            font-weight: bold;
            background: none;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            padding: 8px 15px;
        }
        .nav-button:hover {
            background-color: #4CAF50;
            color: black;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

## Navigation Bar (Less Spacing Between Buttons)
# Create empty columns to center the buttons
col_empty1, col1, col2, col3, col4, col_empty2 = st.columns([2, 1, 1, 1, 1, 2])
with col1:
    if st.button("🏠 Home", key="home"):
        set_page("home")
with col2:
    if st.button("ℹ️ About", key="about"):
        set_page("about")
with col3:
    if st.button("🔬 Disease Recognition", key="disease"):
        set_page("disease")
with col4:
    if st.button("📞 Contact", key="contact"):
        set_page("contact")

# Handle Page Navigation
query_params = st.query_params.get("page", "home")

# Home Page
if query_params == "home":
    st.markdown("<h1 style='text-align: center;'>🌱 PLANT DISEASE RECOGNITION SYSTEM 🌱</h1>", unsafe_allow_html=True)
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("<h3 style='text-align: center;'>🚀 Upload an image to detect plant diseases!</h3>", unsafe_allow_html=True)

# About Page
elif query_params == "about":
    st.header("About 🌾")
    st.info("This dataset consists of 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes.")

# Disease Recognition Page
elif query_params == "disease":
    st.markdown("<h1 style='text-align: center;'>🔬 Disease Recognition</h1>", unsafe_allow_html=True)
    test_image = st.file_uploader("📤 Upload an Image:", type=["jpg", "png", "jpeg"])
    if test_image and st.button("🔍 Predict"):
        with st.spinner("🧠 Analyzing... Please wait..."):
            model = tf.keras.models.load_model('plant_trained_model.keras')
            image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])
            prediction = model.predict(input_arr)
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
            result_index = np.argmax(prediction)
            st.success(f"🌟 Model predicts: **{class_name[result_index]}**")

# Contact Page
elif query_params == "contact":
    st.markdown("<h1 style='text-align: center;'>📞 Contact Us</h1>", unsafe_allow_html=True)
    st.markdown("""
        **Email:** support@agrodetect.com  
    """)

