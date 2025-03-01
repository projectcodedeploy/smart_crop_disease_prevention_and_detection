import streamlit as st
import tensorflow as tf
import numpy as np

# Set Page Configuration
st.set_page_config(page_title="Plant Disease Recognition 🌱", page_icon="🌿", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #e6ffe6;
        }
        .main-title {
            color: #228B22;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .sub-title {
            color: #006400;
            text-align: center;
            font-size: 20px;
        }
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 8px;
            padding: 10px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('plant_trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Disease Prevention Measures
prevention_data = {
    "Tomato___Early_blight": "🌱 Use copper-based fungicides and avoid overhead watering.",
    "Tomato___Late_blight": "🌿 Remove infected leaves and use resistant plant varieties.",
    "Tomato___Leaf_Mold": "💨 Increase air circulation and use sulfur-based sprays.",
    "Corn_(maize)___Common_rust": "🌽 Apply fungicide and use disease-resistant hybrids.",
    "Potato___Late_blight": "🥔 Use certified seeds and avoid excessive irrigation.",
    "Apple___Apple_scab": "🍏 Prune infected branches and apply fungicides.",
    "Grape___Black_rot": "🍇 Remove infected vines and use protective sprays.",
    "Pepper,_bell___Bacterial_spot": "🫑 Use copper sprays and plant resistant varieties.",
}

# Initialize session state
if "predicted_disease" not in st.session_state:
    st.session_state.predicted_disease = None

if "show_prevention" not in st.session_state:
    st.session_state.show_prevention = False

# Sidebar
st.sidebar.title("🌿 Dashboard")
app_mode = st.sidebar.radio("Select Page", ["🏠 Home", "ℹ️ About", "🔬 Disease Recognition"])

# Home Page
if app_mode == "🏠 Home":
    st.markdown('<h1 class="main-title">🌱 PLANT DISEASE RECOGNITION SYSTEM 🌱</h1>', unsafe_allow_html=True)
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown('<h3 class="sub-title">🚀 Upload an image to detect plant diseases!</h3>', unsafe_allow_html=True)

# About Page
elif app_mode == "ℹ️ About":
    st.header("About 🌾")
    st.info("This dataset consists of 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes.")

# Prediction Page
elif app_mode == "🔬 Disease Recognition":
    st.markdown('<h1 class="main-title">🔬 Disease Recognition</h1>', unsafe_allow_html=True)
    
    test_image = st.file_uploader("📤 Upload an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image and st.button("🖼 Show Image"):
        st.image(test_image, use_column_width=True)
    
    if test_image and st.button("🔍 Predict"):
        with st.spinner("🧠 Analyzing... Please wait..."):
            result_index = model_prediction(test_image)
            
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]
            
            st.session_state.predicted_disease = class_name[result_index]
            st.success(f"🌟 Model predicts: **{st.session_state.predicted_disease}**")
            st.session_state.show_prevention = False  # Reset prevention state

    # Show Prevention Measures Button
    if st.session_state.predicted_disease:
        if st.button("🛡 Show Prevention Measures"):
            st.session_state.show_prevention = True

    # Display Prevention Measures
    if st.session_state.show_prevention:
        disease_name = st.session_state.predicted_disease
        prevention_text = prevention_data.get(disease_name, "⚠️ No specific prevention measures available.")
        
        st.subheader("🩺 Prevention Measures")
        st.success(prevention_text)
