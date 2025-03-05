import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Set Page Configuration
st.set_page_config(page_title="Plant Disease Recognition 🌱", page_icon="🌿", layout="wide")

# Custom CSS for Styling
# Custom CSS for Styling & Centering Buttons

st.markdown("""
    <style>
        body, .stApp {
            background-color: #121212;
            color: #ffffff;
        }
        .stSidebar {
            background-color: #1E1E1E;
        }
    </style>
""", unsafe_allow_html=True)

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


HISTORY_FILE = "Crop_disease_histroy.csv"

def save_to_csv(result):
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)  # Use concat instead of append
    else:
        df = pd.DataFrame([result])
    
    df.to_csv(HISTORY_FILE, index=False)

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
    "Tomato___Early_blight": "Use copper-based fungicides and avoid overhead watering.",
    "Tomato___Late_blight": "Remove infected leaves and use resistant plant varieties.",
    "Tomato___Leaf_Mold": "Increase air circulation and use sulfur-based sprays.",
    "Corn_(maize)___Common_rust": "Apply fungicide and use disease-resistant hybrids.",
    "Potato___Late_blight": "Use certified seeds and avoid excessive irrigation.",
    "Apple___Apple_scab": "Prune infected branches and apply fungicides.",
    "Grape___Black_rot": "Remove infected vines and use protective sprays.",
    "Pepper,_bell___Bacterial_spot": "Use copper sprays and plant resistant varieties.",
    "Apple___Black_rot": "Remove and destroy infected fruit; apply fungicides at early stages.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees to break the disease cycle; use fungicide sprays.",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "Rotate crops and use resistant hybrids to prevent infection.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant varieties and apply fungicides at early symptoms.",
    "Grape___Esca_(Black_Measles)": "Prune infected parts and apply fungicides before bud break.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Ensure proper vineyard sanitation and use fungicidal sprays.",
    "Orange___Haunglongbing_(Citrus_greening)": "Control psyllid insect populations and remove infected trees.",
    "Peach___Bacterial_spot": "Use copper-based sprays and plant resistant peach varieties.",
    "Potato___Early_blight": "Practice crop rotation and use fungicides like chlorothalonil.",
    "Squash___Powdery_mildew": "Apply sulfur-based sprays and improve air circulation in fields.",
    "Strawberry___Leaf_scorch": "Avoid overhead watering and remove infected leaves.",
    "Tomato___Bacterial_spot": "Use disease-free seeds and apply copper sprays during early infection.",
    "Tomato___Septoria_leaf_spot": "Remove lower leaves to improve airflow; use fungicides if needed.",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "Introduce natural predators like ladybugs and use neem oil.",
    "Tomato___Target_Spot": "Apply copper-based fungicides and avoid excessive moisture.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use insect netting to prevent whitefly transmission and plant resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "Disinfect tools and hands; avoid tobacco products near tomato plants."
}

# Initialize session state
if "predicted_disease" not in st.session_state:
    st.session_state.predicted_disease = None

if "show_prevention" not in st.session_state:
    st.session_state.show_prevention = False

# Sidebar
st.sidebar.title("🌿 Agro Detect")
app_mode = st.sidebar.radio("Select Page", ["🏠 Home", "ℹ️ About", "🔬 Disease Recognition","📜 History", "📞 Contact"])

# Home Page
if app_mode == "🏠 Home":
    st.markdown('<h1 class="main-title">🌱 PLANT DISEASE RECOGNITION SYSTEM 🌱</h1>', unsafe_allow_html=True)
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown('<h3 style="text-align: center;">🚀 *Upload an image to detect plant diseases!*</h3>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>🚀 *Identify plant diseases and get prevention strategies instantly!*</h3>",unsafe_allow_html=True)


# About Page
elif app_mode == "ℹ️ About":
    st.header("About 🌾")
    st.info("""
🔍 How It Works?\n
1️⃣ Upload an image of a plant leaf.\n
2️⃣ The AI model analyzes the image and identifies the disease.\n
3️⃣ Get real-time results along with disease prevention measures.

✨ Key Features\n
✅ Accurate Crop Disease Detection – Uses advanced deep learning to classify diseases.\n
✅ Instant Prevention Strategies – Provides best practices to manage plant diseases.\n
✅ User-friendly Web Interface – Designed for farmers, researchers, and agronomists.\n

🛠 Technology Used\n
Deep Learning Model: MobileNetV2 (CNN) – Optimized for plant disease detection.\n
Framework: TensorFlow/Keras for AI modeling, OpenCV for image processing.\n
Web App: Streamlit for an intuitive user experience.\n
Deployment: Cloud or local-based system for scalability.\n
""")
    
# Prediction Page
elif app_mode == "🔬 Disease Recognition":
    st.markdown('<h1 class="main-title">🔬 Disease Recognition</h1>', unsafe_allow_html=True)
    
    test_image = st.file_uploader("📤 Upload an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image and st.button("🖼 Show Image"):
        st.image(test_image, use_column_width=True)
    
    if test_image and st.button("🔍 Predict"):
        with st.spinner("🧠 Analyzing... Please wait..."):
            result_index = model_prediction(test_image)
            
            class_name = {
                0: "Apple___Apple_scab",
                1: "Apple___Black_rot",
                2: "Apple___Cedar_apple_rust",
                3: "Apple___healthy",
                4: "Blueberry___healthy",
                5: "Cherry_(including_sour)___Powdery_mildew",
                6: "Cherry_(including_sour)___healthy",
                7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                8: "Corn_(maize)___Common_rust_",
                9: "Corn_(maize)___Northern_Leaf_Blight",
                10: "Corn_(maize)___healthy",
                11: "Grape___Black_rot",
                12: "Grape___Esca_(Black_Measles)",
                13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                14: "Grape___healthy",
                15: "Orange___Haunglongbing_(Citrus_greening)",
                16: "Peach___Bacterial_spot",
                17: "Peach___healthy",
                18: "Pepper,_bell___Bacterial_spot",
                19: "Pepper,_bell___healthy",
                20: "Potato___Early_blight",
                21: "Potato___Late_blight",
                22: "Potato___healthy",
                23: "Raspberry___healthy",
                24: "Soybean___healthy",
                25: "Squash___Powdery_mildew",
                26: "Strawberry___Leaf_scorch",
                27: "Strawberry___healthy",
                28: "Tomato___Bacterial_spot",
                29: "Tomato___Early_blight",
                30: "Tomato___Late_blight",
                31: "Tomato___Leaf_Mold",
                32: "Tomato___Septoria_leaf_spot",
                33: "Tomato___Spider_mites Two-spotted_spider_mite",
                34: "Tomato___Target_Spot",
                35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                36: "Tomato___Tomato_mosaic_virus",
                37: "Tomato___healthy"
            }
            
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
        

        # Ensure session state is initialized
        if "history" not in st.session_state:
            st.session_state.history = []


        result = {
                "Image": test_image,
                "Prediction": disease_name,
                "Prevention": prevention_text
            }


        st.session_state.history.append(result)
        save_to_csv(result)
        
        st.success(f"🌟 Model predicts: **{disease_name}**")
        st.subheader("🩺 Prevention Measures")
        st.success(prevention_text)

# History Page
elif app_mode == "📜 History":
    st.title("📜 Prediction History")

    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)

            # Check if file is empty
            if df.empty:
                st.info("⚠️ No predictions found. Start detecting diseases!")
            else:
                st.table(df)  # Display history table
        except Exception as e:
            st.error(f"❌ Error reading history: {e}")
    else:
        st.info("⚠️ No predictions found. Start detecting diseases!")

elif app_mode == "📞 Contact":
    st.markdown("<h1 style='text-align: center;'>📞 Contact Us</h1>", unsafe_allow_html=True)
    st.markdown("""
        **Email:** agriculturedetect@gmail.com \n
        **GitHub:** https://github.com/projectcodedeploy/smart_crop_disease_prevention_and_detection.git
    """)
