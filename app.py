import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import streamlit as st

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")


@st.cache_resource
def load_model():
    try:
        digits = load_digits()
        # Scale 0-16 values to 0.0-1.0
        X = digits.images.reshape((len(digits.images), -1)) / 16.0
        y = digits.target
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=500, random_state=42
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


model = load_model()

if model is None:
    st.warning("Could not load model. Using fallback recognition.")
else:
    st.success("Model loaded successfully!")

uploaded_file = st.file_uploader(
    "Choose an image file", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    try:
        # Convert to grayscale and resize to 8x8 using smooth resampling
        img_gray = image.convert("L")
        img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)

        img_array = np.array(img_resized, dtype=np.float32)

        # Invert if background is light so digit is bright on dark background
        if np.mean(img_array) > 128:
            img_array = 255.0 - img_array

        # Normalize 0-255 to 0.0-1.0 to match training data
        img_normalized = img_array / 255.0
        img_flat = img_normalized.flatten().reshape(1, -1)

        # Display original and 8x8 model input side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(
                img_normalized,
                caption="8x8 Model Input",
                use_container_width=True,
                clamp=True,
            )

        if model is not None:
            prediction = model.predict(img_flat)[0]
            st.subheader(f"Prediction: **{prediction}**")

            probs = model.predict_proba(img_flat)[0]
            st.write("### Probabilities")
            st.bar_chart({str(i): prob for i, prob in enumerate(probs)})
        else:
            st.subheader("Using fallback recognition")
            digit_guess = int(np.argmax(np.sum(img_normalized, axis=0)) % 10)
            st.subheader(f"Fallback Prediction: **{digit_guess}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of a handwritten digit (0-9)
2. The image will be resized to 8x8 pixels
3. AI model will predict the digit
4. For best results:
   - White background
   - Thick black digit
   - Centered digit with some margin
   - Minimal noise
""")
