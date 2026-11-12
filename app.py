import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import streamlit as st

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")


@st.cache_resource
def load_model():
    try:
        digits = load_digits()
        # Scale 0-16 values to 0.0-1.0 and train on 100% of available data
        X = digits.images.reshape((len(digits.images), -1)) / 16.0
        y = digits.target

        model = MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=500, random_state=42
        )
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Converts an uploaded image into a centered, high-contrast 8x8 array."""
    img_gray = image.convert("L")
    img_array = np.array(img_gray, dtype=np.float32)

    # 1. Invert if background is light so digit is bright on a dark background
    if np.mean(img_array) > 128:
        img_array = 255.0 - img_array

    # 2. Remove faint background shadows and stretch contrast to 0.0 - 1.0
    img_array[img_array < 30] = 0
    if img_array.max() > 0:
        img_array = img_array / img_array.max()

    # 3. Auto-crop to the digit's bounding box and pad into a centered square
    coords = np.argwhere(img_array > 0.1)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = img_array[y0:y1, x0:x1]

        # Add a ~20% margin around the longest side
        size = int(max(cropped.shape) * 1.4)
        padded = np.zeros((size, size), dtype=np.float32)
        y_offset = (size - cropped.shape) // 2
        x_offset = (size - cropped.shape) // 2
        padded[
            y_offset : y_offset + cropped.shape,
            x_offset : x_offset + cropped.shape,
        ] = cropped
        img_array = padded

    # 4. Resize to 8x8 and normalize again after resampling
    img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
    img_resized = img_pil.resize((8, 8), Image.Resampling.LANCZOS)
    img_normalized = np.array(img_resized, dtype=np.float32) / 255.0

    if img_normalized.max() > 0:
        img_normalized = img_normalized / img_normalized.max()

    img_flat = img_normalized.flatten().reshape(1, -1)
    return img_normalized, img_flat


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
        img_normalized, img_flat = preprocess_image(image)

        # Display original and processed 8x8 model input side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(
                img_normalized,
                caption="Processed 8x8 Model Input",
                use_container_width=True,
                clamp=True,
            )

        if model is not None:
            prediction = model.predict(img_flat)[0]
            st.subheader(f"Prediction: **{prediction}**")

            probs = model.predict_proba(img_flat)[0]
            st.write("### Probabilities")
            st.bar_chart({str(i): float(prob) for i, prob in enumerate(probs)})
        else:
            st.subheader("Using fallback recognition")
            digit_guess = int(np.argmax(np.sum(img_normalized, axis=0)) % 10)
            st.subheader(f"Fallback Prediction: **{digit_guess}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of a handwritten digit (0-9)
2. The app automatically crops, centers, and resizes it to 8x8 pixels
3. The AI model predicts the digit
4. For best results:
   - Plain light or dark background
   - Thick, clear digit strokes
   - Minimal background clutter or shadows
""")
