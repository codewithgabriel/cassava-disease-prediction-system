import os
import io
import time
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# App config
st.set_page_config(page_title="Cassava Disease — Hybrid CNN+RF", layout="wide",
                   initial_sidebar_state="expanded")


# CACHED LOADERS
@st.cache_resource
def load_feature_extractor():
    """Load ResNet50 feature extractor used in the notebook (include_top=False, pooling='avg')."""
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return model

@st.cache_resource
def load_rf_model(path):
    """Load RandomForest classifier saved by the notebook via joblib."""
    if not os.path.exists(path):
        return None
    model = joblib.load(path)
    return model


# PREPROCESSING & PREDICTION
def pil_to_resnet_features(pil_img, extractor, target_size=(224, 224)):
    """Return ResNet50 pooled features for a PIL.Image (RGB)."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = img_to_array(pil_img)  # H,W,C
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # ResNet-specific preprocessing
    features = extractor.predict(arr, verbose=0)  # shape (1, feature_dim)
    return features

def predict_from_pil(pil_img, extractor, rf_model):
    """Return (pred_label, class_probabilities dict) or error if rf_model None."""
    if rf_model is None:
        return None, None
    features = pil_to_resnet_features(pil_img, extractor)
    # if classifier doesn't support predict_proba, fallback to predict
    try:
        probs = rf_model.predict_proba(features)[0]
        classes = rf_model.classes_
    except Exception:
        # fallback: model.predict returns labels only
        pred = rf_model.predict(features)[0]
        classes = rf_model.classes_
        probs = np.zeros_like(classes, dtype=float)
        # Mark predicted label with prob 1
        idx = list(classes).index(pred) if pred in classes else 0
        probs[idx] = 1.0
    # build dict
    class_prob = {str(c): float(p) for c, p in zip(classes, probs)}
    # pick top label
    top_label = max(class_prob.items(), key=lambda x: x[1])[0]
    return top_label, class_prob


# Load models

with st.spinner("Loading models..."):
    extractor = load_feature_extractor()
    rf_model = load_rf_model("models/hybrid_model.pkl")
    time.sleep(0.2)


# Sidebar

st.sidebar.title("CassavaGuard")
st.sidebar.markdown("**Model file:** `hybrid_model.pkl`")
if rf_model is None:
    st.sidebar.warning("`hybrid_model.pkl` not found. Run notebook to create it and place it here.")
else:
    # show a small model info panel if available
    try:
        n_trees = getattr(rf_model, "n_estimators", "Unknown")
        classes = getattr(rf_model, "classes_", [])
        st.sidebar.success(f"RF model loaded — trees: {n_trees}")
        st.sidebar.info(f"Classes: {', '.join(map(str, classes))}")
    except Exception:
        st.sidebar.success("RF model loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("Quick tips:\n- Use clear images with leaf centered.\n- Prefer daylight, avoid heavy shadows.\n- Use sample images row if you don't have a photo.")


# Main layout

st.title("Cassava Disease Detection System 🌱")
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Input")
    # Choose input method using tabs
    input_tab = st.tabs(["Upload Image", "Use Camera", "Sample Images"])

    # --- Upload Image tab
    with input_tab[0]:
        uploaded_file = st.file_uploader("Upload a cassava leaf image (jpg/png)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded image preview", use_container_width=False)
                if st.button("Run prediction on uploaded image"):
                    with st.spinner("Predicting..."):
                        pred_label, probs = predict_from_pil(image, extractor, rf_model)
                    if pred_label is None:
                        st.error("Model not loaded. Place `hybrid_model.pkl` in app folder.")
                    else:
                        st.success(f"Prediction: **{pred_label}**")
                        # show probs in sidebar or below
                        df = pd.DataFrame.from_dict(probs, orient="index", columns=["probability"])
                        df = df.sort_values("probability", ascending=False)
                        st.subheader("Top predictions")
                        for idx, (cls, row) in enumerate(df.head(3).iterrows(), 1):
                            st.write(f"{idx}. **{cls}** : {row['probability']:.2%}")
                        st.subheader("All class probabilities")
                        st.write(df, use_container_width=True, height=600)
                       
                        # plot pie chart
                        pie_fig = df.plot.pie(y="probability", labels=df.index, autopct='%1.1f%%', legend=False, figsize=(5,5)).get_figure()
                        st.pyplot(pie_fig)
                        
                        # show bar chart
                        st.bar_chart(df["probability"])
                        
            except Exception as e:
                st.error(f"Could not read the image. Error: {e}")

    # --- Camera tab
    with input_tab[1]:
        camera_img = st.camera_input("Use your camera to take a photo")
        if camera_img is not None:
            try:
                # camera_img is an UploadedFile-like object
                pil_img = Image.open(camera_img).convert("RGB")
                st.image(pil_img, caption="Captured image", use_container_width=True)
                if st.button("Predict captured image"):
                    with st.spinner("Predicting..."):
                        pred_label, probs = predict_from_pil(pil_img, extractor, rf_model)
                    if pred_label is None:
                        st.error("Model not loaded. Place `hybrid_model.pkl` in app folder.")
                    else:
                        st.success(f"Prediction: **{pred_label}**")
                        df = pd.DataFrame.from_dict(probs, orient="index", columns=["probability"])
                        df = df.sort_values("probability", ascending=False)
                        st.subheader("Top predictions")
                        for idx, (cls, row) in enumerate(df.head(3).iterrows(), 1):
                            st.write(f"{idx}. **{cls}** — {row['probability']:.2%}")
                        st.subheader("All class probabilities")
                        st.bar_chart(df["probability"])
            except Exception as e:
                st.error(f"Error processing camera image: {e}")

    # --- Sample Images tab
    
    with input_tab[2]:
        df = pd.DataFrame()  # empty default
        st.write("Quick test images (place images inside the `sample_images/` folder).")
        sample_dir = "sample_images"
        if not os.path.exists(sample_dir):
            st.info("No sample_images/ folder found. Create a folder named `sample_images` and add a few cassava leaf images for quick testing.")
        else:
            files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if len(files) == 0:
                st.info("`sample_images/` is empty. Add some images to it.")
            else:
                # Display sample images in a row (wrap if more than 5)
                per_row = 5
                for i in range(0, len(files), per_row):
                    row_files = files[i:i+per_row]
                    cols = st.columns(len(row_files))
                    for c, fname in zip(cols, row_files):
                        with c:
                            path = os.path.join(sample_dir, fname)
                            try:
                                img = Image.open(path).convert("RGB")
                                st.image(img, caption=fname, use_container_width=True)
                                if st.button(f"Predict: {fname}"):
                                    with st.spinner("Predicting..."):
                                        pred_label, probs = predict_from_pil(img, extractor, rf_model)
                                    if pred_label is None:
                                        st.error("Model not loaded. Place `hybrid_model.pkl` in app folder.")
                                    else:
                                        st.success(f"Prediction: **{pred_label}**")
                                        df = pd.DataFrame.from_dict(probs, orient="index", columns=["probability"])
                                        df = df.sort_values("probability", ascending=False)
                            except Exception as e:
                                st.write("Error loading image:", e)
        if not df.empty:
            st.subheader("Last prediction probabilities")
            st.write(df.head(5), use_container_width=True, height=600)
            
            
            # plot pie chart
            pie_fig = df.plot.pie(y="probability", labels=df.index, autopct='%1.1f%%', legend=False, figsize=(5,5)).get_figure()
            st.pyplot(pie_fig)
            
            # Show bar chart
            st.subheader("Class probabilities")
            st.bar_chart(df["probability"])


with col2:
    st.header("Results & Model Info")
    if rf_model is None:
        st.error("RandomForest model (`hybrid_model.pkl`) not found.")
           
    else:
        # show small model stats
        st.metric("RF trees", getattr(rf_model, "n_estimators", "Unknown"))
        try:
            importances = rf_model.feature_importances_
            st.write("Feature dimension (RF input):", importances.shape[0])
        except Exception:
            pass

        st.markdown("**Model performance:**")
        st.write("- Hybrid model accuracy: **~0.89**")
        st.write("- Macro-average AUC reported: **0.98**")

    st.markdown("---")
    st.subheader("Interpretation tips")
    st.write(
        "• High probability on a class means the model is confident.\n\n"
        "• If probabilities are spread out (all low), the input may be unclear or out-of-distribution.\n\n"
        "• Use the sample images to sanity-check the app if you don't have a photo."
    )

