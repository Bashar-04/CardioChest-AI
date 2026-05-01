import os
import json
import uuid
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from werkzeug.utils import secure_filename


# =========================
# Flask App Config
# =========================
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# Load Config
# =========================
with open("model_config.json", "r") as f:
    config = json.load(f)

LABELS = config["labels"]
ACTIVE_LABELS = config.get("active_labels", LABELS)
THRESHOLD = config["threshold"]
IMG_SIZE = config["img_size"]


# =========================
# Load Model
# =========================
def load_cardio_model():
    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(5, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.load_weights("cardiochest_weights.weights.h5")
    return model


model = load_cardio_model()


# =========================
# Helper Functions
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_likely_xray(image):
    """
    Simple validation to reject non-X-ray images.
    Checks if the image is mostly grayscale and has radiology-like intensity.
    """

    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    color_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))

    max_channel = np.max(arr, axis=2)
    min_channel = np.min(arr, axis=2)
    saturation = np.mean(max_channel - min_channel)

    gray = np.mean(arr, axis=2)
    brightness = np.mean(gray)

    is_grayscale_like = color_diff < 18 and saturation < 25
    valid_brightness = 25 < brightness < 230

    return is_grayscale_like and valid_brightness


def preprocess_image(image):
    image = image.convert("RGB")
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def get_confidence_level(confidence):
    if confidence >= 0.65:
        return "High", "high"
    elif confidence >= 0.40:
        return "Medium", "medium"
    else:
        return "Low", "low"


def get_explanation(label):
    explanations = {
        "Cardiomegaly": "The model detected visual patterns that may be consistent with an enlarged cardiac silhouette.",
        "Edema": "The model detected visual patterns that may indicate fluid-related changes within the lung fields.",
        "Effusion": "The model detected visual patterns that may suggest fluid accumulation around the lungs.",
        "No Finding": "The model did not detect clear abnormal patterns above the selected threshold."
    }

    return explanations.get(
        label,
        "The model detected visual patterns that require further review."
    )


def analyze_prediction(prediction, labels, active_labels, threshold):
    all_probs = {
        label: float(prob)
        for label, prob in zip(labels, prediction)
    }

    active_probs = {
        label: all_probs[label]
        for label in active_labels
        if label in all_probs
    }

    disease_probs = {
        label: prob
        for label, prob in active_probs.items()
        if label != "No Finding"
    }

    sorted_diseases = sorted(
        disease_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top1_label, top1_prob = sorted_diseases[0]
    top2_label, top2_prob = sorted_diseases[1]

    no_finding_prob = active_probs.get("No Finding", 0.0)

    if top1_prob < threshold:
        return {
            "status": "normal",
            "title": "No clear abnormality detected",
            "primary_label": "No Finding",
            "primary_prob": no_finding_prob,
            "secondary_label": None,
            "secondary_prob": 0.0,
            "details": active_probs
        }

    return {
        "status": "abnormal",
        "title": f"Possible {top1_label}",
        "primary_label": top1_label,
        "primary_prob": top1_prob,
        "secondary_label": top2_label,
        "secondary_prob": top2_prob,
        "details": active_probs
    }


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        img_size=IMG_SIZE,
        threshold=THRESHOLD
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["image"]

    if file.filename == "":
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        return render_template(
            "result.html",
            error=True,
            error_title="Unsupported File Type",
            error_message="Please upload a PNG, JPG, or JPEG image only.",
            image_path=None
        )

    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{extension}"

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(save_path)

    image = Image.open(save_path)

    if not is_likely_xray(image):
        return render_template(
            "result.html",
            error=True,
            error_title="Invalid Image Type",
            error_message="The uploaded image does not appear to be a valid grayscale chest X-ray. Please upload a medical chest X-ray image only.",
            image_path=url_for("static", filename=f"uploads/{unique_filename}")
        )

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0]

    analysis = analyze_prediction(
        prediction,
        LABELS,
        ACTIVE_LABELS,
        THRESHOLD
    )

    level_text, level_class = get_confidence_level(analysis["primary_prob"])

    disease_details = {
        k: v
        for k, v in analysis["details"].items()
        if k != "No Finding"
    }

    sorted_details = sorted(
        disease_details.items(),
        key=lambda x: x[1],
        reverse=True
    )

    all_probs = {
        label: float(prob)
        for label, prob in zip(LABELS, prediction)
    }

    all_probs_sorted = sorted(
        all_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return render_template(
        "result.html",
        error=False,
        analysis=analysis,
        level_text=level_text,
        level_class=level_class,
        explanation=get_explanation(analysis["primary_label"]),
        sorted_details=sorted_details,
        all_probs_sorted=all_probs_sorted,
        image_path=url_for("static", filename=f"uploads/{unique_filename}")
    )


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)