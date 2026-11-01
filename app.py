import json
from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "clasificador.keras"
CLASSES_PATH = ROOT / "models" / "clases.json"
DISPLAY_NAMES = {"lapicera": "Lapicera", "lapiz": "Lápiz"}


@st.cache_resource
def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def main() -> None:
    st.set_page_config(page_title="Clasificador de útiles", page_icon="✏️")
    st.title("¿Lápiz o lapicera?")
    st.write("Subí una foto con un solo objeto visible para clasificarlo.")

    if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
        st.error("Todavía no hay un modelo entrenado para cargar.")
        st.code("uv run python train.py", language="powershell")
        st.stop()

    metadata = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    class_names = metadata.get("class_names")
    image_size = metadata.get("image_size")
    if (
        not isinstance(class_names, list)
        or set(class_names) != set(DISPLAY_NAMES)
        or not isinstance(image_size, list)
        or len(image_size) != 2
        or not all(isinstance(size, int) and size > 0 for size in image_size)
    ):
        st.error("Los metadatos del modelo no tienen un formato válido.")
        st.stop()

    model = load_model(str(MODEL_PATH))
    uploaded_file = st.file_uploader(
        "Elegí una imagen",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is None:
        return

    try:
        with Image.open(uploaded_file) as uploaded_image:
            image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("No se pudo abrir la imagen. Probá con un archivo JPG, PNG o WEBP válido.")
        st.stop()

    st.image(image, caption="Imagen cargada", width="stretch")
    height, width = image_size
    resized_image = image.resize((width, height), Image.Resampling.BILINEAR)
    image_array = tf.keras.utils.img_to_array(resized_image)
    predictions = model(
        tf.convert_to_tensor(image_array[None, ...]),
        training=False,
    ).numpy()[0]

    if len(predictions) != len(class_names):
        st.error("La salida del modelo no coincide con las clases guardadas.")
        st.stop()

    predicted_index = int(tf.argmax(predictions).numpy())
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])
    st.success(f"Categoría: {DISPLAY_NAMES[predicted_class]}")
    st.metric("Confianza estimada", f"{confidence:.1%}")
    st.caption("La confianza es una estimación del modelo, no una garantía.")


main()
