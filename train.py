import json
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
VALIDATION_DIR = ROOT / "data" / "validation"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "clasificador.keras"
CLASSES_PATH = MODEL_DIR / "clases.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 20
SEED = 42
EXPECTED_CLASSES = {"lapiz", "lapicera"}


def main() -> None:
    if not TRAIN_DIR.is_dir() or not VALIDATION_DIR.is_dir():
        raise FileNotFoundError(
            "No se encontraron data/train y data/validation. "
            "Revisá la estructura de carpetas del proyecto."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names
    if set(class_names) != EXPECTED_CLASSES:
        raise ValueError(
            "Se esperaban exactamente las carpetas 'lapiz' y 'lapicera' "
            f"en data/train; se encontraron: {class_names}"
        )
    if validation_ds.class_names != class_names:
        raise ValueError(
            "Las clases de data/validation no coinciden con las de data/train."
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="aumento_de_datos",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    validation_loss, validation_accuracy = model.evaluate(
        validation_ds, verbose=0
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    CLASSES_PATH.write_text(
        json.dumps(
            {"class_names": class_names, "image_size": IMAGE_SIZE},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Modelo guardado en: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Clases guardadas en: {CLASSES_PATH.relative_to(ROOT)}")
    print(f"Exactitud en validación: {validation_accuracy:.1%}")
    print(
        "Esta métrica corresponde a las fotos reservadas para validación; "
        "todavía no es una prueba independiente con fotos nuevas."
    )


if __name__ == "__main__":
    main()
