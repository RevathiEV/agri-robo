import os


def main():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required to convert the model to TFLite. "
            "Run this script on a machine where TensorFlow is installed."
        ) from exc

    project_root = os.path.dirname(os.path.abspath(__file__))
    source_candidates = [
        os.path.join(project_root, "tomato_disease_model_best.h5"),
        os.path.join(project_root, "tomato_disease_model.h5"),
    ]
    source_path = next((path for path in source_candidates if os.path.exists(path)), None)

    if source_path is None:
        raise SystemExit(
            "No source Keras model found. Expected tomato_disease_model_best.h5 "
            "or tomato_disease_model.h5 in the project root."
        )

    output_path = os.path.join(project_root, "tomato_disease_model.tflite")

    print(f"Loading source model: {source_path}")
    model = tf.keras.models.load_model(source_path, compile=False)

    print("Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, "wb") as output_file:
        output_file.write(tflite_model)

    print(f"TFLite model written to: {output_path}")


if __name__ == "__main__":
    main()
