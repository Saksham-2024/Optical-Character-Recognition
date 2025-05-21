import tensorflow as tf
import numpy as np
import typing
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer

class ImageToWordModel:
    def __init__(self, model_path: str, char_list: typing.Union[str, list]):
        tf.keras.config.enable_unsafe_deserialization()

        # Dummy CERMetric
        class DummyCERMetric(tf.keras.metrics.Metric):
            def __init__(self, name='CER', **kwargs):
                super().__init__(name=name, **kwargs)
            def update_state(self, y_true, y_pred, sample_weight=None): pass
            def result(self): return 0.0
            def reset_states(self): pass

        # Dummy WERMetric
        class DummyWERMetric(tf.keras.metrics.Metric):
            def __init__(self, name='WER', **kwargs):
                super().__init__(name=name, **kwargs)
            def update_state(self, y_true, y_pred, sample_weight=None): pass
            def result(self): return 0.0
            def reset_states(self): pass

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "CTCloss": lambda y_true, y_pred: y_pred,  # Dummy loss
                "CERMetric": DummyCERMetric,
                "WERMetric": DummyWERMetric,
            }
        )
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, 384, 96)
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.predict(image_pred)
        text = ctc_decoder(preds, self.char_list)[0]
        return text



# Example usage:
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import cv2
    import yaml

    with open("Models/04_sentence_recognition/202504150000/configs.yaml", "r") as file:
        configs = yaml.safe_load(file)

    model_path = configs["model_path"] + "/model.keras"
    vocab = configs["vocab"]

    model = ImageToWordModel(model_path=model_path, char_list=vocab)

    val_csv_path = configs["model_path"] + "/val.csv"
    df = pd.read_csv(val_csv_path).values.tolist()

    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        prediction_text = model.predict(image)

        print("Image: ", image_path)
        print("Label:", label)
        print("Prediction: ", prediction_text)

        display_text = f"Pred: {prediction_text} | GT: {label}"
        image_display = cv2.resize(image, (800, 200))  # Resize for better visibility if needed
        cv2.imshow(display_text, image_display)

        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break
        cv2.destroyAllWindows()

