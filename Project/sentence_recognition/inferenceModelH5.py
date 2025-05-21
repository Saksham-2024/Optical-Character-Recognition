import cv2
import typing
import numpy as np
import tensorflow as tf

from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

class ImageToWordModel:
    def __init__(self, model_path: str, char_list: typing.Union[str, list]):
        # Load the Keras model from the .h5 file
        self.model = tf.keras.models.load_model(model_path, custom_objects={"CTCloss": self.CTCloss})
        self.char_list = char_list

    def CTCloss(self, y_true, y_pred):
        return tf.reduce_mean(tf.nn.ctc_loss(y_true, y_pred, logits_time_major=False))

    def predict(self, image: np.ndarray):
        # Resize the image to match the model's input dimensions
        image = ImageResizer.resize_maintaining_aspect_ratio(image, 384, 96)  # Update dimensions if needed
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        # Get predictions
        preds = self.model.predict(image_pred)

        # Decode predictions
        text = ctc_decoder(preds, self.char_list)[0]
        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/04_sentence_recognition/202504141251/configs.yaml")
    model = ImageToWordModel(model_path="Models/04_sentence_recognition/202504141251/saved_model.h5", char_list=configs.vocab)
    df = pd.read_csv("Models/04_sentence_recognition/202504141251/val.csv").values.tolist()

    accum_cer, accum_wer = [], []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print("Image: ", image_path)
        print("Label:", label)
        print("Prediction: ", prediction_text)
        print(f"CER: {cer}; WER: {wer}")

        accum_cer.append(cer)
        accum_wer.append(wer)

        # Display the image with the prediction
        cv2.imshow(prediction_text, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print average CER and WER
    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")