# %%
import io
import ipywidgets as widgets
from IPython.display import display, Image as IPImage
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = load_model('./checkpoints/checkpoints_ours_2/weights.1716893020.hdf5')


uploader = widgets.FileUpload(accept='image/*', multiple=False)
display(uploader)

output = widgets.Output()


class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def on_data_change(change):
    output.clear_output()
    if uploader.value:
        uploaded_file = uploader.value[0]

        img_data = io.BytesIO(uploaded_file['content'])

        image = Image.open(img_data).convert('L')
        image = ImageOps.invert(image)
        image = image.resize((28, 28))

        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0

        print(image_array.shape)

        prediction = model.predict(image_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]

        display(IPImage(data=img_data.getvalue()))
        with output:
            print(f"Predicted class: {predicted_class}")


uploader.observe(on_data_change, names='value')
display(output)
