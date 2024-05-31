import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps

# Load model (assuming the path is correct)
model = load_model('./checkpoints/checkpoints_ours_2/weights.1716893020.hdf5')

# Define class labels
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def classify_image(image):
    """Preprocesses the image and predicts the class using the loaded model."""
    data_list = list(image.values())[0]
    image = np.array(data_list).astype(np.uint8)
    print("Data values (unique):", np.unique(image))  # 고유 값 출력
    print("Data values (min, max):", image.min(), image.max())
    print("Shape before processing:", image.shape)

    # Visualize the raw image data
    plt.imshow(image, cmap='gray')
    plt.show()
    
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.title("Input Image")
    plt.show()

    # Convert to PIL Image
    image = Image.fromarray(image).convert('L')  # Convert to grayscale PIL Image
    image = ImageOps.invert(image)  # Invert colors
    image = image.resize((28, 28))  # Resize to 28x28 (model's expected size)
    # Prepare the image for the model
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Normalize pixel values
    print("Shape after processing:", image.shape)

    # Predict the class
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]

    return predicted_class, image


# Create Gradio interface with drawing capabilities
interface = gr.Interface(
    fn=classify_image,
    inputs="sketchpad",
    outputs=[gr.Label(num_top_classes=1),gr.Image()],
    title="EMNIST image classification",
    description="Draw a character on the canvas. The model will predict what it is!"
)

# Launch the interface
interface.launch(share=True)  # Open in a new web browser window