import numpy as np
import tensorflow as tf

from app import model, preprocess_image, PLANT_CLASSES

def test_image(img_path):
    print("\nTesting:", img_path)
    img = preprocess_image(img_path)
    pred = model.predict(img)[0]
    print("Raw probs:", pred)
    print("Argmax index:", int(np.argmax(pred)))
    print("Predicted label:", PLANT_CLASSES[int(np.argmax(pred))])

if __name__ == "__main__":
    test_image(r"C:\Users\SWARAJ\OneDrive\Desktop\ayurspace\dataset\neem\Image_1.jpg")        # neem ki image ka path
    test_image(r"C:\Users\SWARAJ\OneDrive\Desktop\ayurspace\dataset\tulsi\Image_1.jpg")       # tulsi ki image ka path
    test_image(r"C:\Users\SWARAJ\OneDrive\Desktop\ayurspace\dataset\ashwagandha\Ashwagandha.jpeg")    # ashwagandha ki image ka path
    test_image(r"C:\Users\SWARAJ\OneDrive\Desktop\ayurspace\dataset\aloe_vera\Image_1.jpg")        # aloe vera ki image ka path
