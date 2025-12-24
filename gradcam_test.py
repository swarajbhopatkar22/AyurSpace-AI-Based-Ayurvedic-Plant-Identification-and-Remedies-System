import numpy as np
import cv2
import tensorflow as tf
from gradcam_utils import get_img_array, make_gradcam_heatmap
from app import model, PLANT_CLASSES

# ResNet50 की last conv layer ka naam:
LAST_CONV_LAYER = "conv5_block3_out"

def run_gradcam(img_path, output_path="gradcam_result.jpg"):
    # 1. image load + preprocess
    img_array = get_img_array(img_path, size=(224, 224))

    # 2. normal prediction
    preds = model.predict(img_array)[0]
    class_idx = int(np.argmax(preds))
    plant_key = PLANT_CLASSES[class_idx]
    confidence = float(np.max(preds) * 100.0)
    print("Predicted:", plant_key, "Confidence:", confidence)

    # 3. heatmap बनाओ
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name=LAST_CONV_LAYER,
        pred_index=class_idx
    )

    # 4. original image pe overlay
    orig = cv2.imread(img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.5, heatmap_color, 0.5, 0)

    # 5. file save karo
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Grad-CAM image saved at:", output_path)

if __name__ == "__main__":
    # yahan apni test image ka path do:
    img_path = r"C:\Users\SWARAJ\OneDrive\Desktop\ayurspace\dataset\neem\Image_1.jpg"
    run_gradcam(img_path, output_path="gradcam_neem.jpg")
