import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, img_path):

    # Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))

    # Convert BGR → RGB for correct display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.expand_dims(img_rgb / 255.0, axis=0)

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, 'mixed7')

    # Process heatmap
    heatmap = cv2.resize(heatmap, (150, 150))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert heatmap to RGB for matplotlib
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose
    superimposed_img = heatmap_rgb * 0.4 + img_rgb
    superimposed_img = np.uint8(superimposed_img)

    # Plot side-by-side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(superimposed_img)
    plt.axis("off")

    plt.tight_layout()
    plt.show()