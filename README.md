# k-means-clustring
# 🎨 K-Means for Image Compression

This project demonstrates how **K-Means Clustering** can be used to compress images by reducing the number of unique colors.

---

## 📌 Concept Overview

### 🔁 K-Means Iterations

Each iteration includes:

1. **Assignment Step** – Assign each pixel to the closest centroid based on RGB similarity.
2. **Update Step** – Update each centroid to the mean of all assigned pixels.

Repeat for `num_iterations`. After convergence:
- Replace each pixel with its centroid’s color.
- Result: Image with `K` colors instead of thousands → smaller size, faster load.

---

## 🧠 Why Use K-Means for Image Compression?

✅ Reduces file size  
✅ Maintains perceptual image quality  
✅ Great for illustrations, icons, simple graphics

---

## ⚠️ Limitations

- ❌ Not ideal for high-gradient or photographic images  
- ❌ Color banding and artifacts may occur  
- ❌ K-means assumes spherical clusters, which is an oversimplification of real color distributions

---

## 🎯 Choosing the Optimal Number of Clusters (K)

| Method            | Description                                       |
|------------------|---------------------------------------------------|
| Elbow Method      | Plot distortion vs. K, look for an "elbow" point |
| Silhouette Score  | Measures cohesion/separation of clusters         |
| MSE / PSNR        | Quantitative similarity metrics                  |
| SSIM              | Structural image quality score                   |
| Manual Inspection | Evaluate compression visually                    |

---

## ⚖️ Lossy vs. Lossless Compression

| Type         | Can Reconstruct Perfectly? | Example   | K-Means? |
|--------------|-----------------------------|-----------|----------|
| Lossless     | ✅ Yes                       | PNG       | ❌ No     |
| **Lossy**    | ❌ No                        | JPEG      | ✅ Yes    |

> K-Means is **lossy** because it replaces each pixel’s original color with the nearest centroid color, reducing precision for compression.

---

## 🐢 Performance Bottlenecks & Solutions

| Challenge               | Solution                                                                 |
|------------------------|--------------------------------------------------------------------------|
| High computation        | Use `MiniBatchKMeans` or `Elkan` algorithm from `scikit-learn`           |
| Slow convergence        | Limit `num_iterations`, use good centroid initialization                 |
| Memory usage            | Downsample the image before clustering, then upscale colors              |
| Poor initialization     | Use `k-means++` init (default in sklearn)                                |
| Processing time         | Use `NumPy`, `Numba`, or GPU acceleration (`CuPy`, `PyTorch`)            |

---

## 💻 Example: Python Code Using scikit-learn

```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt
from PIL import Image

# Load image and reshape
image = Image.open("input.jpg")
image_np = np.array(image)
w, h, d = image_np.shape
pixels = image_np.reshape((-1, 3))

# Apply K-Means
k = 16  # Number of colors
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)
compressed_pixels = kmeans.cluster_centers_[kmeans.predict(pixels)]

# Reshape back to image
compressed_image = compressed_pixels.reshape((w, h, 3)).astype(np.uint8)

# Save & display
Image.fromarray(compressed_image).save("compressed.jpg")
plt.imshow(compressed_image)
plt.axis('off')
plt.show()
