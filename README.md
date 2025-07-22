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
## example with images 
<img width="1022" height="523" alt="image" src="https://github.com/user-attachments/assets/7d57540c-0991-43f1-98df-223721701414" />
<img width="1031" height="523" alt="image" src="https://github.com/user-attachments/assets/99083baa-49d1-42d9-ba7e-21536a3ed103" />
<img width="1030" height="520" alt="image" src="https://github.com/user-attachments/assets/eb5cdf0c-5b1c-4815-879e-6281700d4b83" />


## 💻 Example: Python Code Using scikit-learn

```python
import sys
import numpy as np
from skimage import io, img_as_float
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

def kmeans_clustering(image_vectors, k, num_iterations):
    lbs = np.full((image_vectors.shape[0],), -1)
    clstr_proto = np.random.rand(k, 3)
    for i in range(num_iterations):
        print('Iteration: ' + str(i + 1))
        point_label = [None for k_i in range(k)]
        for rgb_i, rgb in enumerate(image_vectors):
            rgb_row = np.repeat(rgb, k).reshape(3, k).T
            closest_label = np.argmin(np.linalg.norm(rgb_row - clstr_proto, axis=1))
            lbs[rgb_i] = closest_label
            if point_label[closest_label] is None:
                point_label[closest_label] = []
            point_label[closest_label].append(rgb)
        for k_i in range(k):
            if point_label[k_i] is not None:
                new_cluster_prototype = np.asarray(point_label[k_i]).sum(axis=0) / len(point_label[k_i])
                clstr_proto[k_i] = new_cluster_prototype

    return lbs, clstr_proto

def closest_centroids(X, c):
    K = np.size(c, 0)
    idx = np.zeros((np.size(X, 0), 1))
    arr = np.empty((np.size(X, 0), 1))
    for i in range(0, K):
        y = c[i]
        temp = np.ones((np.size(X, 0), 1)) * y
        b = np.power(np.subtract(X, temp), 2)
        a = np.sum(b, axis=1)
        a = np.asarray(a)
        a.resize((np.size(X, 0), 1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr, 0, axis=1)
    idx = np.argmin(arr, axis=1)
    return idx

def compute_centroids(X, idx, K):
    n = np.size(X, 1)
    centroids = np.zeros((K, n))
    for i in range(0, K):
        ci = idx
        ci = ci.astype(int)
        total_number = sum(ci);
        ci.resize((np.size(X, 0), 1))
        total_matrix = np.matlib.repmat(ci, 1, n)
        ci = np.transpose(ci)
        total = np.multiply(X, total_matrix)
        centroids[i] = (1 / total_number) * np.sum(total, axis=0)
    return centroids

def plot_image_colors_by_color(name, image_vectors):
    fig = plt.figure()
    ax = Axes3D(fig)
    for rgb in image_vectors:
        ax.scatter(rgb[0], rgb[1], rgb[2], c=rgb, marker='o')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    fig.savefig(name + '.png')

def plot_image_colors_by_label(name, image_vectors, lbs, clstr_proto):
    fig = plt.figure()
    ax = Axes3D(fig)

    for rgb_i, rgb in enumerate(image_vectors):
        ax.scatter(rgb[0], rgb[1], rgb[2], c=clstr_proto[lbs[rgb_i]], marker='o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    fig.savefig(name + '.png')

if __name__ == '__main__':

    J = sys.argv[1]
    K = int(sys.argv[2])
    itr = int(sys.argv[3])

    image = io.imread(J)[:, :, :3]
    image = img_as_float(image)
    image_dimensions = image.shape
    image_name = image
    image_vectors = image.reshape(-1, image.shape[-1])
    lbs, color_centroids = kmeans_clustering(image_vectors, k=K, num_iterations=itr)
    output_image = np.zeros(image_vectors.shape)

    for i in range(output_image.shape[0]):
        output_image[i] = color_centroids[lbs[i]]
    output_image = output_image.reshape(image_dimensions)

    print('Saving the Compressed Image')
    io.imsave('Compressed.jpg', output_image)
    print('Image Compression Completed')
    info = os.stat(J)
    print("Image size before : ", info.st_size / 1024, "KB")
    info = os.stat('Compressed.jpg')
    print("Image size : ", info.st_size / 1024, "KB")
