from skimage.morphology import skeletonize
from skimage import draw, filters
from skimage.io import imread, imshow
from skimage.color import rgb2gray
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from PIL import Image
import pandas as pd
from sklearn.cluster import DBSCAN


# load image from file
img_name = "/home/fort/Documents/APT 3A/Cours/Ekinocs/Output_General/Output/Session_2/Otsu_R/OTSU_R_0_22_0_88_0_2322.jpg"
img = Image.open(img_name)
img_array = np.array(img)
img_array = img_array[:, :, 0]
image_median = filters.median(img_array, np.ones((2, 2)))
img_array = np.where(img_array < 255, 0, img_array)
img_array = np.where(img_array == 255, 1, img_array)
print(img_array)
skeleton = skeletonize(img_array)

plt.figure(figsize=(10, 8))
plt.subplot(111)
plt.imshow(img_array, cmap="gray", interpolation="nearest")
plt.show()


def DBSCAN_clustering(img, epsilon, min_point):

    # extraction of white pixels coordinates
    img_array = np.array(img)
    mat_coord = np.argwhere(img_array[:] == 255)

    # clustering using DBSCAN
    mat_clustered = DBSCAN(eps=epsilon, min_samples=min_point).fit(mat_coord)

    # Panda dataframe
    dataframe_coord = pd.DataFrame(mat_coord)
    dataframe_coord = dataframe_coord.rename(columns={0: "X", 1: "Y"})
    label = pd.DataFrame(mat_clustered.labels_)
    label = label.rename(columns={0: "label"})

    # Dataframe gathering each plants pixel its label
    dataframe_coord = pd.concat([dataframe_coord, label], axis=1)
    return dataframe_coord


def plot_cluster(dataframe_coord):
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)
    ax.scatter(
        dataframe_coord["X"].tolist(),
        dataframe_coord["Y"].tolist(),
        c=dataframe_coord["label"].tolist(),
        s=0.5,
        cmap="Paired",
    )
    plt.show()


dataframe_coord = DBSCAN_clustering(image_median, 20, 5)
plot_cluster(dataframe_coord)