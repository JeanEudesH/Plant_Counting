import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def DBSCAN_clustering(img, epsilon, min_point):

    # extraction of white pixels coordinates
    img_array = np.array(img)
    mat_coord = np.argwhere(img_array[:, :, 0] == 255)

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


def Total_Plant_Position(path_image_input, epsilon, min_point):
    list_image = listdir(path_image_input)
    for image in list_image:
        print("start ", image)
        img = Image.open(path_image_input + "/" + image)
        print("DBSCAN")
        dataframe_coord = DBSCAN_clustering(img, epsilon, min_point)
        fitting = find_equation(dataframe_coord)
        plot_cluster(dataframe_coord, image, fitting)
        break
    return


def func(x, a, b, c):
    y = a * (x ** 2) + b * x + c
    return y


def find_equation(dataframe_coord):
    label_row = np.unique(dataframe_coord[["label"]].to_numpy())
    fit_label = {}
    for row in label_row:
        dataframe_coord_row = dataframe_coord[dataframe_coord["label"] == row][
            ["X", "Y"]
        ]
        fitting, _ = curve_fit(func, dataframe_coord_row["X"], dataframe_coord_row["Y"])
        fit_label.setdefault(row, fitting)
    return fit_label


def plot_cluster(dataframe_coord, image, fit_label):
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)
    ax.scatter(
        dataframe_coord["X"].tolist(),
        dataframe_coord["Y"].tolist(),
        c=dataframe_coord["label"].tolist(),
        s=0.5,
        cmap="Paired",
    )
    label_row = np.unique(dataframe_coord[["label"]].to_numpy())
    for row in label_row:
        maxi = dataframe_coord[dataframe_coord["label"] == row]["X"].max()
        mini = dataframe_coord[dataframe_coord["label"] == row]["X"].min()

        ax.plot(
            [x for x in range(mini, maxi, 1)],
            [
                func(
                    x,
                    fit_label.get(row)[0],
                    fit_label.get(row)[1],
                    fit_label.get(row)[2],
                )
                for x in range(mini, maxi, 1)
            ],
        )
    plt.show()
    fig.savefig("/home/fort/Bureau/results/" + image.split(".")[0] + ".png")
    return


Total_Plant_Position(
    path_image_input="/home/fort/Documents/APT 3A/Cours/Ekinocs/Output_General/Output/Session_2/Otsu_R",
    epsilon=20,
    min_point=10,
)