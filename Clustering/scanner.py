import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import curve_fit


def Total_Plant_Position(path_image_input, epsilon, min_point):
    list_image = listdir(path_image_input)
    if '.DS_Store' in list_image:
        list_image.remove('.DS_Store')
    for image in list_image:
        print("start ", image)
        imgColor = Image.open(path_image_input + "/" + image)
        # Be sure to be in a greyscale images, with only one channel
        img = imgColor.convert(mode='L')

        # Temporaire
        img_array = np.array(img)
        # plt.plot(img_array, '+', c='k')
        # print(img_array)
        unique, counts = np.unique(img_array, return_counts=True)
        odict = dict(zip(unique, counts))
        # print(odict)
        # plt.plot(odict.keys(), odict.values(), '+')
        # print(np.unique(img_array, return_counts=True))

        print("DBSCAN")
        dataframe_coord = DBSCAN_clustering(img, epsilon, min_point)
        coordPixelsMedian = pixel_median(dataframe_coord, img)
        # fitting = fit_curve(coordPixelsMedian)
        plot_cluster(coordPixelsMedian, dataframe_coord, image, 1)

    return


def pixel_median(dataframe_coord, img):
    # Get the image and transforms it into a np array
    img_array = np.array(img)

    # For each cluster determined by DBSCAN
    label_row = np.unique(dataframe_coord[["label"]].to_numpy())
    coord_Pixels_img = []

    print('nombre de rangs', len(label_row))
    print('les rangs sont ', label_row)
    for row in label_row:
        coord_row = dataframe_coord[dataframe_coord["label"] == row][[
            "X", "Y"]].to_numpy()
        # Threshold to avoid smaller clusters creating noise
        if coord_row.shape[0] > 1000:
            print(coord_row.shape)
            coord_Pixels_row = []
            # List of median pixels (one per line of the image)

            # Get the median pixel for a row of pixels in a labelled row
            # img_height = img_array.shape[0]
            img_height = img_array.shape[0]
            img_width = img_array[1]

            # Ici le pas est à 5 mais peut-être choisir un pas plus
            # proportionnel à la taille de l'image (à faire empiriquement)

            step = 5
            for i in range(0,img_height,step):
                interval_coord = np.where((coord_row > i) & (coord_row <= i+step))
                if len(interval_coord) > 0:
                    # if len(np.where(np.logical_and(coord_row > i,
                    # coord_row < i+5))) > 0:
                    pos = interval_coord[0]
                    if len(coord_row[pos].T[0]) > 0:

                        pixelMedian = statistics.median(coord_row[pos].T[0])
                        #print(pixelMedian)
                        coord_Pixels_row.append([int(pixelMedian),
                                                 coord_row[pos].T[1][0]])
            coord_Pixels_img.append(coord_Pixels_row)

    return coord_Pixels_img


def pixel_median(dataframe_coord, img):
    # Get the image and transforms it into a np array
    img_array = np.array(img)
    img_height = img_array.shape[1]
    img_width = img_array.shape[0]

    # For each cluster determined by DBSCAN
    label_row = np.unique(dataframe_coord[["label"]].to_numpy())
    coord_Pixels_img = []

    for row in label_row:
        coord_row = dataframe_coord[dataframe_coord["label"] == row][[
            "X", "Y"]].to_numpy()
        # Threshold to avoid smaller clusters creating noise
        if coord_row.shape[0] > 10:
            coord_Pixels_row = []
            # List of median pixels (one per line of the image)
            # Get the median pixel for a row of pixels in a labelled row
            # Ici le pas est à 5 mais peut-être choisir un pas plus
            # proportionnel à la taille de l'image (à faire empiriquement)

            step = 3
            for i in range(0,img_height,step):
                # Get all pixels in the rows whom Y coordinates is within the
                # designated range
                coord_row_tr = coord_row.T
                interval_coord, = np.where((coord_row_tr[1] > i)
                                           & (coord_row_tr[1] <= i+step))

                if len(interval_coord) > 0:
                    samplePixels = coord_row[interval_coord]
                    # We want the median value on the Y coordinates and its
                    # corresponding X value
                    YMedian = statistics.median(samplePixels.T[1])
                    XMedian = statistics.median(samplePixels.T[0])

                    # Interpolation when number of data is even, so we need to get the closest value in our dataset
                    #print(pixelMedian)
                    coord_Pixels_row.append([int(XMedian), int(YMedian)])
            coord_Pixels_img.append(coord_Pixels_row)

    return coord_Pixels_img


def DBSCAN_clustering(img, epsilon, min_point):

    # extraction of white pixels coordinates
    img_array = np.array(img)
    mat_coord = np.argwhere(img_array == 255)

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


def func(x, a, b, c):
    y = a * (x ** 2) + b * x + c
    return y


def fit_curve(coordPixels):

    for row in range(len(coordPixels)):
        X = []
        Y = []
        row_label = []
        for pixel in range(len(coordPixels[row])):
            row_label.append(pixel)
            X.append(coordPixels[row][pixel][0])
            Y.append(coordPixels[row][pixel][1])
        Xarray = np.array(X)
        Yarray = np.array(Y)
        row_array = np.array(row_label)

        fitting, _ = curve_fit(func, Xarray, Yarray)

        #fit_label.setdefault(row, fitting)

    return fitting


def plot_cluster(coordPixels, dataframe_coord, image, fitting):

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)
    ax.scatter(
        dataframe_coord["X"].tolist(),
        dataframe_coord["Y"].tolist(),
        c=dataframe_coord["label"].tolist(),
        s=0.5,
        cmap="Paired",
    )
    # for row in range(len(coordPixels)):
    #     X = []
    #     Y = []
    #     for pixel in range(len(coordPixels[row])):
    #         X.append(coordPixels[row][pixel][0])
    #         Y.append(coordPixels[row][pixel][1])

    #     ax.plot( X, Y, '+',c='r')
    # fitting_int = [int(i) for i in fitting]
    # print(fitting_int)
    # ax.plot(X, func(np.array(X), *fitting_int), 'b-')


        # ax.plot(
        #     [x for x in range(mini, maxi, 1)],
        #     [
        #         func(
        #             x,
        #             fit_label.get(row)[0],
        #             fit_label.get(row)[1],
        #             fit_label.get(row)[2],
        #         )
        #         for x in range(mini, maxi, 1)
        #     ],
        # )

    plt.show()
    #fig.savefig("/home/fort/Bureau/results/" + image.split(".")[0] + ".png")
    return


# def find_equation(dataframe_coord):
#     label_row = np.unique(dataframe_coord[["label"]].to_numpy())
#     fit_label = {}
#     for row in label_row:
#         print("ROW ", row, type(row))
#         dataframe_coord_row = dataframe_coord[dataframe_coord["label"] == row][
#             ["X", "Y"]
#         ]
#         print("coordonnees rang", dataframe_coord_row)
#         fitting, _ = curve_fit(func, dataframe_coord_row["X"], dataframe_coord_row["Y"])
#         fit_label.setdefault(row, fitting)
#     return fit_label


# def plot_cluster(dataframe_coord, image, fit_label):
#     fig = plt.figure(figsize=(8, 10))
#     ax = fig.add_subplot(111)
#     ax.scatter(
#         dataframe_coord["X"].tolist(),
#         dataframe_coord["Y"].tolist(),
#         c=dataframe_coord["label"].tolist(),
#         s=0.5,
#         cmap="Paired",
#     )
#     label_row = np.unique(dataframe_coord[["label"]].to_numpy())
#     for row in label_row:
#         maxi = dataframe_coord[dataframe_coord["label"] == row]["X"].max()
#         mini = dataframe_coord[dataframe_coord["label"] == row]["X"].min()

#         ax.plot(
#             [x for x in range(mini, maxi, 1)],
#             [
#                 func(
#                     x,
#                     fit_label.get(row)[0],
#                     fit_label.get(row)[1],
#                     fit_label.get(row)[2],
#                 )
#                 for x in range(mini, maxi, 1)
#             ],
#         )
#     plt.show()
#     #fig.savefig("/home/fort/Bureau/results/" + image.split(".")[0] + ".png")
#     return


Total_Plant_Position(
    path_image_input="./../../images",
    epsilon=10,
    min_point=10,
)