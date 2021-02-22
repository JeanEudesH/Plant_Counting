import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import statistics
from math import sqrt
import time


"""
Détermination des rangs
Réduction du nombre de pixels (Mettre une condition selon laquelle si le rang
                               est à un angle trop horizontal, prendre les
                               colonnes plutôt que les lignes)
Détermination du rang le plus long et emplacement des rangs
Distance minimale  et direction pour chaque pixel
Direction moyenne
"""


def DBSCAN_clustering(img, epsilon, min_point):

    # extraction of white pixels coordinates
    img_array = np.array(img)
    mat_coord = np.argwhere(img_array == 255)

    # clustering using DBSCAN
    mat_clustered = DBSCAN(eps=epsilon, min_samples=min_point).fit(mat_coord)

    # Panda dataframe
    dataframe_coord = pd.DataFrame(mat_coord)
    dataframe_coord = dataframe_coord.rename(columns={0: "Y", 1: "X"})
    label = pd.DataFrame(mat_clustered.labels_)
    label = label.rename(columns={0: "label"})

    # Dataframe gathering each plants pixel its label
    dataframe_coord = pd.concat([dataframe_coord, label], axis=1)
    return dataframe_coord


def pixel_median(dataframe_coord, img):
    # Get the image and transforms it into a np array
    img_array = np.array(img)
    img_height = img_array.shape[1]
    img_width = img_array.shape[0]

    # For each cluster determined by DBSCAN
    label_row = np.unique(dataframe_coord[["label"]].to_numpy())
    coord_Pixels_img = []

    for row in label_row:
        # np.array of n lines and 2 columns:
            # n being the number of pixels in this row
            # Coordinates of these pixels
        coord_row = dataframe_coord[dataframe_coord["label"] == row][[
            "Y", "X"]].to_numpy()
        # np.array of 2 lines and n columns
        coord_row_tr = coord_row.T

        # Threshold to avoid smaller clusters not significant (noise)
        # Threshold at 0.1% of the size of the picture
        if coord_row.shape[0] > 0.0001*(img_height*img_width):

            coord_Pixels_row = []
            # List of median pixels (one per line of the image)
            # Get the median pixel for a row of pixels in a labelled row

            step = 2 # Also in function of the size of the image
            for i in range(0, img_height, step):
                # Get all pixels in the rows whom Y coordinates is within the
                # designated range
                interval_coord, = np.where((coord_row_tr[0] > i)
                                           & (coord_row_tr[0] <= i+step))

                if len(interval_coord) > 0:
                    # Check if pixels are in the studied area

                    samplePixels = coord_row[interval_coord]
                    # We want the median value on the Y coordinates and its
                    # corresponding X value
                    YMedian = statistics.median(samplePixels.T[0])
                    XMedian = statistics.median(samplePixels.T[1])

                    # Interpolation when number of data is even, so we need to
                    # get the closest value in our dataset

                    coord_Pixels_row.append([int(YMedian), int(XMedian)])
            coord_Pixels_img.append(coord_Pixels_row)

    return coord_Pixels_img


def distance_min_1pixel(pixel_coord, row_compared):
    # pixel_coord is a list of 2 values [X, Y]
    # row_compared is a list of pixels with their coordinates presented like pixel_coord
    Y1, X1 = pixel_coord[0], pixel_coord[1]
    distance_min = 20000
    # Number high enough so that the minimal distance has to be inferior
    direction = [0, 0]
    for pixel in row_compared:
        Y2, X2 = pixel[0], pixel[1]
        distance = sqrt((X2-X1)**2 + (Y2-Y1)**2)
        if distance < distance_min:
            distance_min = distance
            direction = [Y2-Y1, X2-X1]

    return distance_min, direction


def order_size_rows(coordPixelsRow):

    rows_len = [len(coordPixelsRow[row]) for row in range(len(coordPixelsRow))]
    index_rows = [i for i in range(len(coordPixelsRow))]

    size_rows_sorted = [i for _, i in sorted(zip(rows_len, index_rows),
                                             reverse=True)]

    return size_rows_sorted


def dist_direction_row(coord_pixels):

    # List of the index of the rows, sorted by size,
    # the first index belongs to the biggest row
    index_rows = order_size_rows(coord_pixels)
    start_row_pos = coord_pixels[index_rows[0]]
    final_row = coord_pixels[index_rows[1]]

    directions = []
    distances = []

    # Parse through all pixels of the starting row
    for pixel in start_row_pos:
        # Not sure that the distance minimal is really useful
        dist_min, direction = distance_min_1pixel(pixel, final_row)
        directions.append(direction)
        distances.append(dist_min)

    return directions


def direction_mean(directions_all_pixels):

    Y_mean, X_mean = 0, 0
    nb_pixels = len(directions_all_pixels)

    for pixel in directions_all_pixels:
        Y_mean += pixel[0]
        X_mean += pixel[1]

    Y_mean, X_mean = Y_mean/nb_pixels, X_mean/nb_pixels

    return [Y_mean, X_mean]


def direction_med(directions_all_pixels):

    # Transposed array to get all X in the same list and all Y together
    Y_med = statistics.median(np.array(directions_all_pixels).T[0])
    X_med = statistics.median(np.array(directions_all_pixels).T[1])

    return [Y_med, X_med]


def plot_cluster(coordPixels, dataframe_coord, size_img,
                 direction_med, direction_mean):

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)
    ax.scatter(
        dataframe_coord["X"].tolist(),
        dataframe_coord["Y"].tolist(),
        c=dataframe_coord["label"].tolist(),
        s=0.5,
        cmap="Paired",
    )
    for row in range(len(coordPixels)):
        X = []
        Y = []
        for pixel in range(len(coordPixels[row])):
            Y.append(coordPixels[row][pixel][0])
            X.append(coordPixels[row][pixel][1])

        ax.plot(X, Y, '+', c='r')

    Y_vec_dir_mean = [size_img[0]//2, size_img[0]//2 + direction_mean[0],
                      size_img[0]//2 - direction_mean[0]]
    X_vec_dir_mean = [size_img[1]//2, size_img[1]//2 + direction_mean[1],
                      size_img[1]//2 - direction_mean[1]]
    plt.plot(X_vec_dir_mean, Y_vec_dir_mean, c='b')

    Y_vec_dir_med = [size_img[0]//2, size_img[0]//2 + direction_med[0],
                     size_img[0]//2 - direction_med[0]]
    X_vec_dir_med = [size_img[1]//2, size_img[1]//2 + direction_med[1],
                     size_img[1]//2 - direction_med[1]]
    plt.plot(X_vec_dir_med, Y_vec_dir_med, c='g')

    plt.show()
    # fig.savefig("/home/fort/Bureau/results/" + image.split(".")[0] + ".png")
    return


def Total_Plant_Position(path_image_input, epsilon, min_point):

    start_time = time.time()

    list_image = listdir(path_image_input)
    if '.DS_Store' in list_image:
        list_image.remove('.DS_Store')
    for image in list_image:
        start_time_img = time.time()
        print("--------------------------- \n start ", image)
        imgColor = Image.open(path_image_input + "/" + image)
        # Be sure to be in a greyscale images, with only one channel
        img = imgColor.convert(mode='L')

        # Temporaire
        img_array = np.array(img)
        # plt.plot(img_array, '+', c='k')
        # print(img_array)
        unique, counts = np.unique(img_array, return_counts=True)
        # odict = dict(zip(unique, counts))
        # print(odict)
        # plt.plot(odict.keys(), odict.values(), '+')
        # print(np.unique(img_array, return_counts=True))

        print("DBSCAN")
        dataframe_coord = DBSCAN_clustering(img, epsilon, min_point)
        coordPixelsMedian = pixel_median(dataframe_coord, img)
        directions = dist_direction_row(coordPixelsMedian)
        dir_mean = direction_mean(directions)
        dir_med = direction_med(directions)

        print("dirMean", dir_mean, "dir_med", dir_med)
        # pixel_median return a list of lists of size 2.
        # List of the coordonnates of the pixels
        plot_cluster(coordPixelsMedian,
                     dataframe_coord,
                     img_array.shape,
                     dir_med,
                     dir_mean)

        print("--- %s seconds ---" % (time.time() - start_time_img))

    print("--- %s seconds ---" % (time.time() - start_time))
    return


Total_Plant_Position(
    path_image_input="./../../images",
    epsilon=10,
    min_point=10,
)
