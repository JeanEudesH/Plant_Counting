"""
@PFR_tournesol

Steps:

- Characterization of the rows in the image with DBSCAN clustering

- Reduction of the number of pixels in each rows with a median filter
    (Issue when the rows are horizontal, rotate the image in that case or do
     the reduction on the X axis instead of the Y axis)

- Order the index of the rows according to their size

- Find a pixel with the minimal distance to a given pixel of another row
    (to get a direction vector)

- Get the direction vector of each pixel of a row
    (direction to another row, second longest)

- Calculate the mean and median direction vector for the whole row based
    on all the directions from the pixels

- Fourier analysis by scanning in the direction set by the direction vector:
    Obtention of the inter-row distance

- Reassemble the separated clusters belonging to one rows by running the scan

- With rows going from side to side in the image, find the inter-plant distance
"""

# Imports
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import statistics
from math import sqrt
import time


def DBSCAN_clustering(img, epsilon, min_point):
    """
    Detection of rows of plants in the image with the DBSCAN clustering method.

    Parameters
    ----------
    img : IMAGE object
        Otsu image of the field. Pre-treatment possible with the script
        Process_image_for_FT.py in the Pre_Treatments folder.

    epsilon : FLOAT
        Parameter for the DBSCAN function of the scikit-learn library.
        The maximum distance between two samples for one to be considered as in
        the neighborhood of the other. This is not a maximum bound on the
        distances of points within a cluster. This is the most important DBSCAN
        parameter to choose appropriately for your data set and distance
        function.
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    min_point : INTEGER
        Parameter for the DBSCAN function of the scikit-learn library.
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Returns
    -------
    dataframe_coord : PANDA DATAFRAME
        Panda dataframe containing all the plants pixels (white in the image)
        coordinates and their corresponding cluster or row label.
        One column for the row label, one for the X coordinate, one the Y.

    """

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
    """
    Reduction of the number of pixels plants in each row. Done by "reading" the
    image horizontally.

    Parameters
    ----------
    dataframe_coord : PANDA DATAFRAME
        Panda dataframe with one column for the row label, one for the X
        coordinate, one the Y coordinate of the plants pixels.

    img : IMAGE object
        Otsu image of the field. Pre-treatment possible with the script
        Process_image_for_FT.py in the Pre_Treatments folder.
        Image used in DBSCAN clustering.

    Returns
    -------
    coord_Pixels_img : LIST of LISTS
        List of the rows coordinates. Length of the list is the number of rows.
        In each list (row), the coordinates of the pixels are presented in
        lists of length of 2 [X1, Y1].
        Small example for an imaginary image with 2 rows and less than 5 pixels
        in the rows:
            [[[Y1, X1], [Y2, X2], [Y3, X3], [Y4, X4]],
             [[Y5, X5], [Y6, X6], [Y7, X7]]]

    """

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
        coord_row = dataframe_coord[dataframe_coord["label"] == row][
            ["Y", "X"]
        ].to_numpy()
        # np.array of 2 lines and n columns
        coord_row_tr = coord_row.T

        # Threshold to avoid smaller clusters not significant (noise)
        # Threshold at 0.1% of the size of the picture
        if coord_row.shape[0] > 0.0001 * (img_height * img_width):

            coord_Pixels_row = []
            # List of median pixels (one per line of the image)
            # Get the median pixel for a row of pixels in a labelled row

            step = 2  # Put in function of the size of the image?
            for i in range(0, img_height, step):
                # Get all pixels in the rows whom Y coordinates is within the
                # designated range
                (interval_coord,) = np.where(
                    (coord_row_tr[0] > i) & (coord_row_tr[0] <= i + step)
                )

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
    """
    Calculate the minimal distance from a given pixel to another pixel from
    another row (to be found).

    Parameters
    ----------
    pixel_coord : LIST of length 2
        X and Y coordinates: [Y, X]

    row_compared : LIST of LISTS
        List of the [Y, X] pixels coordinates in which we search the closest
        pixel to the one given in input.

    Returns
    -------
    distance_min : FLOAT
        Distance from the pixel to the closest pixel in the row.

    direction : LIST
        Direction vector between the pixel and the closest pixel in the row.

    """

    Y1, X1 = pixel_coord[0], pixel_coord[1]
    distance_min = 20000
    # Arbitrary number high enough so the minimal distance has to be inferior
    direction = [0, 0]
    for pixel in row_compared:
        Y2, X2 = pixel[0], pixel[1]
        distance = sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)
        if distance < distance_min:
            distance_min = distance
            direction = [Y2 - Y1, X2 - X1]

    return distance_min, direction


def order_size_rows(coordPixelsRow):
    """
    Order the index of the rows given by DBSCAN_clustering according to their
    size, in decreasing order.

    Parameters
    ----------
    coordPixelsRow : LIST
        List of the rows coordinates obtained with the function pixel_median.
        Length of the list is the number of rows and in each list (row), the
        coordinates of the pixels are presented in lists of length of 2.

    Returns
    -------
    size_rows_sorted : LIST
        List of the index of the rows (from the pixel_median function) sorted
        by size of the row in a decreasinf order.

    """

    rows_len = [len(coordPixelsRow[row]) for row in range(len(coordPixelsRow))]
    index_rows = [i for i in range(len(coordPixelsRow))]

    size_rows_sorted = [i for _, i in sorted(zip(rows_len, index_rows), reverse=True)]

    return size_rows_sorted


def dist_direction_row(coord_pixels):
    """
    Get the direction from all pixels in the bigger row in the image to the
    second bigger row of the image.

    Parameters
    ----------
    coord_pixels : LIST
        List of the rows coordinates obtained with the function pixel_median.
        Length of the list is the number of rows and in each list (row), the
        coordinates of the pixels are presented in lists of length of 2.

    Returns
    -------
    directions : LIST of LISTS
        List of all the directions for each pixel of the row. A direction is a
        list of size 2, with the Y and X direction.

    """

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
    """
    Calculate the mean direction based on all direction for each pixel toward
    the second bigger row.

    Parameters
    ----------
    directions_all_pixels : LIST of LISTS
        List of all the directions for each pixel of the row. A direction is a
        list of size 2, with the Y and X direction.

    Returns
    -------
    mean_dir : LIST
        [Y, X] mean direction.

    """

    Y_mean, X_mean = 0, 0
    nb_pixels = len(directions_all_pixels)

    for pixel in directions_all_pixels:
        Y_mean += pixel[0]
        X_mean += pixel[1]

    Y_mean, X_mean = Y_mean / nb_pixels, X_mean / nb_pixels

    mean_dir = [Y_mean, X_mean]

    return mean_dir


def direction_med(directions_all_pixels):

    # Transposed array to get all X in the same list and all Y together
    Y_med = statistics.median(np.array(directions_all_pixels).T[0])
    X_med = statistics.median(np.array(directions_all_pixels).T[1])

    return [Y_med, X_med]


def translate_row(longest_row, direction, img_array):
    # Move the bigger row with a certain step to determine
    # Direction chosen and move with translate row
    # Step is a poroportion of the vector direction

    # BOundaries si les coord sont hors image, lui dire tout va bien et arrêt
    # si plus aucun pixel dans l'image (fonction supplémentaire ? )
    # Move forward
    step = 0.01
    forward = 0
    new_longest_row = []
    size_Y, size_X = get_image_size(
        img_array
    )  # attention a voir si les Y et les X sont inversé

    for coord in longest_row:
        coord_Y = coord[0] - direction[0] * step
        coord_X = coord[1] - direction[1] * step
        forward = forward - sqrt(direction[0] ** 2 + direction[1] ** 2) * (step ** 2)

        if (coord_Y > 0 and coord_Y < size_Y) and (coord_X > 0 and coord_X < size_X):
            new_longest_row.append([int(coord_Y), int(coord_X)])
    return new_longest_row, forward


def get_image_size(img_array):
    Y, X = img_array.shape

    return Y, X


def sum_plant_pixels(coord_row, img_matrix):
    count = 0
    for coord_pixel in coord_row:
        if img_matrix[coord_pixel[0]][coord_pixel[1]] > 0:
            count += 1
    return count


def Fourier_inter_row(X, Y):
    # Fourier calculation
    # X: all steps of the scan (its position?)
    # Y: Number of white pixels
    # Use the "line" from median pixel

    # Use sum_plant_pixels et translate_row
    # +/- 1 pour le sens
    result_fft = 1
    return result_fft


def calculate_dist_interRow(coordPixelsMedian, direction, img_array):
    # on a l'index pour ordonner les rangs du plus grand au plus petit
    size_rows_sorted = order_size_rows(coordPixelsMedian)
    # On prend le rang le plus grand
    longest_row_for = coordPixelsMedian[size_rows_sorted[0]]
    move_step_for = []
    sum_pixel_for = []
    while (
        len(longest_row_for) > 0
    ):  # Tant qu'il ya des points dans l'image on avant le rang
        longest_row_for, forward = translate_row(
            longest_row_for, direction, img_array
        )  # On obtient le nouveau nuage de points
        # On fait la fonction sumavec le nouveau nuage de point et on ajoute forward à la liste des déplacements
        move_step_for.append(forward)

        sum_pixel = sum_plant_pixels(longest_row_for, img_array)
        sum_pixel_for.append(sum_pixel)

    # longest_row_back = coordPixelsMedian[size_rows_sorted[0]]
    # while len(
    #    longest_row_back > 0
    # ):  # Tant qu'il y a des points dans l'image on recule le rang
    #    direction_back = [direction[0] * (-1), direction[1] * (-1)]
    # longest_row_back, backward = translate_row(longest_row, direction_back, img_array)

    return sum_pixel_for, move_step_for


def plot_fft():
    # plot la distance inter rang
    pass


def row_fusion():
    # if the scan recover 2 clusters, we can estimate they belong to the same row
    pass


def iteration_row_fusion():
    # For each iteration take the longer row
    # loop row_fusion
    pass


def calculate_dist_interPlant():
    # Get an area of constant size, thickness constant and a row going from
    # side to side of the image
    # X:
    # Y:
    pass


def plot_cluster(coordPixels, dataframe_coord, size_img, direction_med, direction_mean):

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

        ax.plot(X, Y, "+", c="r")

    Y_vec_dir_mean = [
        size_img[0] // 2,
        size_img[0] // 2 + direction_mean[0],
        size_img[0] // 2 - direction_mean[0],
    ]
    X_vec_dir_mean = [
        size_img[1] // 2,
        size_img[1] // 2 + direction_mean[1],
        size_img[1] // 2 - direction_mean[1],
    ]
    plt.plot(X_vec_dir_mean, Y_vec_dir_mean, c="b")

    Y_vec_dir_med = [
        size_img[0] // 2,
        size_img[0] // 2 + direction_med[0],
        size_img[0] // 2 - direction_med[0],
    ]
    X_vec_dir_med = [
        size_img[1] // 2,
        size_img[1] // 2 + direction_med[1],
        size_img[1] // 2 - direction_med[1],
    ]
    plt.plot(X_vec_dir_med, Y_vec_dir_med, c="g")

    plt.show()
    # fig.savefig("/home/fort/Bureau/results/" + image.split(".")[0] + ".png")
    return


def Total_Plant_Position(path_image_input, epsilon, min_point):

    start_time = time.time()

    list_image = listdir(path_image_input)
    if ".DS_Store" in list_image:
        list_image.remove(".DS_Store")
    for image in list_image:
        start_time_img = time.time()
        print("--------------------------- \n start ", image)
        imgColor = Image.open(path_image_input + "/" + image)
        # Be sure to be in a greyscale images, with only one channel
        img = imgColor.convert(mode="L")

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
        plot_cluster(
            coordPixelsMedian, dataframe_coord, img_array.shape, dir_med, dir_mean
        )
        # calcul of the inter-row distance
        sum_pixel_for, move_step_for = calculate_dist_interRow(
            coordPixelsMedian, dir_med, img_array
        )
        plt.plot(move_step_for, sum_pixel_for)
        plt.show()
        print("--- %s seconds ---" % (time.time() - start_time_img))
        break
    print("--- %s seconds ---" % (time.time() - start_time))
    return


Total_Plant_Position(
    path_image_input="/home/fort/Bureau/test_image",
    epsilon=20,
    min_point=20,
)
