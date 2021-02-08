import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def get_json_file_content(_path_json_file):
    f = open(_path_json_file)
    return json.load(f)


def plot_json(path_json, path_otsu_image, path_output):
    X = []
    Y = []
    data = get_json_file_content(path_json)
    for row in range(len(data)):
        for plant in range(len(data[row])):
            X.append(data[row][plant][0])
            Y.append(data[row][plant][1])

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)
    plant = ax.scatter(Y, X, s=80, marker="+", color="r")
    img = Image.open(path_otsu_image)
    # fig.savefig(path_output + "DBSCAN_Fuzzy_cluster.png")
    plt.imshow(img)
    plt.show()


def plot_csv(path):
    X = []
    Y = []
    data = np.genfromtxt(path, delimiter=",")
    for i in range(data.shape[0]):
        X.append(data[i][3])
        Y.append(data[i][2])
    plt.scatter(X, Y, marker="+")
    plt.show()


plot_csv(
    "/home/fort/Documents/APT 3A/Cours/Ekinocs/Output_General/Output/Session_2/Adjusted_Position_Files/Adjusted_plant_positions_0_22_0_88_0_2322.csv"
)

path_json = "/home/fort/Documents/APT 3A/Cours/Ekinocs/Ouput_General/Ouput_General/Output_FA/Session_1/Plant_FT_Predictions/PredictedRows_Img_0_2504.json"
path_otsu_image = "/home/fort/Documents/APT 3A/Cours/Ekinocs/Output_General/Output/Session_2/Otsu/OTSU_0_22_0_88_0_2322.jpg"
path_output = ""
plot_json(path_json, path_otsu_image, path_output)