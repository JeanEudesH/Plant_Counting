# -*- coding: utf-8 -*-
""" Petit morceau de code pour calculer quelques métrics:
le nombre de rang dans une image et le nombre de plantes
par images"""

from os import listdir
import json


def folder_parsing(_path_folder_input):
    """
    Pour tous les fichiers qui sont dans un dossier,
    récupérer les chemins de fichiers JSON et les renvoyer sous
    la forme d'une liste
    """
    list_file = listdir(_path_folder_input)
    list_path_json = []
    for f in list_file:
        if f.split(".")[-1] == "json":
            list_path_json.append(_path_folder_input + f)
    return list_path_json


def get_json_file_content(path_json_file):
    f = open(path_json_file)
    return json.load(f)


def count_row_and_plants(path_json_file):
    """
    Dans un JSON, compte le nombre de rangs et le nombre de plantes
    """
    data = get_json_file_content(path_json_file)
    plants_number = 0
    row_number = len(data)
    for row in range(len(data)):
        plants_number += len(data[row])
    return row_number, plants_number


def theoretical_plants_number(path_json_file):
    """
    Donne le nombre theorique de plantes dans une image a
    partir de son nom.
    """
    theoretical_number = path_json_file.split("/")[-1].split(".")[0].split("_")[-1]
    return int(theoretical_number)


def main(_path_folder_input):
    """
    Pour tous les fichier JSON, renvoie le nombre de rangs, et le
    nombre de plantes ainsi qui son pourcentage d'identité avec le
    nombre théorique
    """
    results = open(_path_folder_input + "metrics.txt", "w")
    list_path_json = folder_parsing(_path_folder_input)
    for path_json_file in list_path_json:
        row_number, plants_number = count_row_and_plants(path_json_file)
        theoretical_number = theoretical_plants_number(path_json_file)
        plants_ratio = plants_number / theoretical_number
        results.write(
            "theoretical_number = "
            + str(theoretical_number)
            + "\t plants_number = "
            + str(plants_number)
            + "\t plant ratio = "
            + str(plants_ratio)
            + "\t row number = "
            + str(row_number)
            + "\n"
        )
    return (
        theoretical_number,
        plants_number,
        plants_ratio,
        row_number,
    )


print(
    main(
        "/home/fort/Bureau/results/Resultat_clustering_plants/Resultats_clustering_session_1/"
    )
)
