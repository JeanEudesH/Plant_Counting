# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:55:31 2020

@author: eliot
"""

import os
import json
import numpy as np
from PIL import Image
import sys

if not "D:/Documents/IODAA/Fil Rouge/Plant_Counting" in sys.path:
    sys.path.append("D:/Documents/IODAA/Fil Rouge/Plant_Counting")

import MAS_v17 as MAS

# =============================================================================
# Utility Functions Definition
# =============================================================================
def import_data(_path, _file_names, _import_function):
    data = []
    for _n in _file_names:
        data += [_import_function(_path + "/" + _n)]
    return data

def get_json_file_content(_path_json_file):
    f = open(_path_json_file)
    return json.load(f)

def get_img_array(path_img):
    img = Image.open(path_img)
    return np.array(img)

def get_file_lines(path_csv_file):
    file_object = open(path_csv_file, 'r')
    file_content = file_object.readlines()
    file_object.close()
    return(file_content)
    
# =============================================================================
# General Path Definition
# =============================================================================
#path of the images 
### TO BE CHANGED AS PER USER NEED
path_input_root = "D:/Documents/IODAA/Fil Rouge/Resultats"
unity_date = "dIP_vs_dIR_curved_more_ratio"
densite = 7
dIP_dIR = "02_07_disjoint"
session_number = 1

recon_policy = "local_XY"

path_input_raw = f"{path_input_root}/{unity_date}/densite={densite}/{dIP_dIR}/virtual_reality"
path_input_adjusted_position_files = f"{path_input_root}/{unity_date}/densite={densite}/{dIP_dIR}_analysis/Output/Session_1/Adjusted_Position_Files_0"
path_input_OTSU = f"{path_input_root}/{unity_date}/densite={densite}/{dIP_dIR}_analysis/Output/Session_1/Otsu"
# path_input_PLANT_FT_PRED = f"{path_input_root}/{unity_date}/densite={densite}/{dIP_dIR}_analysis/Output_FA/Session_1/Plant_FT_Predictions"
path_input_PLANT_FT_PRED = f"{path_input_root}/{unity_date}/densite={densite}/{dIP_dIR}_analysis/Resultats_Clustering/Position_Files_0"

# path_input_raw = f"{path_input_root}/{unity_date}/virtual_reality"
# path_input_adjusted_position_files = f"{path_input_root}/{unity_date}_analysis/Output/Session_1/Adjusted_Position_Files"
# path_input_OTSU = f"{path_input_root}/{unity_date}_analysis/Output/Session_1/Otsu_R"
# path_input_PLANT_FT_PRED = f"{path_input_root}/{unity_date}_analysis/Output_FA/Session_1/Plant_FT_Predictions"

names_input_raw = os.listdir(path_input_raw)
names_input_adjusted_position_files = os.listdir(path_input_adjusted_position_files)
names_input_OTSU = os.listdir(path_input_OTSU)
names_input_PLANT_FT_PRED = os.listdir(path_input_PLANT_FT_PRED)

# =============================================================================
# Data Collection
# =============================================================================
print("Data Collection...", end = " ")

subset_size = 4

data_input_raw = import_data(path_input_raw,
                             names_input_raw[:subset_size],
                             get_img_array)
data_adjusted_position_files = import_data(path_input_adjusted_position_files,
                                           names_input_adjusted_position_files[:subset_size],
                                           get_file_lines)
data_input_OTSU = import_data(path_input_OTSU,
                              names_input_OTSU[:subset_size],
                              get_img_array)
data_input_PLANT_FT_PRED = import_data(path_input_PLANT_FT_PRED,
                                       names_input_PLANT_FT_PRED[:subset_size],
                                       get_json_file_content)

print("Done")

# =============================================================================
# Simulation Parameters Definition
# =============================================================================
RAs_group_size = 25
RAs_group_steps = 2
Simulation_steps = 0

## TODO1: Fuse and fill doesn't work in curved mode...
## TODO2AL are not sorted along the row... Sort them to make the repositionning work : Done but maybe not the cleanest solution
# -> Don't work on indices but let eacsh RAL to keep track of its neighbours ?
## TODO3: The sorting issue impacts the computation of the interplant distance... : OK now either you can sort the agents (not recommanded)
# or you can maintain their neighbours as attributes
## TODO4 : some rows where destroyed at first step : OK added an extra parameter _check_rows_proximity
## TODO5: Recoding Fill and fuse functions with our new setting : to be tested
## TODO6: What to do with the argument InterPlant_Diffs ? It seems like I can remove it completely
RALs_fuse_factor = 0.5
RALs_fill_factor = 1.5

_image_index = 0

print(names_input_OTSU[_image_index])
print(names_input_adjusted_position_files[_image_index])
print(names_input_PLANT_FT_PRED[_image_index])

# =============================================================================
# Simulation Definition
# =============================================================================
print("Simulation Definition:")
MAS_Simulation = MAS.Simulation_MAS(data_input_raw[_image_index],
                                    data_input_PLANT_FT_PRED[_image_index],
                                    data_input_OTSU[_image_index],
                                    RAs_group_size, RAs_group_steps,
                                    RALs_fuse_factor, RALs_fill_factor,
                                    [0,0],
                                    data_adjusted_position_files[_image_index],
                                    recon_policy=recon_policy)
MAS_Simulation.Initialize_AD()
MAS_Simulation.Perform_Simulation_newEndCrit(Simulation_steps,
                                             _coerced_X=True, # coerced X : permet le repositionnement
                                             _coerced_Y=False,
                                             _analyse_and_remove_Rows=False,
                                             _edge_exploration = False,
                                             _check_rows_proximity=False)
                                             # _edge_exploration : dit d'aller explorer vers les bords de l'image
                                             # au cas ou l'analyse de Fourier manque l'intialisation sur les bords

# =============================================================================
# Simulation Analysis
# =============================================================================
# =============================================================================
print("Computing Scores...", end = " ")
MAS_Simulation.Get_RALs_infos()
MAS_Simulation.Compute_Scores()
print("Done")

print(MAS_Simulation.simu_steps_times)
print("NB Rals =", MAS_Simulation.RALs_recorded_count[-1])
print("TP =", MAS_Simulation.TP)
print("FN =", MAS_Simulation.FN)
print("FP =", MAS_Simulation.FP)
# =============================================================================

MAS_Simulation.Show_Adjusted_And_RALs_positions(_save=False, _save_path=f"{path_input_root}/{unity_date}_analysis/Images_MAS/{recon_policy}_repositioning")
# MAS_Simulation.Show_nb_RALs()
# MAS_Simulation.Show_RALs_Deicision_Scores()
MAS_Simulation.Show_Adjusted_Positions()

import matplotlib.pyplot as plt
plt.show()
