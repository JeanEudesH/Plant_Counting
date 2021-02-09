# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:55:35 2020

@author: eliot
"""
import os
import sys

from numpy.lib.function_base import _CORE_DIMENSION_LIST

# os.chdir("../Pre_Treatments")
# import Process_image_for_FT as PiFT

# os.chdir("../Fourier")
# import FrequencyAnalysis as FA

# os.chdir("../MAS")
# import Multi_Images_Simulation_v12bis as MIS

if not "D:/Documents/IODAA/Fil Rouge/Plant_Counting" in sys.path:
    sys.path.append("D:/Documents/IODAA/Fil Rouge/Plant_Counting")

import Pre_Treatments.Process_image_for_FT as PiFT
import Fourier.FrequencyAnalysis as FA
import MAS.Multi_Images_Simulation_v12bis as MIS

def CompleteProcess(_path_input_rgb_img, _path_output_root,
                    
                    _labelled_images = False, _path_position_files=None,
                    _rows_real_angle=0,
                    
                    _make_unique_folder_per_session=False, _session=1,
                    _do_Otsu=True, _do_AD=True,
                    _save_AD_score_images=False, _save_BSAS_images=False,
                    _bsas_threshold=1,
                    
                    _bin_div_X=2, _bin_div_Y=4,
                    
                    _RAs_group_size=20, _RAs_group_steps=2, _Simulation_steps=50,
                    _RALs_fuse_factor=0.5, _RALs_fill_factor=1.5, densite=None, experiment=None):
    
    # PiFT.All_Pre_Treatment(_path_input_rgb_img,
    #                   _path_output_root,
    #                   _path_position_files,
    #                   _rows_real_angle,
    #                   _make_unique_folder_per_session, _session,
    #                   _do_Otsu, _do_AD,
    #                   _save_AD_score_images, _save_BSAS_images,
    #                   _bsas_threshold)
    
    # FA.All_Fourier_Analysis(_path_output_root,
    #                      _session,
    #                      _bin_div_X, _bin_div_Y)
    
    MIS.All_Simulations(_path_input_rgb_img,
                    _path_output_root,
                    _labelled_images,
                    _session,
                    _RAs_group_size, _RAs_group_steps, _Simulation_steps,
                    _RALs_fuse_factor, _RALs_fill_factor,
                    densite=densite, experiment=experiment)

if (__name__=="__main__"):
    # root = "D:/Documents/IODAA/Fil Rouge/Resultats/dIP_vs_dIR_linear"
    # os.chdir(root)
    
    # densites = ["densite=5", "densite=7", "densite=9"]

    # for densite in densites:
    #     os.chdir(os.path.join(root, densite))
    #     experiments = os.listdir()

    #     for experiment in experiments:
    #         path_input_rgb_img = os.path.join(os.getcwd(), experiment, "virtual_reality")
    #         path_output_root = os.path.join(os.getcwd(), f"{experiment}_analysis")
    #         path_position_files=os.path.join(os.getcwd(), experiment, "Position_Files")

    #         print(f"Working on experiment {densite} - {experiment}")

    #         try:
    #             CompleteProcess(_path_input_rgb_img=path_input_rgb_img,
    #                             _path_output_root=path_output_root,
                                
    #                             # Trois lignes sur ma branche (pas a jour) et pas presentes sur la ligne de Baptiste
    #                             _labelled_images = True,
    #                             _path_position_files=path_position_files,
    #                             _rows_real_angle=90,

    #                             _make_unique_folder_per_session=False, _session=1,
    #                             _do_Otsu=True, _do_AD=True,
    #                             _save_AD_score_images=False, _save_BSAS_images=False,
    #                             _bsas_threshold=1,
                                
    #                             _bin_div_X=2, _bin_div_Y=4,
                                
    #                             _RAs_group_size=25, _RAs_group_steps=2, _Simulation_steps=50,
    #                             # nombre de PXA sous les ordres d'agent plante
    #                             # RAs_group_size * 2 = cote du carre => 
    #                             # passer a 25 augmente la taille du PA.

    #                             # see MAS.MetaSimulation
    #                             _RALs_fuse_factor=0.7, _RALs_fill_factor=1.5,
    #                             densite=densite, experiment=experiment)                    
    #         except:
    #             with open(os.path.join(path_output_root, "debug.log"), "w") as log_file:
    #                 log_file.write("an exception occured")
    #             with open("D:\\Documents\\IODAA\\Fil Rouge\\Resultats\\dIP_vs_dIR_linear\\results.log", "a") as log_file:
    #                 log_file.write(f"# {densite} - {experiment}")
    #                 log_file.write("\n")
    #                 log_file.write("AN ERROR OCCURED")

    CompleteProcess(_path_input_rgb_img="D:/Documents/IODAA/Fil Rouge/Resultats/dIP_vs_dIR_curved/densite=7/027_054/virtual_reality",
                    _path_output_root="D:/Documents/IODAA/Fil Rouge/Resultats/dIP_vs_dIR_curved/densite=7/027_054_analysis/",
    # CompleteProcess(_path_input_rgb_img="D:/Documents/IODAA/Fil Rouge/Resultats/dIP_vs_dIR_linear_fixed_density/densite=5/0.22_0.88/virtual_reality",
    #                 _path_output_root="D:/Documents/IODAA/Fil Rouge/Resultats/dIP_vs_dIR_linear_fixed_density_analysis/densite=5/0.22_0.88/",
                    
    # Trois lignes sur ma branche (pas a jour) et pas presentes sur la ligne de Baptiste
                     _labelled_images = True,
                     _path_position_files="D:/Documents/IODAA/Fil Rouge/Resultats/dIP_vs_dIR_curved/densite=7/027_054/Position_Files",
                     _rows_real_angle=90,

                    _make_unique_folder_per_session=False, _session=1,
                    _do_Otsu=True, _do_AD=True,
                    _save_AD_score_images=False, _save_BSAS_images=False,
                    _bsas_threshold=1,
                    
                    _bin_div_X=2, _bin_div_Y=4,
                    
                    _RAs_group_size=20, _RAs_group_steps=2, _Simulation_steps=50,
                    # nombre de PXA sous les ordres d'agent plante
                    # RAs_group_size * 2 = cote du carre => 
                    # passer a 25 augmente la taille du PA.

                    # see MAS.MetaSimulation
                    _RALs_fuse_factor=0.7, _RALs_fill_factor=1.5)
    