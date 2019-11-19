#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:48:11 2019

@author: zipat
"""

from scipy.io import loadmat
from pathlib import Path
import numpy as np
from glob import glob
from os.path import isdir, isfile
from itertools import chain, combinations
import pandas as pd

def recursively_extract_element_from_array(array_to_search):
    t = np.ndarray
    while t == np.ndarray:
        if not(array_to_search.size):
            return None
        el = array_to_search[0]
        t = type(el)
        array_to_search = el
    return(el)

def to_extract(folder):
    all_dir = glob("{}/*".format(folder))
    all_dir = [el for el in [el for el in all_dir if isdir(el)] if len(glob("{}/resultsROI*".format(el))) != 0]
    return all_dir

def extract_subjects_name_from_conn_structure(conn_file):
    conn_file_mat = loadmat(conn_file)
    root_conn_functionals = conn_file_mat["CONN_x"]["Setup"][0][0][0][0][6][0]
    subject_name = []
    for el in range(0,len(root_conn_functionals)):
        sub_el = recursively_extract_element_from_array(root_conn_functionals[el])
        subject_name.append(Path(sub_el).stem)
    return(subject_name)

def create_roi_to_roi_line(subject_name, results_mat):
    res_mat = loadmat(results_mat)
    rois = list(chain.from_iterable(res_mat["names"][0]))
    col_names = ["subject"] + [ "_".join(el) for el in list(combinations(rois, 2))] 
    cor_mat = res_mat["Z"][:,0:len(rois)]
    cor_line = cor_mat[np.triu_indices_from(cor_mat)]
    cor_line_clean = cor_line[~np.isnan(cor_line)]
    df_line = [subject_name] + list(cor_line_clean)
    df = pd.DataFrame([df_line], columns = col_names)
    return(df)
    
        
    
def extract_roi_to_roi_values_from_conn(conn_file, output_dir = None):
    """Extract roi to roi values from an already estimated conn first level.
                   
        Parameters
        ----------
        conn_file : a string poiting to an existing file that is a conn model with the first level estimated
        output_dir : the directory were the extracted data are to be written in csv files, if not specified
                    the csv file are written in the main directory of the conn design
        
        
        The function returns as much csv files as roi to roi estimated designs in the conn first_level directory with
        the connectivity values of the source ROIs only
        
    """
    if not isfile(conn_file):
        raise ValueError("conn_file must be an exisiting filename")
    res_directory = "{}/{}/results/firstlevel".format(str(Path(conn_file).parents[0]), Path(conn_file).stem)
    if not isdir(res_directory):
        raise ValueError("the first levels of the selected conn project have not been estimated yet")
    subjects_name = extract_subjects_name_from_conn_structure(conn_file)
    analysis_to_extract = to_extract(res_directory)  
    if len(analysis_to_extract) == 0:
        raise Exception ("No ROI to ROI first level to analyse")
    print("I have found {} ROI to ROI analysis with {} subjects. Proceeding to values extraction".format(len(analysis_to_extract), len(subjects_name)))
    for analysis in  analysis_to_extract:
        analysis_name = Path(analysis).stem
        files_to_treat = glob("{}/resultsROI_*_Condition001.mat".format(analysis))
        list_of_df = list()
        for name, res in zip(subjects_name, files_to_treat):
            list_of_df.append(create_roi_to_roi_line(name,res))
        df = pd.concat(list_of_df)
        if output_dir:
            file_name = "{}/{}_{}subjects_{}rois.csv".format(output_dir, analysis_name, df.shape[0], df.shape[1])
        else:
            main_directory = "{}/{}".format(str(Path(conn_file).parents[0]), Path(conn_file).stem)
            file_name = "{}/{}_{}subjects_{}rois.csv".format(main_directory, analysis_name, df.shape[0], df.shape[1])
        df.to_csv(file_name)
    
    

