#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:48:11 2019

@author: zipat
"""

from scipy.io import loadmat
from pathlib import Path, PureWindowsPath, PurePath
import numpy as np
from glob import glob
from os.path import isdir, isfile, sep
from itertools import chain, combinations
import pandas as pd
from nipype.interfaces.spm.preprocess import Normalize12
from shutil import move
from nilearn.input_data import NiftiLabelsMasker
from pandas import DataFrame, read_csv

def recursively_extract_element_from_array(array_to_search):
        t = np.ndarray
        while t == np.ndarray:
            if not(array_to_search.size):
                return None
            el = array_to_search[0]
            t = type(el)
            array_to_search = el
        return(el)

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

class conn_project(object):
     def __init__(self, conn_file):
        self.conn_file = str(Path(conn_file))
        if not isfile(conn_file):
            raise ValueError("conn_file must be an exisiting filename")
        self.conn_dir = "{}/{}".format(str(Path(conn_file).parents[0]), Path(conn_file).stem)
        if not isdir(self.conn_dir):
            raise ValueError("conn_file is not associated with an exisiting directory")
        self.results_dir = "{}/{}/results/firstlevel".format(str(Path(conn_file).parents[0]), Path(conn_file).stem)
        if not isdir(self.conn_dir):
            raise ValueError("the first level of the conn project have not been estimated")
        self.text_file_separator = ","

     def roi_to_roi_to_extract(self):
        all_dir = glob("{}/*".format(self.results_dir))
        all_dir = [el for el in [el for el in all_dir if isdir(el)] if len(glob("{}/resultsROI*".format(el))) != 0]
        if len(all_dir) == 0:
            raise Exception ("No ROI to ROI first level to analyse")
        self.roi_to_roi_dirs = all_dir

     def voxel_to_voxel_to_extract(self):
        all_dir = glob("{}/*".format(self.results_dir))
        all_dir = [el for el in [el for el in all_dir if isdir(el)] if len(glob("{}/BETA*".format(el))) != 0 and len(glob("{}/ICA*".format(el))) == 0 and len(glob("{}/*Source*".format(el))) == 0]
        if len(all_dir) == 0:
            raise Exception ("No Voxel to Voxel first level to analyse")
        self.voxel_to_voxel_dirs = all_dir

        
     def extract_subjects_name_from_conn_structure(self, preprocessing_prefix = "swau"):
        conn_file_mat = loadmat(self.conn_file)
        root_conn_functionals = conn_file_mat["CONN_x"]["Setup"][0][0][0][0][6][0]
        subject_name = []
        for el in range(0,len(root_conn_functionals)):
            sub_el = recursively_extract_element_from_array(root_conn_functionals[el])
            if "\\" in sub_el: 
                subject_name.append(PureWindowsPath(sub_el).stem.replace(preprocessing_prefix, ""))
            else:
                subject_name.append(Path(sub_el).stem.replace(preprocessing_prefix, ""))
        self.subjects_name = subject_name
        self.subjects_name.sort()
        
        
     def find_func_dir(self, parent_tree = None, local_branch = None):
        conn_file_mat = loadmat(self.conn_file)
        func_dir_example = conn_file_mat["CONN_x"]["Setup"][0][0][0][0][6][0][0][0][0][0][0][0]
        if "\\" in func_dir_example:
            func_dir_example = PureWindowsPath(func_dir_example)
        else:
            func_dir_example = Path(func_dir_example)
        if not(parent_tree is None):
            self.func_dir = PurePath(local_branch) / sep.join(func_dir_example.parents[0].parts[parent_tree:])
        else:
            self.func_dir = func_dir_example.parents[0]
     
     def charge_inverse_warp(self):
         self.inverse_warp = glob("{}/iy*".format(self.func_dir))
         self.inverse_warp.sort()
         
     def extract_roi_to_roi_values_from_conn(self, output_dir = None):
        """Extract roi to roi values from an already estimated conn first level.
                       
            Parameters
            ----------
            conn_file : a string poiting to an existing file that is a conn model with the first level estimated
            output_dir : the directory were the extracted data are to be written in csv files, if not specified
                        the csv file are written in the main directory of the conn design
            
            
            The function returns as much csv files as roi to roi estimated designs in the conn first_level directory with
            the connectivity values of the source ROIs only
            
        """
        print("I have found {} ROI to ROI analysis with {} subjects. Proceeding to values extraction".format(len(self.roi_to_roi_dirs), len(self.subjects_name)))
        for analysis in  self.roi_to_roi_dirs:
            analysis_name = Path(analysis).stem
            files_to_treat = glob("{}/resultsROI_*_Condition001.mat".format(analysis))
            list_of_df = list()
            for name, res in zip(self.subjects_name, files_to_treat):
                list_of_df.append(create_roi_to_roi_line(name,res))
            df = pd.concat(list_of_df)
            if output_dir:
                file_name = "{}/{}_{}subjects_{}rois.csv".format(output_dir, analysis_name, df.shape[0], df.shape[1])
            else:
                main_directory = "{}/{}".format(str(Path(self.conn_file).parents[0]), Path(self.conn_file).stem)
                file_name = "{}/{}_{}subjects_{}rois.csv".format(main_directory, analysis_name, df.shape[0], df.shape[1])
            df.to_csv(file_name)
            
     def send_voxel_to_voxel_to_single_subject_space(self):
         for analysis in self.voxel_to_voxel_dirs:
             analysis_name = Path(analysis).stem
             files_to_treat = glob("{}/BETA_*_Component001.nii".format(analysis))
             for name, img, iy in zip(self.subjects_name, files_to_treat, self.inverse_warp):
                 print("Treating analysis {} of subject {}".format(analysis_name, name))
                 nrml = Normalize12()
                 nrml.inputs.jobtype = "write"
                 nrml.inputs.deformation_file = iy
                 nrml.inputs.apply_to_files = img
                 nrml.run()
                 move("{}/w{}.nii".format(Path(img).parents[0], Path(img).stem),
                 "{}/{}_{}_single_subject.nii".format(Path(img).parents[0], name, analysis_name))
     
     def extract_roiwise_voxel2voxel_index(self, templates):
         for analysis in self.voxel_to_voxel_dirs:
             analysis_name = Path(analysis).stem
             files_to_treat = glob("{}/BETA_*_Component001.nii".format(analysis))
             files_to_treat.sort()
             for template_name, template_dict in templates.items():
                 template_img = template_dict["image"]
                 template_csv = template_dict["csv"]
                 mskr = NiftiLabelsMasker(template_img)
                 mat = mskr.fit_transform(files_to_treat)
                 col_names = read_csv(template_csv, sep = self.text_file_separator).loc[:,"ROIname"][read_csv(template_csv, sep = self.text_file_separator).loc[:,"ROIid"].isin(np.array(mskr.labels_).round())]
                 df = DataFrame(mat, columns = col_names)
                 df.loc[:,"Subject"] = self.subjects_name
                 cols = ["Subject"] + list(col_names)
                 df = df[cols]
                 df.to_csv("{}/{}_{}_{}subjects.csv".format(self.conn_dir, analysis_name, template_name, mat.shape[0]), sep = self.text_file_separator)

        
                     

