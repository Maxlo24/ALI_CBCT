from GlobalVar import*

import csv
import SimpleITK as sitk
import numpy as np
import torch
import os
import json

# ----- MONAI ------
# from monai.losses import DiceCELoss
# from monai.inferers import sliding_window_inference
from monai.transforms import (
    transform,
    Compose,
    AddChannel,
    ScaleIntensity,
    SpatialCrop,
    BorderPad,
)

from utils import(
    ReadFCSV,
    SetSpacing,
    ItkToSitk,
    GenControlePoint,
    WriteJson
)

# #####################################
#  Environement
# #####################################

class Environement :
    def __init__(
        self,
        padding,
        verbose = False
    ) -> None:
        """
        Args:
            images_path : path of the image with all the different scale,
            landmark_fiducial : path of the fiducial list linked with the image,
        """
        self.padding = padding.astype(np.int16)
        self.verbose = verbose
        self.transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist()),ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)])
        # self.transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist())])
        self.predicted_landmarks = {}

    def LoadImages(self,images_path):
        self.images_path = images_path

        data = []
        sizes = []
        spacings = []
        origins = []
        
        for path in images_path:
            img = sitk.ReadImage(path)
            # sizes.append(np.array(img.GetSize()))
            spacings.append(np.array(img.GetSpacing()))
            origin = img.GetOrigin()
            origins.append(np.array([origin[2],origin[1],origin[0]]))
            img_ar = sitk.GetArrayFromImage(img)
            sizes.append(np.shape(img_ar))
            data.append(torch.from_numpy(self.transform(img_ar)).type(torch.float16))

        self.dim = len(data)
        self.data = data
        self.sizes = sizes
        self.spacings = spacings
        self.origins = origins

        self.ResetLandmarks()


    def GenerateImages(self,ref_img,spacing_lst):
        self.images_path = ref_img

        data = []
        sizes = []
        spacings = []
        origins = []
        
        for spacing in spacing_lst:
            img = ItkToSitk(SetSpacing(ref_img,[spacing,spacing,spacing]))
            # sizes.append(np.array(img.GetSize()))
            spacings.append(np.array(img.GetSpacing()))
            origin = img.GetOrigin()
            origins.append(np.array([origin[2],origin[1],origin[0]]))
            img_ar = sitk.GetArrayFromImage(img)#.astype(dtype=np.float32)
            sizes.append(np.shape(img_ar))
            data.append(torch.from_numpy(self.transform(img_ar)).type(torch.float16))

        self.dim = len(data)
        self.data = data
        self.sizes = sizes
        self.spacings = spacings
        self.origins = origins

        self.ResetLandmarks()

    def ResetLandmarks(self):
        dim_lm = []
        for i in range(self.dim):
            dim_lm.append({})
        self.dim_landmarks = dim_lm

    def AddPredictedLandmark(self,lm_id,lm_pos):
        self.predicted_landmarks[lm_id] = lm_pos

    def LoadFCSVLandmarks(self,fiducial_path):
        fcsv_lm_dic = ReadFCSV(fiducial_path)
        for lm,data in fcsv_lm_dic.items() :
            lm_ph_coord = np.array([float(data["z"]),float(data["y"]),float(data["x"])])
            for i in range(self.dim):
                lm_coord = (lm_ph_coord+ abs(self.origins[i]))/self.spacings[i]
                lm_coord = lm_coord.astype(int)
                self.dim_landmarks[i][lm] = lm_coord

    def LoadJsonLandmarks(self,fiducial_path):
        with open(fiducial_path) as f:
            data = json.load(f)

        markups = data["markups"][0]["controlPoints"]
        for markup in markups:
            lm_ph_coord = np.array([markup["position"][2],markup["position"][1],markup["position"][0]])
            for i in range(self.dim):
                lm_coord = (lm_ph_coord+ abs(self.origins[i]))/self.spacings[i]
                lm_coord = lm_coord.astype(int)
                self.dim_landmarks[i][markup["label"]] = lm_coord

        # print(markups)

    def LandmarkIsPresent(self,landmark):
        if landmark in self.dim_landmarks[0].keys():
            return True
        else:
            if self.verbose:
                print(landmark, "missing in patient ", os.path.basename(os.path.dirname(self.images_path[0])))
            return False

    def GetLandmarkPos(self,dim,landmark):
        return self.dim_landmarks[dim][landmark]

    def GetL2DistFromLandmark(self, dim, position, target):
        label_pos = self.GetLandmarkPos(dim,target)
        return np.linalg.norm(position-label_pos)**2

    def GetSize(self,dim):
        return self.sizes[dim]

    def GetSpacing(self,dim):
        return self.spacings[dim]

    def GetZone(self,dim,center,crop_size):
        cropTransform = SpatialCrop(center.tolist() + self.padding,crop_size)
        rescale = ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)
        crop = rescale(cropTransform(self.data[dim])).type(torch.float16)
        return crop

    def GetRewardLst(self,dim,position,target,mvt_matrix):
        reward_lst = []
        agent_dist = self.GetL2DistFromLandmark(dim,position,target)
        for move in mvt_matrix:
            neighbor_coord = position + move
            dist_from_lm = self.GetL2DistFromLandmark(dim,neighbor_coord,target)
            reward_lst.append(agent_dist - dist_from_lm)
        return reward_lst

    def GetBestMove(self,dim,position,target,mvt_matrix):
        best_action = np.argmax(self.GetRewardLst(dim,position,target,mvt_matrix))
        return best_action

    def GetRandomPos(self,dim,target,radius):
        min_coord = [0,0,0]
        max_coord = self.GetSize(dim)

        close_to_lm = 0
        if dim == 0:
            close_to_lm = np.random.rand()

        if close_to_lm >= 0.2:
            rand_coord = np.random.randint(1, self.GetSize(dim), dtype=np.int16)
        else:
            rand_coord = np.random.randint([1,1,1], radius*2) - radius
            rand_coord = self.GetLandmarkPos(dim,target) + rand_coord
            for axe,val in enumerate(rand_coord):
                rand_coord[axe] = max(val,min_coord[axe])
                rand_coord[axe] = min(val,max_coord[axe])
        
        rand_coord = rand_coord.astype(np.int16)
        # rand_coord = self.GetLandmarkPos(dim,target)
        return rand_coord

    def GetRandomSample(self,dim,target,radius,crop_size,mvt_matrix):
        rand_coord = self.GetRandomPos(dim,target,radius)
        sample = self.GetSample(dim,target,rand_coord,crop_size,mvt_matrix)
        return sample

    def GetSample(self,dim,target,coord,crop_size,mvt_matrix):
        sample = {}
        sample["state"] = self.GetZone(dim,coord,crop_size)
        sample["target"] = self.GetBestMove(dim,coord,target,mvt_matrix)
        # sample["size"] = self.GetZone(dim,rand_coord,crop_size).size()
        # sample["coord"] = coord
        # sample["target"] = torch.from_numpy(GetTargetOutputFromAction(best_action))
        return sample

    def SavePredictedLandmarks(self):

        ref_origin = self.origins[-1]
        ref_spacing = self.spacings[-1]
        physical_origin = abs(ref_origin/ref_spacing)

        # print(ref_origin,ref_spacing,physical_origin)

        landmark_dic = {}
        for landmark,pos in self.predicted_landmarks.items():

            real_label_pos = (pos-physical_origin)*ref_spacing
            real_label_pos = [real_label_pos[2],real_label_pos[1],real_label_pos[0]]
            # print(real_label_pos)
            if GROUPES[landmark] in landmark_dic.keys():
                landmark_dic[GROUPES[landmark]].append({"label": landmark, "coord":real_label_pos})
            else:landmark_dic[GROUPES[landmark]] = [{"label": landmark, "coord":real_label_pos}]


        # print(landmark_dic)

        for group,list in landmark_dic.items():

            scan_name = os.path.basename(self.images_path[-1]).split(".")
            elements = scan_name[0].split("_")
            patient = elements[0] + "_" + elements[1]
            json_name = patient + "_pred_lm_"+group+".mrk.json"

            file_path = os.path.join(os.path.dirname(self.images_path[0]),json_name)
            groupe_data = {}
            for lm in list:
                groupe_data[lm["label"]] = {"x":lm["coord"][0],"y":lm["coord"][1],"z":lm["coord"][2]}

            lm_lst = GenControlePoint(groupe_data)
            WriteJson(lm_lst,file_path)

            # fiducial_name = patient + "_pred_lm_"+group+".fcsv"
            # # print(fiducial_name)

            # file_name = os.path.join(os.path.dirname(self.images_path[0]),fiducial_name)
            # f = open(file_name,'w')
            
            # f.write("# Markups fiducial file version = 4.11\n")
            # f.write("# CoordinateSystem = LPS\n")
            # f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
            # for id,element in enumerate(list):
            #     f.write(str(id)+","+str(element["coord"][0])+","+str(element["coord"][1])+","+str(element["coord"][2])+",0,0,0,1,1,1,0,"+element["label"]+",,\n")
            # # # f.write( data + "\n")
            # f.close