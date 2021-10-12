from GlobalVar import*

import csv
import SimpleITK as sitk
import numpy as np
import torch

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

# #####################################
#  Environement
# #####################################

class Environement :
    def __init__(
        self,
        images_path,
        padding,
        verbose = False
    ) -> None:
        """
        Args:
            images_path : path of the image with all the different scale,
            landmark_fiducial : path of the fiducial list linked with the image,
        """
        self.padding = padding.astype(np.int16)
        self.LoadImages(images_path)
        self.ResetLandmarks()
        self.verbose = verbose

    def LoadImages(self,images_path):
        transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist()),ScaleIntensity(minv = 0.0, maxv = 1.0, factor = None)])

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
            data.append(torch.from_numpy(transform(img_ar)))

        self.dim = len(data)
        self.data = data
        self.sizes = sizes
        self.spacings = spacings
        self.origins = origins

    def ResetLandmarks(self):
        dim_lm = []
        for i in range(self.dim):
            dim_lm.append({})
        self.dim_landmarks = dim_lm


    def LoadLandmarks(self,fiducial_path):
        fcsv_lm_lst = ReadFCSV(fiducial_path)
        for lm in fcsv_lm_lst :
            lm_ph_coord = np.array([float(lm["z"]),float(lm["y"]),float(lm["x"])])
            for i in range(self.dim):
                lm_coord = (lm_ph_coord+ abs(self.origins[i]))/self.spacings[i]
                lm_coord = lm_coord.astype(int)
                self.dim_landmarks[i][lm["label"]] = lm_coord

    def GetSize(self,dim):
        return self.sizes[dim]

    def GetSpacing(self,dim):
        return self.spacings[dim]

    def GetLandmarkPos(self,dim,landmark):
        return self.dim_landmarks[dim][landmark]

    def GetL2DistFromLandmark(self, dim, position, target):
        label_pos = self.GetLandmarkPos(dim,target)
        return np.linalg.norm(position-label_pos)**2

    def GetZone(self,dim,center,crop_size):
        cropTransform = SpatialCrop(center.tolist() + self.padding,crop_size)
        crop = cropTransform(self.data[dim])
        return crop

    def GetRandomSample(self,dim,target,radius,crop_size):
        sample = {}
        min_coord = [0,0,0]
        max_coord = self.GetSize(dim)

        if dim == 0:
            rand_coord = np.random.randint(1, self.GetSize(dim), dtype=np.int16)
        else:
            rand_coord = np.random.randint([1,1,1], radius*2) - radius
            rand_coord = self.GetLandmarkPos(dim,target) + rand_coord
            for axe,val in enumerate(rand_coord):
                rand_coord[axe] = max(val,min_coord[axe])
                rand_coord[axe] = min(val,max_coord[axe])
            rand_coord = rand_coord.astype(np.int16)

        best_action = np.argmax(self.GetRewardLst(dim,rand_coord,target))

        sample["state"] = self.GetZone(dim,rand_coord,crop_size)
        # sample["size"] = self.GetZone(dim,rand_coord,crop_size).size()
        # sample["coord"] = rand_coord
        # sample["target"] = best_action
        sample["target"] = torch.from_numpy(GetTargetOutputFromAction(best_action))
        # sample["reward"] = np.where()

        return sample

    def GetRewardLst(self,dim,position,target):
        reward_lst = []
        agent_dist = self.GetL2DistFromLandmark(dim,position,target)
        for move in MOVEMENT_MATRIX:
            neighbor_coord = position + move
            dist_from_lm = self.GetL2DistFromLandmark(dim,neighbor_coord,target)
            reward_lst.append(agent_dist - dist_from_lm)
        return reward_lst

    def GetBestMove(self):
        best_action = np.argmax(self.GetRewardLst())
        if self.verbose:
            print("Best move is ", MOVEMENT_MATRIX[best_action])
        return best_action

    
def ReadFCSV(filePath):
    """
    Read fiducial file ".fcsv" and return a liste of landmark dictionnary

    Parameters
    ----------
    filePath
     path of the .fcsv file 
    """
    Landmark_lst = []
    with open(filePath, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if "#" not in row[0]:
                landmark = {}
                landmark["id"], landmark["x"], landmark["y"], landmark["z"], landmark["label"] = row[0], row[1], row[2], row[3], row[11]
                Landmark_lst.append(landmark)
    return Landmark_lst