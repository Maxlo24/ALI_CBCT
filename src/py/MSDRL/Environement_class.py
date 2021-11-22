from GlobalVar import*

import csv
import SimpleITK as sitk
import numpy as np
import torch
import os
import json
import copy

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
    Rotated,
    Rotate
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
        device,
        verbose = False
    ) -> None:
        """
        Args:
            images_path : path of the image with all the different scale,
            landmark_fiducial : path of the fiducial list linked with the image,
        """

        self.padding = padding.astype(np.int16)
        self.device = device
        self.verbose = verbose
        self.transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist()),ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)])
        # self.transform = Compose([AddChannel(),BorderPad(spatial_border=[1,1,1]),ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)])

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
            sizes.append(np.array(np.shape(img_ar)))
            data.append(torch.from_numpy(self.transform(img_ar)).type(torch.float16))

        self.dim = len(data)
        self.data = data
        self.original_data = data.copy()
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
            sizes.append(np.array(np.shape(img_ar)))
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

        self.original_dim_landmarks = copy.deepcopy(self.dim_landmarks)


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

        self.original_dim_landmarks = copy.deepcopy(self.dim_landmarks)

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

        self.original_dim_landmarks = copy.deepcopy(self.dim_landmarks)

        # print(markups)

    def LandmarkIsPresent(self,landmark):
        if landmark in self.dim_landmarks[0].keys():
            return True
        else:
            if self.verbose:
                print(landmark, "missing in patient ", os.path.basename(os.path.dirname(self.images_path[0])))
            return False

    def GenerateLandmarkImg(self,dim):

        new_image = sitk.Image(np.array(self.original_data[dim][0].shape).tolist(), sitk.sitkInt16)
        
        img_ar = np.array(sitk.GetArrayFromImage(new_image))#.astype(dtype=np.float32)
        img_ar = img_ar.transpose(2,1,0)

        lm_id = 0
        for lm,pos in self.original_dim_landmarks[dim].items():
            lm_id+=1
            ppos = pos + self.padding
            img_ar[ppos[0]][ppos[1]][ppos[2]] = lm_id
            img_ar[ppos[0]+1][ppos[1]][ppos[2]] = lm_id
            img_ar[ppos[0]][ppos[1]+1][ppos[2]] = lm_id
            img_ar[ppos[0]][ppos[1]][ppos[2]+1] = lm_id
            img_ar[ppos[0]-1][ppos[1]][ppos[2]] = lm_id
            img_ar[ppos[0]][ppos[1]-1][ppos[2]] = lm_id
            img_ar[ppos[0]][ppos[1]][ppos[2]-1] = lm_id

        output = torch.from_numpy(img_ar).unsqueeze(0).type(torch.int16)
        # print(output.shape)
        return output

    # TRANSFORMS

    def SetRandomRotation(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rand_angle_theta =  2*np.random.rand()*np.pi
        rand_angle_phi = np.random.rand()*np.pi
        # rot_transform = Rotated(
        #     keys=["image","landmarks"],
        #     angle=[rand_angle_theta,rand_angle_phi,0],
        #     mode=("bilinear","nearest"),
        #     keep_size=False,
        # )
        BillinearRot = Rotate(
            angle=[rand_angle_theta,rand_angle_phi,0],
            mode="bilinear",
            keep_size=False,
        )
        NearestRot = Rotate(
            angle=[rand_angle_theta,rand_angle_phi,0],
            mode="nearest",
            keep_size=False,
        )


        for i,data in enumerate(self.original_data):
            print("Rotating env:",self.images_path[i])
            # data_dic = {
            #     "image":data.to(self.device),
            #     "landmarks":self.GenerateLandmarkImg(i).to(self.device)
            #     }

            os.system("gpustat")

            # print(rotated_data["image"])
            rotated_img = BillinearRot(data.to(self.device)).cpu()
            self.data[i] = rotated_img
            self.sizes[i] = rotated_img[0].shape - self.padding*2
            os.system("gpustat")

            # print(type(self.sizes[i]))
            # print(type(rotated_img[0].shape - self.padding*2))
            rotated_lm_img = NearestRot(self.GenerateLandmarkImg(i).to(self.device)).cpu()
            lm_array = rotated_lm_img[0].numpy().astype(np.int16)
            # print(np.shape(lm_array))
            os.system("gpustat")

            torch.cuda.empty_cache()
            os.system("gpustat")


            lm_id = 0
            for lm,pos in self.dim_landmarks[i].items():
                lm_id+=1
                ppos = np.where(lm_array==lm_id)
                self.dim_landmarks[i][lm] = np.array([int(np.mean(ppos[0])),int(np.mean(ppos[1])),int(np.mean(ppos[2]))]) - self.padding


    # GET

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
        agent_dist = self.GetL2DistFromLandmark(dim,position,target)
        get_reward = lambda move : agent_dist - self.GetL2DistFromLandmark(dim,position + move,target)
        reward_lst = list(map(get_reward,mvt_matrix))
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

    def GetRandomPoses(self,dim,target,radius,pos_nbr):
        if dim == 0:
            porcentage = 0.2 #porcentage of data around landmark
            centered_pos_nbr = int(porcentage*pos_nbr)
            rand_coord_lst = self.GetRandomPosesInAllScan(dim,pos_nbr-centered_pos_nbr)
            rand_coord_lst += self.GetRandomPosesArounfLabel(dim,target,radius,centered_pos_nbr)
        else:
            rand_coord_lst = self.GetRandomPosesArounfLabel(dim,target,radius,pos_nbr)

        return rand_coord_lst

    def GetRandomPosesInAllScan(self,dim,pos_nbr):
        max_coord = self.GetSize(dim)
        get_rand_coord = lambda x: np.random.randint(1, max_coord, dtype=np.int16)
        rand_coord_lst = list(map(get_rand_coord,range(pos_nbr)))
        return rand_coord_lst
    
    def GetRandomPosesArounfLabel(self,dim,target,radius,pos_nbr):
        min_coord = [0,0,0]
        max_coord = self.GetSize(dim)
        landmark_pos = self.GetLandmarkPos(dim,target)

        get_random_coord = lambda x: landmark_pos + np.random.randint([1,1,1], radius*2) - radius

        rand_coords = map(get_random_coord,range(pos_nbr))

        correct_coord = lambda coord: np.array([min(max(coord[0],min_coord[0]),max_coord[0]),min(max(coord[1],min_coord[1]),max_coord[1]),min(max(coord[2],min_coord[2]),max_coord[2])])
        rand_coords = list(map(correct_coord,rand_coords))

        return rand_coords
        # for axe,val in enumerate(rand_coord):
        #     rand_coord[axe] = max(val,min_coord[axe])
        #     rand_coord[axe] = min(val,max_coord[axe])


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

    def GetSampleFromPoses(self,dim,target,pos_lst,crop_size,mvt_matrix):

        get_sample = lambda coord : {
            "state":self.GetZone(dim,coord,crop_size),
            "target": np.argmax(self.GetRewardLst(dim,coord,target,mvt_matrix))
            }
        sample_lst = list(map(get_sample,pos_lst))

        return sample_lst

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


    def SaveCBCT(self,dim,out_path):
        data = self.data[dim]
        scan_name = os.path.basename(self.images_path[dim])
        print("Saving:",scan_name)
        # print(self.padding)
        # print(data[0][self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1],self.padding[2]:-self.padding[2]].shape)
        output = sitk.GetImageFromArray(
            data[0][self.padding[0]:-self.padding[0],
            self.padding[1]:-self.padding[1],
            self.padding[2]:-self.padding[2]].type(torch.float32)
        )
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(out_path,scan_name))
        writer.Execute(output)

    def SaveEnvironmentState(self):

        # self.SaveCBCT(0,"")

        landmarks = []
        for lm_dic in self.dim_landmarks:
            dic = {}
            for lm,pos in lm_dic.items():
                dic[lm] = pos.tolist()
            landmarks.append(dic)

        data = {
            "files_path":self.images_path,
            "Landmarks":landmarks
        }

        with open('data.json', 'w') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=1)
