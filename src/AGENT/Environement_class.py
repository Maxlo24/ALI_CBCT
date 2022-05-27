from importlib.resources import path
import GlobalVar as GV
# from skimage import exposure

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
        patient_id,
        padding,
        device,
        correct_contrast = False,
        verbose = False,

    ) -> None:
        """
        Args:
            images_path : path of the image with all the different scale,
            landmark_fiducial : path of the fiducial list linked with the image,
        """
        self.patient_id = patient_id
        self.padding = padding.astype(np.int16)
        self.device = device
        self.verbose = verbose
        self.transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist())])
        # self.transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist()),ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)])

        self.scale_nbr = 0

        # self.transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist())])
        self.available_lm = []

        self.data = {}

        self.predicted_landmarks = {}

    def LoadImages(self,images_path):

        for scale,path in images_path.items():
            data = {"path":path}
            img = sitk.ReadImage(path)
            img_ar = sitk.GetArrayFromImage(img)
            data["image"] = torch.from_numpy(self.transform(img_ar)).type(torch.int16)

            data["spacing"] = np.array(img.GetSpacing())
            origin = img.GetOrigin()
            data["origin"] = np.array([origin[2],origin[1],origin[0]])
            data["size"] = np.array(np.shape(img_ar))

            data["landmarks"] = {}

            self.data[scale] = data
            self.scale_nbr += 1

    def LoadJsonLandmarks(self,fiducial_path):
        # print(fiducial_path)
        # test = []

        with open(fiducial_path) as f:
            data = json.load(f)

        markups = data["markups"][0]["controlPoints"]
        for markup in markups:
            if markup["label"] not in GV.LABELS:
                print(fiducial_path)
                print(f"{GV.bcolors.WARNING}WARNING : {markup['label']} is an unusual landmark{GV.bcolors.ENDC}")
            # test.append(markup["label"])
            mark_pos = markup["position"]
            lm_ph_coord = np.array([mark_pos[2],mark_pos[1],mark_pos[0]])
            self.available_lm.append(markup["label"])
            for scale,scale_data in self.data.items():
                lm_coord = ((lm_ph_coord+ abs(scale_data["origin"]))/scale_data["spacing"]).astype(np.int16)
                scale_data["landmarks"][markup["label"]] = lm_coord

        # print(test)


    def SavePredictedLandmarks(self,scale_key):
        img_path = self.data[scale_key]["path"]
        print(f"Saving predicted landmarks for patient{self.patient_id} at scale {scale_key}")

        ref_origin = self.data[scale_key]["origin"]
        ref_spacing = self.data[scale_key]["spacing"]
        physical_origin = abs(ref_origin/ref_spacing)

        # print(ref_origin,ref_spacing,physical_origin)

        landmark_dic = {}
        for landmark,pos in self.data[scale_key]["landmarks"].items():

            real_label_pos = (pos-physical_origin)*ref_spacing
            real_label_pos = [real_label_pos[2],real_label_pos[1],real_label_pos[0]]
            # print(real_label_pos)
            if GV.LABEL_GROUPES[landmark] in landmark_dic.keys():
                landmark_dic[GV.LABEL_GROUPES[landmark]].append({"label": landmark, "coord":real_label_pos})
            else:landmark_dic[GV.LABEL_GROUPES[landmark]] = [{"label": landmark, "coord":real_label_pos}]


        # print(landmark_dic)

        for group,list in landmark_dic.items():


            json_name = f"{self.patient_id}_lm_Pred_{group}.mrk.json"

            file_path = os.path.join(os.path.dirname(img_path),json_name)
            groupe_data = {}
            for lm in list:
                groupe_data[lm["label"]] = {"x":lm["coord"][0],"y":lm["coord"][1],"z":lm["coord"][2]}

            lm_lst = GenControlePoint(groupe_data)
            WriteJson(lm_lst,file_path)

    def ResetLandmarks(self):
        for scale in self.data.keys():
            self.data[scale]["landmarks"] = {}

        self.available_lm = []

    def LandmarkIsPresent(self,landmark):
        return landmark in self.available_lm

    def GetLandmarkPos(self,scale,landmark):
        return self.data[scale]["landmarks"][landmark]

    def GetL2DistFromLandmark(self, scale, position, target):
        label_pos = self.GetLandmarkPos(scale,target)
        return np.linalg.norm(position-label_pos)**2

    def GetZone(self,scale,center,crop_size):
        cropTransform = SpatialCrop(center.tolist() + self.padding,crop_size)
        rescale = ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)
        crop = cropTransform(self.data[scale]["image"])
        # print(tor ch.max(crop))
        crop = rescale(crop).type(torch.float32)
        return crop

    def GetRewardLst(self,scale,position,target,mvt_matrix):
        agent_dist = self.GetL2DistFromLandmark(scale,position,target)
        get_reward = lambda move : agent_dist - self.GetL2DistFromLandmark(scale,position + move,target)
        reward_lst = list(map(get_reward,mvt_matrix))
        return reward_lst
    
    def GetRandomPoses(self,scale,target,radius,pos_nbr):
        if scale == GV.SCALE_KEYS[0]:
            porcentage = 0.2 #porcentage of data around landmark
            centered_pos_nbr = int(porcentage*pos_nbr)
            rand_coord_lst = self.GetRandomPosesInAllScan(scale,pos_nbr-centered_pos_nbr)
            rand_coord_lst += self.GetRandomPosesArounfLabel(scale,target,radius,centered_pos_nbr)
        else:
            rand_coord_lst = self.GetRandomPosesArounfLabel(scale,target,radius,pos_nbr)

        return rand_coord_lst

    def GetRandomPosesInAllScan(self,scale,pos_nbr):
        max_coord = self.data[scale]["size"]
        get_rand_coord = lambda x: np.random.randint(1, max_coord, dtype=np.int16)
        rand_coord_lst = list(map(get_rand_coord,range(pos_nbr)))
        return rand_coord_lst
    
    def GetRandomPosesArounfLabel(self,scale,target,radius,pos_nbr):
        min_coord = [0,0,0]
        max_coord = self.data[scale]["size"]
        landmark_pos = self.GetLandmarkPos(scale,target)

        get_random_coord = lambda x: landmark_pos + np.random.randint([1,1,1], radius*2) - radius

        rand_coords = map(get_random_coord,range(pos_nbr))

        correct_coord = lambda coord: np.array([min(max(coord[0],min_coord[0]),max_coord[0]),min(max(coord[1],min_coord[1]),max_coord[1]),min(max(coord[2],min_coord[2]),max_coord[2])])
        rand_coords = list(map(correct_coord,rand_coords))

        return rand_coords

    def GetSampleFromPoses(self,scale,target,pos_lst,crop_size,mvt_matrix):

        get_sample = lambda coord : {
            "state":self.GetZone(scale,coord,crop_size),
            "target": np.argmax(self.GetRewardLst(scale,coord,target,mvt_matrix))
            }
        sample_lst = list(map(get_sample,pos_lst))

        return sample_lst

    # def GenerateImages(self,ref_img,spacing_lst):
    #     self.images_path = [ref_img]

    #     data = []
    #     sizes = []
    #     spacings = []
    #     origins = []
    #     print(spacing_lst)
    #     for spacing in spacing_lst:
    #         img = ItkToSitk(SetSpacing(ref_img,[spacing,spacing,spacing]))
    #         # sizes.append(np.array(img.GetSize()))
    #         spacings.append(np.array(img.GetSpacing()))
    #         origin = img.GetOrigin()
    #         origins.append(np.array([origin[2],origin[1],origin[0]]))
    #         img_ar = sitk.GetArrayFromImage(img)#.astype(dtype=np.float32)
    #         img_ar_correct = self.CorrectImgContrast(img_ar,0.01, 0.99)
    #         sizes.append(np.array(np.shape(img_ar_correct)))
    #         data.append(torch.from_numpy(self.transform(img_ar_correct)).type(torch.int16))

    #     self.dim = len(data)
    #     self.data = data
    #     self.sizes = sizes
    #     self.spacings = spacings
    #     self.origins = origins

    #     self.ResetLandmarks()



    # def AddPredictedLandmark(self,lm_id,lm_pos):
    #     self.predicted_landmarks[lm_id] = lm_pos



        # print(markups)


    # def GenerateLandmarkImg(self,dim):

    #     # print(self.original_data[dim][0].shape)
    #     new_image = sitk.Image(np.array(self.original_data[dim][0].shape).tolist(), sitk.sitkInt16)
        
    #     img_ar = np.array(sitk.GetArrayFromImage(new_image))#.astype(dtype=np.float32)
    #     img_ar = img_ar.transpose(2,1,0)

    #     lm_id = 0
    #     for lm,pos in self.original_dim_landmarks[dim].items():
    #         lm_id+=1
    #         ppos = pos + [1,1,1]
    #         img_ar[ppos[0]][ppos[1]][ppos[2]] = lm_id
    #         img_ar[ppos[0]+1][ppos[1]][ppos[2]] = lm_id
    #         img_ar[ppos[0]][ppos[1]+1][ppos[2]] = lm_id
    #         img_ar[ppos[0]][ppos[1]][ppos[2]+1] = lm_id
    #         img_ar[ppos[0]-1][ppos[1]][ppos[2]] = lm_id
    #         img_ar[ppos[0]][ppos[1]-1][ppos[2]] = lm_id
    #         img_ar[ppos[0]][ppos[1]][ppos[2]-1] = lm_id

    #     output = torch.from_numpy(img_ar).unsqueeze(0).type(torch.int16)
    #     # print(output.shape)
    #     return output

    # TRANSFORMS

    # def SetRandomRotation(self):
    #     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     rand_angle_theta =  2*np.random.rand()*np.pi
    #     rand_angle_phi = np.random.rand()*np.pi

    #     BillinearRot = Rotate(
    #         angle=[rand_angle_theta,rand_angle_phi,0],
    #         mode="bilinear",
    #         keep_size=False,
    #     )
    #     NearestRot = Rotate(
    #         angle=[rand_angle_theta,rand_angle_phi,0],
    #         mode="nearest",
    #         keep_size=False,
    #     )
    #     pad = BorderPad(spatial_border=(self.padding - [1,1,1]).tolist())

    #     for i,data in enumerate(self.original_data):
    #         print("Rotating env:",self.images_path[i])


    #         # os.system("gpustat")
    #         # print(self.sizes)

    #         # print(data.shape)
    #         rotated_img = BillinearRot(data)
    #         rotated_lm_img = NearestRot(self.GenerateLandmarkImg(i))

    #         self.data[i] = pad(rotated_img)
    #         self.sizes[i] = rotated_img[0].shape - np.array([2,2,2])

    #         # print(self.sizes)
    #         # print(type(self.sizes[i]))
    #         # print(type(rotated_img[0].shape - self.padding*2))
    #         lm_array = rotated_lm_img[0].numpy().astype(np.int16)
    #         # print(np.shape(lm_array))

    #         torch.cuda.empty_cache()

    #         lm_id = 0
    #         for lm,pos in self.dim_landmarks[i].items():
    #             lm_id+=1
    #             ppos = np.where(lm_array==lm_id)
    #             if len(ppos[0])>0:
    #                 self.dim_landmarks[i][lm] = np.array([int(np.mean(ppos[0])),int(np.mean(ppos[1])),int(np.mean(ppos[2]))]) - [1,1,1]
    #             else:
    #                 print("Lost :", lm)
    # GET



    # def GetBestMove(self,dim,position,target,mvt_matrix):
    #     best_action = np.argmax(self.GetRewardLst(dim,position,target,mvt_matrix))
    #     return best_action

    # def GetRandomPos(self,dim,target,radius):
    #     min_coord = [0,0,0]
    #     max_coord = self.GetSize(dim)

    #     close_to_lm = 0
    #     if dim == 0:
    #         close_to_lm = np.random.rand()

    #     if close_to_lm >= 0.2:
    #         rand_coord = np.random.randint(1, self.GetSize(dim), dtype=np.int16)
    #     else:
    #         rand_coord = np.random.randint([1,1,1], radius*2) - radius
    #         rand_coord = self.GetLandmarkPos(dim,target) + rand_coord
    #         for axe,val in enumerate(rand_coord):
    #             rand_coord[axe] = max(val,min_coord[axe])
    #             rand_coord[axe] = min(val,max_coord[axe])
        
    #     rand_coord = rand_coord.astype(np.int16)
    #     # rand_coord = self.GetLandmarkPos(dim,target)
    #     return rand_coord


        # for axe,val in enumerate(rand_coord):
        #     rand_coord[axe] = max(val,min_coord[axe])
        #     rand_coord[axe] = min(val,max_coord[axe])


    # def GetRandomSample(self,dim,target,radius,crop_size,mvt_matrix):
    #     rand_coord = self.GetRandomPos(dim,target,radius)
    #     sample = self.GetSample(dim,target,rand_coord,crop_size,mvt_matrix)
    #     return sample

    # def GetSample(self,dim,target,coord,crop_size,mvt_matrix):
    #     sample = {}
    #     sample["state"] = self.GetZone(dim,coord,crop_size)
    #     sample["target"] = self.GetBestMove(dim,coord,target,mvt_matrix)
    #     # sample["size"] = self.GetZone(dim,rand_coord,crop_size).size()
    #     # sample["coord"] = coord
    #     # sample["target"] = torch.from_numpy(GetTargetOutputFromAction(best_action))
    #     return sample


    # def SavePredictedLandmarks(self,id = "pred_lm"):

    #     ref_origin = self.origins[-1]
    #     ref_spacing = self.spacings[-1]
    #     physical_origin = abs(ref_origin/ref_spacing)

    #     # print(ref_origin,ref_spacing,physical_origin)

    #     landmark_dic = {}
    #     for landmark,pos in self.predicted_landmarks.items():

    #         real_label_pos = (pos-physical_origin)*ref_spacing
    #         real_label_pos = [real_label_pos[2],real_label_pos[1],real_label_pos[0]]
    #         # print(real_label_pos)
    #         if GROUPES[landmark] in landmark_dic.keys():
    #             landmark_dic[GROUPES[landmark]].append({"label": landmark, "coord":real_label_pos})
    #         else:landmark_dic[GROUPES[landmark]] = [{"label": landmark, "coord":real_label_pos}]


    #     # print(landmark_dic)

    #     for group,list in landmark_dic.items():

    #         scan_name = os.path.basename(self.images_path[-1]).split(".")
    #         try:
    #             elements = scan_name[0].split("_")
    #             patient = elements[0] + "_" + elements[1]
    #             json_name = patient + "_"+id+"_"+group+".mrk.json"
    #         except :
    #             json_name = scan_name[0] + "_" + id + ".mrk.json"

    #         file_path = os.path.join(os.path.dirname(self.images_path[0]),json_name)
    #         groupe_data = {}
    #         for lm in list:
    #             groupe_data[lm["label"]] = {"x":lm["coord"][0],"y":lm["coord"][1],"z":lm["coord"][2]}

    #         lm_lst = GenControlePoint(groupe_data)
    #         WriteJson(lm_lst,file_path)

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


    # def SaveCBCT(self,dim,out_path):
    #     data = self.data[dim]
    #     # scan_name = os.path.basename(self.images_path[dim])
    #     print("Saving:",out_path)
    #     # print(self.padding)
    #     # print(data[0][self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1],self.padding[2]:-self.padding[2]].shape)
    #     output = sitk.GetImageFromArray(
    #         data[0][self.padding[0]:-self.padding[0],
    #         self.padding[1]:-self.padding[1],
    #         self.padding[2]:-self.padding[2]]
    #     )
    #     # output = sitk.GetImageFromArray(
    #     #     data[0].type(torch.float32)
    #     # )
    #     writer = sitk.ImageFileWriter()
    #     writer.SetFileName(out_path)
    #     writer.Execute(output)

    # def SaveEnvironmentState(self,out_dir,id):

    #     # self.SavePredictedLandmarks(id)

    #     landmarks = []
    #     for lm_dic in self.dim_landmarks:
    #         dic = {}
    #         # print(lm_dic)
    #         for lm,pos in lm_dic.items():
    #             dic[lm] = pos.tolist()
    #         landmarks.append(dic)

    #     for dim in range(self.dim):
    #         path = os.path.basename(self.images_path[dim]).replace("_scan_","_scan_"+id+"_")
    #         path = os.path.join(out_dir,path)
    #         # print(path)
    #         self.SaveCBCT(dim,path)

    #         data = {
    #             # "files_path":self.images_path,
    #             "Landmarks":landmarks[dim]
    #         }
    #         path = os.path.basename(self.images_path[dim]).replace("_scan_","_landmarks_"+id+"_")
    #         path = path.split(".")[0] + ".json"
    #         path = os.path.join(out_dir,path)
    #         with open(path, 'w') as fp:
    #             json.dump(data, fp, ensure_ascii=False, indent=1)

    # def CorrectImgContrast(self,img,min_porcent,max_porcent):
    #     img_min = np.min(img)
    #     img_max = np.max(img)
    #     img_range = img_max - img_min
    #     # print(img_min,img_max,img_range)

    #     definition = 1000
    #     histo = np.histogram(img,definition)
    #     cum = np.cumsum(histo[0])
    #     cum = cum - np.min(cum)
    #     cum = cum / np.max(cum)

    #     res_high = list(map(lambda i: i> max_porcent, cum)).index(True)
    #     res_max = (res_high * img_range)/definition + img_min

    #     res_low = list(map(lambda i: i> min_porcent, cum)).index(True)
    #     res_min = (res_low * img_range)/definition + img_min

    #     img = np.where(img > res_max, res_max,img)
    #     img = np.where(img < res_min, res_min,img)

    #     return img


