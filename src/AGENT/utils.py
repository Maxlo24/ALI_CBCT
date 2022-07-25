import csv
import numpy as np
import SimpleITK as sitk
import itk
import os
import glob
import torch
import json

import GlobalVar as GV
# from skimage import exposure


import seaborn as sns
import matplotlib.pyplot as plt


def GetAgentLst(agents_param, lm_lst):
    print("-- Generating agents --")

    agent_lst = []
    for label in lm_lst:
        print(f"{GV.bcolors.OKCYAN}Generating Agent for the lamdmark: {GV.bcolors.OKBLUE}{label}{GV.bcolors.ENDC}")
        agt = agents_param["type"](
            targeted_landmark=label,
            movements = agents_param["movements"],
            scale_keys = agents_param["scale_keys"],
            FOV=agents_param["FOV"],
            start_pos_radius = agents_param["spawn_rad"],
            speed_per_scale = agents_param["speed_per_scale"],
            focus_radius = agents_param["focus_radius"],
            verbose = agents_param["verbose"]
        )
        agent_lst.append(agt)

    print(f"{GV.bcolors.OKGREEN}{len(agent_lst)} agent successfully generated. {GV.bcolors.ENDC}")

    return agent_lst


# environments_param = {
#     "type" : Environement,
#     "dir" : args.dir_scans,
#     "scale_spacing" : scale_spacing,
#     "padding" : agent_FOV,
#     "landmark_group" : landmark_group,
#     "verbose" : False,
# }



def GetEnvironmentLst(environments_param):


    error = False
    scale_spacing = environments_param["scale_spacing"]

    patients = {}
    
    scan_summary_dic = {}
    landmark_summary_group = {}


    print(f"{GV.bcolors.OKCYAN}Reading folder {GV.bcolors.ENDC}:"+environments_param["dir"])
    print("Selected scale_spacing : ", scale_spacing)
    
    spacing_str = ["sp"+str(spacing).replace(".","-") for spacing in scale_spacing]

    normpath = os.path.normpath("/".join([environments_param["dir"], '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        baseName = os.path.basename(img_fn).split(".")[0]

        if os.path.isfile(img_fn):

            if "_lm_" in baseName:
                patient_info = baseName.split("_lm_")
            elif "_scan_" in baseName:
                patient_info = baseName.split("_scan_")
            elif "_Pred" in baseName or 'Result' in baseName:
                print(f'{GV.bcolors.OKCYAN}Skipping file {GV.bcolors.ENDC}: {img_fn}')
            else:
                print(f"{GV.bcolors.WARNING}Unknown file format : {img_fn} ----->{GV.bcolors.FAIL} missing '_lm_' or '_scan_' in the filename{GV.bcolors.ENDC}")
                error = True

            patient = patient_info[0]
            info = patient_info[1]

            if patient not in patients:
                patients[patient] = {"scans" : {}, "landmarks" : {}}

            if "_scan_" in baseName and info in spacing_str:
                # print(info)
                sp = info.replace("sp","")
                patients[patient]["scans"][sp] = img_fn

                if sp not in scan_summary_dic:
                    scan_summary_dic[sp] = 1
                else:
                    scan_summary_dic[sp] += 1

            elif "_lm_" in baseName:
                patients[patient]["landmarks"][info] = img_fn

                if info not in landmark_summary_group:
                    landmark_summary_group[info] = 1
                else:
                    landmark_summary_group[info] += 1



    nbr_patient = len(patients.keys())
    print(f"Number of patients: {nbr_patient}")

    for sp in scan_summary_dic.keys():
        nbr_scan = scan_summary_dic[sp]
        # print("Number of scans for the spacing ", sp, " : ", nbr_scan)
        if nbr_scan != nbr_patient:
            error = True
            print(f"{GV.bcolors.FAIL}FAIL : the number of scans for the spacing {sp} is different from the number of patients{GV.bcolors.ENDC}")
    for lm in landmark_summary_group.keys():
        nbr_lm = landmark_summary_group[lm]
        # print("Number of landmarks for the landmark ", lm, " : ", nbr_lm)
        if nbr_lm != nbr_patient:
            print(f"{GV.bcolors.WARNING}Warning : the number of landmarks for the landmark {lm} is different from the number of patients{GV.bcolors.ENDC}")

    if error:
        raise ValueError(f"{GV.bcolors.FAIL}Error in the dataset folder{GV.bcolors.ENDC}")

    print("-- Generating environments --")
    environement_lst = []
    for patient,data in patients.items():
        print(f"{GV.bcolors.OKCYAN}Generating Environement for the patient: {GV.bcolors.OKBLUE}{patient}{GV.bcolors.ENDC}")
        env = environments_param["type"](
            patient_id = patient,
            device = environments_param["device"],
            padding = environments_param["padding"],
            verbose = environments_param["verbose"],

        )

        env.LoadImages(data["scans"])
        for lm_file in data["landmarks"].values():
            env.LoadJsonLandmarks(lm_file)
        environement_lst.append(env)

    print(f"{GV.bcolors.OKGREEN}{len(environement_lst)} environment successfully generated. {GV.bcolors.ENDC}")

    return environement_lst

    
def GenEnvironmentLst(patient_dic ,env_type, padding = 1, device = GV.DEVICE):
    environement_lst = []
    for patient,data in patient_dic.items():
        print(f"{GV.bcolors.OKCYAN}Generating Environement for the patient: {GV.bcolors.OKBLUE}{patient}{GV.bcolors.ENDC}")
        env = env_type(
            patient_id = patient,
            device = device,
            padding = padding,
            verbose = False,
        )
        env.LoadImages(data["scans"])
        environement_lst.append(env)
    return environement_lst



def CorrectHisto(filepath,outpath,min_porcent=0.01,max_porcent = 0.95,i_min=-1500, i_max=4000):

    print("Correcting scan contrast :", filepath)
    input_img = sitk.ReadImage(filepath) 
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(input_img)


    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min
    # print(img_min,img_max,img_range)

    definition = 1000
    histo = np.histogram(img,definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i> max_porcent, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    res_low = list(map(lambda i: i> min_porcent, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min

    res_min = max(res_min,i_min)
    res_max = min(res_max,i_max)


    # print(res_min,res_min)

    img = np.where(img > res_max, res_max,img)
    img = np.where(img < res_min, res_min,img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output


def CorrectContrast(img_array,min_porcent=0.01,max_porcent = 0.95,i_min=-1500, i_max=4000):
    img_min = np.min(img_array)
    img_max = np.max(img_array)
    img_range = img_max - img_min
    # print(img_min,img_max,img_range)

    definition = 1000
    histo = np.histogram(img_array,definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i> max_porcent, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    res_low = list(map(lambda i: i> min_porcent, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min

    res_min = max(res_min,i_min)
    res_max = min(res_max,i_max)


    # print(res_min,res_min)

    img = np.where(img_array > res_max, res_max,img_array)
    img = np.where(img_array < res_min, res_min,img_array)

    return img

def GetBrain(dir_path):
    brainDic = {}
    normpath = os.path.normpath("/".join([dir_path, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and ".pth" in img_fn:
            lab = os.path.basename(os.path.dirname(os.path.dirname(img_fn)))
            num = os.path.basename(os.path.dirname(img_fn))
            if lab in brainDic.keys():
                brainDic[lab][num] = img_fn
            else:
                network = {num : img_fn}
                brainDic[lab] = network

    return brainDic

    # print(brainDic)
    # out_dic = {}
    # for l_key in brainDic.keys():
    #     networks = []
    #     for n_key in range(len(brainDic[l_key].keys())):
    #         networks.append(brainDic[l_key][n_key])

    #     out_dic[l_key] = networks

    # return out_dic

def ResampleImage(input,size,spacing,origin,direction,interpolator,VectorImageType):
        ResampleType = itk.ResampleImageFilter[VectorImageType, VectorImageType]

        resampleImageFilter = ResampleType.New()
        resampleImageFilter.SetOutputSpacing(spacing.tolist())
        resampleImageFilter.SetOutputOrigin(origin)
        resampleImageFilter.SetOutputDirection(direction)
        resampleImageFilter.SetInterpolator(interpolator)
        resampleImageFilter.SetSize(size)
        resampleImageFilter.SetInput(input)
        resampleImageFilter.Update()

        resampled_img = resampleImageFilter.GetOutput()
        return resampled_img

def ItkToSitk(itk_img):
    new_sitk_img = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_img), isVector=itk_img.GetNumberOfComponentsPerPixel()>1)
    new_sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
    new_sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
    new_sitk_img.SetDirection(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten())
    return new_sitk_img

def SetSpacing(filepath,output_spacing=[0.5, 0.5, 0.5],outpath=-1):
    """
    Set the spacing of the image at the wanted scale 

    Parameters
    ----------
    filePath
     path of the image file 
    output_spacing
     whanted spacing of the new image file (default : [0.5, 0.5, 0.5])
    outpath
     path to save the new image
    """

    print("Resample :", filepath, ", with spacing :", output_spacing)
    img = itk.imread(filepath)
    # arr_img = itk.GetArrayFromImage(img)
    # print(np.min(arr_img),np.max(arr_img))
    # arr_img = np.where(arr_img < 2500, arr_img,2500)
    # print(np.min(arr_img),np.max(arr_img))

    # img_rescale = itk.GetImageFromArray(arr_img)

    spacing = np.array(img.GetSpacing())
    output_spacing = np.array(output_spacing)

    if not np.array_equal(spacing,output_spacing):

        size = itk.size(img)
        scale = spacing/output_spacing

        output_size = (np.array(size)*scale).astype(int).tolist()
        output_origin = img.GetOrigin()

        #Find new origin
        output_physical_size = np.array(output_size)*np.array(output_spacing)
        input_physical_size = np.array(size)*spacing
        output_origin = np.array(output_origin) - (output_physical_size - input_physical_size)/2.0

        img_info = itk.template(img)[1]
        pixel_type = img_info[0]
        pixel_dimension = img_info[1]

        VectorImageType = itk.Image[pixel_type, pixel_dimension]

        if True in [seg in os.path.basename(filepath) for seg in ["seg","Seg"]]:
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        else:
            InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Scan with spacing :", output_spacing)

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,output_size,output_spacing,output_origin,img.GetDirection(),interpolator,VectorImageType)

        if outpath != -1:
            itk.imwrite(resampled_img, outpath)
        return resampled_img

    else:
        # print("Already at the wanted spacing")
        if outpath != -1:
            itk.imwrite(img, outpath)
        return img


def CorrectCSV(filePath, Rcar = [" ", "-1"], Rlab = ["RGo_LGo", "RCo_LCo", "LCo_RCo", "LGo_RGo"]):
    """
    Remove all the unwanted parts of a fiducial file ".fcsv" :
    - the spaces " "
    - the dash ! "-1"
    - the labels in the list

    Parameters
    ----------
    filePath
     path of the .fcsv file 
     Rcar : caracter to remove
     Rlab : landmark to remove
    """
    file_data = []
    with open(filePath, mode='r') as fcsv_file:
        csv_reader = csv.reader(fcsv_file)
        for row in csv_reader:
            file_data.append(row)

    with open(filePath, mode='w') as fcsv_file:
        writer = csv.writer(fcsv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row in file_data:
            keep = True
            if "#" not in row[0]:
                for car in Rcar : row[11] = row[11].replace(car,"")
                if True in [label in row[11] for label in Rlab] : keep = False

            if(keep):
                writer.writerow(row)

def RenameLandmarkCSV(fiducial_path, rename_lst):

        with open(fiducial_path) as f:
            data = json.load(f)

        markups = data["markups"][0]["controlPoints"]
        for markup in markups:
            for key in rename_lst:
                if key in markup["label"]:
                    markup["label"] = key

        data["markups"][0]["controlPoints"] = markups

        with open(fiducial_path, 'w') as f:
            json.dump(data, f)

def GetImageInfo(filepath):
    ref = sitk.ReadImage(filepath)
    ref_size = np.array(ref.GetSize())
    ref_spacing = np.array(ref.GetSpacing())
    ref_origin = np.array(ref.GetOrigin())
    ref_direction = np.array(ref.GetDirection())

    return ref_size,ref_spacing,ref_origin,ref_direction

def CreateNewImageFromRef(filepath):
    ref_size,ref_spacing,ref_origin,ref_direction = GetImageInfo(filepath)
    image = sitk.Image(ref_size.tolist(), sitk.sitkInt16)
    image.SetOrigin(ref_origin.tolist())
    image.SetSpacing(ref_spacing.tolist())
    image.SetDirection(ref_direction.tolist())

    return image

def GetSphereMaskCoord(h,w,l,center,rad):
    X, Y, Z = np.ogrid[:h, :w, :l]
    dist_from_center = np.sqrt((X - center[2])**2 + (Y-center[1])**2 + (Z-center[0])**2)
    mask = dist_from_center <= rad

    return np.array(np.where(mask))

def PlotAgentPath(agent,rad = 2):
    paths = agent.position_mem
    environement = agent.environement

    for dim,path in enumerate(paths):
        refImg = environement.images_path[dim]
        ref_size,ref_spacing,ref_origin,ref_direction = GetImageInfo(refImg)
        image_3D = CreateNewImageFromRef(refImg)
        for coord in path :

            maskCoord = GetSphereMaskCoord(ref_size[0],ref_size[1],ref_size[2],coord,rad)
            maskCoord=maskCoord.tolist()

            for i in range(0,len(maskCoord[0])):
                image_3D.SetPixel([maskCoord[0][i],maskCoord[1][i],maskCoord[2][i]],1)

        maskCoord = GetSphereMaskCoord(ref_size[0],ref_size[1],ref_size[2],environement.GetLandmarkPos(dim,agent.target),rad*2)
        maskCoord=maskCoord.tolist()

        for i in range(0,len(maskCoord[0])):
            image_3D.SetPixel([maskCoord[0][i],maskCoord[1][i],maskCoord[2][i]],2)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.basename(refImg))
        writer.Execute(image_3D)

        print("Agent path generated at :", refImg)
        

def ReadFCSV(filePath):
    """
    Read fiducial file ".fcsv" and return a liste of landmark dictionnary

    Parameters
    ----------
    filePath
     path of the .fcsv file 
    """
    Landmark_dic = {}
    with open(filePath, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if "#" not in row[0]:
                landmark = {}
                landmark["id"], landmark["x"], landmark["y"], landmark["z"], landmark["label"] = row[0], float(row[1]), float(row[2]), float(row[3]), row[11]
                Landmark_dic[row[11]] = landmark
    return Landmark_dic

def SaveJsonFromFcsv(file_path,out_path):
    """
    Save a .fcsv in a .json file

    Parameters
    ----------
    file_path : path of the .fcsv file 
    out_path : path of the .json file 
    """
    groupe_data = ReadFCSV(file_path)
    lm_lst = GenControlePoint(groupe_data)
    WriteJson(lm_lst,out_path)

def GenControlePoint(groupe_data):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark,data in groupe_data.items():
        id+=1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [data["x"], data["y"], data["z"]],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "preview"
        }
        lm_lst.append(controle_point)

    return lm_lst

def WriteJson(lm_lst,out_path):
    false = False
    true = True
    file = {
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
    "markups": [
        {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": false,
            "labelFormat": "%N-%d",
            "controlPoints": lm_lst,
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.0],
                "color": [0.5, 0.5, 0.5],
                "selectedColor": [0.26666666666666669, 0.6745098039215687, 0.39215686274509806],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 2.0,
                "glyphType": "Sphere3D",
                "glyphScale": 2.0,
                "glyphSize": 5.0,
                "useGlyphScale": true,
                "sliceProjection": false,
                "sliceProjectionUseFiducialColor": true,
                "sliceProjectionOutlinedBehindSlicePlane": false,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": false,
                "snapMode": "toVisibleSurface"
            }
        }
    ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=4)

    f.close

def  ReslutAccuracy(environments, scale):

    print(f'{GV.bcolors.OKGREEN}====== RESULTS ======{GV.bcolors.ENDC}')

    error_dic = {"labels":[], "error":[]}
    patients = {}

    fail = 0
    max = 0
    mean = 0
    nbr_pred = 0
    error_lst = []

    for environment in environments:
        for landmark,pos in environment.predicted_landmarks.items():
            if landmark in environment.data[scale]["landmarks"].keys():
                error = np.linalg.norm(np.array(pos) - np.array(environment.data[scale]["landmarks"][landmark]))
                error_dic["labels"].append(landmark)
                error_dic["error"].append(error)

                if error > max: max = error
                if error > 10: fail +=1
                mean += error
                nbr_pred+=1
                error_lst.append(error)

    print(fail,'fail')
    print("STD :", np.std(error_lst))
    print('Error max :',max)
    print('Mean error',mean/nbr_pred)
    
    return error_dic

def ResultDiscretAccuracy(env_lst,sp):
    error_dic = {"labels":[], "error":[]}
    for env in env_lst:
        for lm,pos in env.predicted_landmarks.items():
            dist = np.linalg.norm(pos-env.GetLandmarkPos(-1,lm))*sp
            error_dic["labels"].append(lm)
            error_dic["error"].append(dist)
    return error_dic


def PlotResults(data):
    sns.set_theme(style="whitegrid")
    # data = {"labels":["B","B","N","N","B","N"], "error":[0.1,0.5,1.6,1.9,0.3,1.3]}    

    # print(tips)
    ax = sns.violinplot(x="labels", y="error", data=data, cut=0)
    plt.show()


def ReadJson(fiducial_path):
    lm_dic = {}

    with open(fiducial_path) as f:
            data = json.load(f)
    markups = data["markups"][0]["controlPoints"]
    for markup in markups:
        lm_dic[markup["label"]] = {"x":markup["position"][0],"y":markup["position"][1],"z":markup["position"][2]}
    return lm_dic
def SaveFiducialFromArray(data,scan_image,outpath,label_list):
    """
    Generate a fiducial file from an array with label

    Parameters
    ----------
    data
     array with the labels
    scan_image
     scan of referance
    outpath
     outpath of the fiducial path
    label_list
     liste of label associated with the array
     """

    print("Generating fiducial file at : ", os.path.basename(scan_image))
    ref_size,ref_spacing,ref_origin,ref_direction = GetImageInfo(scan_image)
    physical_origin = abs(ref_origin/ref_spacing)
    print(ref_direction)

    # print(physical_origin)

    label_pos_lst = []
    for i in range(len(label_list)):
        label_pos = np.array(np.where(data==i+1))
        label_pos = label_pos.tolist()
        label_coords = np.array([label_pos[2][0],label_pos[1][0],label_pos[0][0]], dtype='float')
        nbrPoint = 1
        for j in range(1,len(label_pos[0])):
            nbrPoint+=1
            label_coords += np.array([label_pos[2][j],label_pos[1][j],label_pos[0][j]] , dtype='float')
            # label_coords.append(coord)

        label_coord = label_coords/nbrPoint #+ np.array([0.45,0.45,0.45])

        label_pos = (label_coord-physical_origin)*ref_spacing
        label_pos_lst.append({"label": label_list[i], "coord" : label_pos})
        # print(label_pos)

    fiducial_name = os.path.basename(scan_image).split(".")[0]
    fiducial_name = fiducial_name.replace("scan","CBCT")
    fiducial_name = fiducial_name.replace("or","CB")
    fiducial_name += ".fcsv"

    file_name = os.path.join(outpath,fiducial_name)
    f = open(file_name,'w')
    
    f.write("# Markups fiducial file version = 4.11\n")
    f.write("# CoordinateSystem = LPS\n")
    f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
    for id,element in enumerate(label_pos_lst):
        f.write(str(id)+","+str(element["coord"][0])+","+str(element["coord"][1])+","+str(element["coord"][2])+",0,0,0,1,1,1,0,"+element["label"]+",,\n")
    # # f.write( data + "\n")
    f.close
    

def CheckCrops(Master,agent,dim):
    Master.GeneratePosDataset("train",Master.max_train_memory_size)
    Master.GeneratePosDataset("val",Master.max_val_memory_size)
    _,tds = Master.GenerateDataLoader("train",agent,dim)
    _,vds = Master.GenerateDataLoader("val",agent,dim)

    # print(tds)

    if not os.path.exists("crop"):
        os.makedirs("crop")

    for n,dic in enumerate(tds):
        arr = dic["state"]
        print(arr[0].shape)
        output = sitk.GetImageFromArray(arr[0].type(torch.float32))
        writer = sitk.ImageFileWriter()
        writer.SetFileName(f"crop/test_{agent.target}_{dim}_{n}.nii.gz")
        writer.Execute(output)

    # for key,value in Master.pos_dataset.items():
    #     for k,v in value.items():
    #         # print(v)
    #         for n,dq in enumerate(v):
    #             for dim,obj in enumerate(dq):
    #                 # print(n,obj)
    #                 arr = obj["env"].GetSample(n,agent.target,obj["coord"],agent.FOV,agent.movement_matrix)
    #                 # print(arr)
    #                 output = sitk.GetImageFromArray(arr["state"][0][:].type(torch.float32))
    #                 writer = sitk.ImageFileWriter()
    #                 writer.SetFileName(f"crop/test_{key}_{k}_{dim}_{n}.nii.gz")
    #                 writer.Execute(output)