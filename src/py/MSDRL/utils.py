import csv
import numpy as np
import SimpleITK as sitk
import itk
import os
import glob
import torch
import json

from GlobalVar import*

import seaborn as sns
import matplotlib.pyplot as plt


def GetAgentLst(agents_param):
    agent_lst = []
    for fcsv in agents_param["landmarks"]:
        for label in LABELS[fcsv]:
            print("Generating Agent for the lamdmark :" , label)
            agt = agents_param["type"](
                targeted_landmark=label,
                # models=DRLnet,
                movements = agents_param["movements"],
                env_dim = agents_param["dim"],
                FOV=agents_param["FOV"],
                start_pos_radius = agents_param["spawn_rad"],
                verbose = agents_param["verbose"]
            )
            agent_lst.append(agt)
    return agent_lst

def GetTrainingEnvironementsAgents(environments_param,agents_param):
    scan_lst = []
    for spacing in environments_param["spacings"]:
       scan_lst.append([])

    U_fcsv_lst = []
    L_fcsv_lst = []
    CB_fcsv_lst = []
    

    print("Reading folder : ", environments_param["dir"])
    print("Selected spacings : ", environments_param["spacings"])
    		
    normpath = os.path.normpath("/".join([environments_param["dir"], '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            baseName = os.path.basename(img_fn)
            if True in [scan in baseName for scan in ["scan","Scan"]]:
                for i,spacing in enumerate(environments_param["spacings"]):
                    if "_"+str(spacing) in baseName:
                        scan_lst[i].append(img_fn)

        if os.path.isfile(img_fn) and ".mrk.json" in img_fn:
            baseName = os.path.basename(img_fn)
            if "_U." in baseName :
                U_fcsv_lst.append(img_fn)
            elif "_L." in baseName :
                L_fcsv_lst.append(img_fn)
            elif "_CB." in baseName :
                CB_fcsv_lst.append(img_fn)


    data_lst = []
    for n in range(0,len(scan_lst[0])):
        data = {}

        images_path = []
        for i,spacing in enumerate(environments_param["spacings"]):
            images_path.append(scan_lst[i][n])
        data["images"] = images_path
        data["U"] = U_fcsv_lst[n]
        data["L"] = L_fcsv_lst[n]
        data["CB"] = CB_fcsv_lst[n]

        data_lst.append(data)

    # print(data_lst)

    
    environement_lst = []
    for data in data_lst:
        print("Generating Environement for :" , os.path.dirname(data["images"][0]))
        env = environments_param["type"](
            padding = np.array(agents_param["FOV"])/2+1,
            verbose=environments_param["verbose"]
            )
        env.LoadImages(data["images"])
        for lm in agents_param["landmarks"]:
            env.LoadJsonLandmarks(data[lm])

        environement_lst.append(env)

    agent_lst = GetAgentLst(agents_param)

    print("Number of Environement generated :",len(environement_lst))
    print("Number of Agent generated :",len(agent_lst))

    return environement_lst,agent_lst

def GenPredictEnvironment(environments_param,agents_param):
    scan_lst = []
    print("Reading folder : ", environments_param["dir"])

    normpath = os.path.normpath("/".join([environments_param["dir"], '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            scan_lst.append(img_fn)


    environement_lst = []
    for scan in scan_lst:
        print("Generating Environement for :" , scan)
        env = environments_param["type"](
            padding = np.array(agents_param["FOV"])/2+1,
            verbose=environments_param["verbose"]
            )
        env.GenerateImages(scan,environments_param["spacings"])

        environement_lst.append(env)
    return environement_lst

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

    # print(brainDic)
    out_dic = {}
    for l_key in brainDic.keys():
        networks = []
        for n_key in range(len(brainDic[l_key].keys())):
            networks.append(brainDic[l_key][str(n_key)])

        out_dic[l_key] = networks

    return out_dic

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
    _ the labels in the list

    Parameters
    ----------
    filePath
     path of the .fcsv file 
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
                "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 3.0,
                "glyphType": "Sphere3D",
                "glyphScale": 1.0,
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

def  ReslutAccuracy(fiducial_dir):

    error_dic = {"labels":[], "error":[]}
    patients = {}
    normpath = os.path.normpath("/".join([fiducial_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn) and ".mrk.json" in img_fn:
            baseName = os.path.basename(img_fn)
            patient = os.path.basename(os.path.dirname(img_fn))
            if patient not in patients.keys():
                patients[patient] = {"U":{},"L":{},"CB":{}}

            if "_pred_" in baseName:
                if "_U." in baseName :
                    patients[patient]["U"]["pred"]=img_fn
                elif "_L." in baseName :
                    patients[patient]["L"]["pred"]=img_fn
                elif "_CB." in baseName :
                    patients[patient]["CB"]["pred"]=img_fn
            else:
                if "_U." in baseName :
                    patients[patient]["U"]["target"]=img_fn
                elif "_L." in baseName :
                    patients[patient]["L"]["target"]=img_fn
                elif "_CB." in baseName :
                    patients[patient]["CB"]["target"]=img_fn


    f = open(os.path.join(fiducial_dir,"Result.txt"),'w')
    for patient,fiducials in patients.items():
        print("Results for patient",patient)
        f.write("Results for patient "+ str(patient)+"\n")

        for group,targ_res in fiducials.items():
            print(" ",group,"landmarks:")
            f.write(" "+ str(group)+" landmarks:\n")
            if "pred" in targ_res.keys():
                target_lm_dic = ReadJson(targ_res["target"])
                pred_lm_dic = ReadJson(targ_res["pred"])
                for lm,t_data in target_lm_dic.items():
                    if lm in pred_lm_dic.keys():
                        a = np.array([float(t_data["x"]),float(t_data["y"]),float(t_data["z"])])
                        p_data = pred_lm_dic[lm]
                        b = np.array([float(p_data["x"]),float(p_data["y"]),float(p_data["z"])])
                        # print(a,b)
                        dist = np.linalg.norm(a-b)
                        error_dic["labels"].append(lm)
                        error_dic["error"].append(dist)
                        print("  ",lm,"error = ", dist)
                        f.write("  "+ str(lm)+" error = "+str(dist)+"\n")
            f.write("\n")
        f.write("\n")
    
    f.close
    return error_dic


def PlotResults(data):
    sns.set_theme(style="whitegrid")
    # data = {"labels":["B","B","N","N","B","N"], "error":[0.1,0.5,1.6,1.9,0.3,1.3]}    

    # print(tips)
    ax = sns.violinplot(x="labels", y="error", data=data)
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
    

def CheckCrops(Master,agent):
    Master.GeneratePosDataset("train",Master.max_train_memory_size)
    Master.GeneratePosDataset("val",Master.max_val_memory_size)

    if not os.path.exists("crop"):
        os.makedirs("crop")

    for key,value in Master.pos_dataset.items():
        for k,v in value.items():
            # print(v)
            for n,dq in enumerate(v):
                for dim,obj in enumerate(dq):
                    # print(n,obj)
                    arr = obj["env"].GetSample(n,agent.target,obj["coord"],agent.FOV,agent.movement_matrix)
                    # print(arr)
                    output = sitk.GetImageFromArray(arr["state"][0][:].type(torch.float32))
                    writer = sitk.ImageFileWriter()
                    writer.SetFileName(f"crop/test_{key}_{k}_{dim}_{n}.nii.gz")
                    writer.Execute(output)