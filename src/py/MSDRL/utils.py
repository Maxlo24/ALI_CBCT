import csv
import numpy as np
import SimpleITK as sitk
import itk
import os
import glob

from GlobalVar import*
from Models_class import DRLnet
from Agents_class import (
    DQNAgent,
    RLAgent
)
from Environement_class import Environement
from TrainingManager_class import TrainingMaster

def GetEnvironementsAgents(dir_scans,spacing_lst,agent_type,agent_FOV,landmarks):
    dim = len(spacing_lst)
    scan_lst = []
    for spacing in spacing_lst:
       scan_lst.append([])

    U_fcsv_lst = []
    L_fcsv_lst = []
    CB_fcsv_lst = []
    

    print("Reading folder : ", dir_scans)
    print("Selected spacings : ", spacing_lst)
    		
    normpath = os.path.normpath("/".join([dir_scans, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            baseName = os.path.basename(img_fn)
            if True in [scan in baseName for scan in ["scan","Scan"]]:
                for i,spacing in enumerate(spacing_lst):
                    if "_"+str(spacing) in baseName:
                        scan_lst[i].append(img_fn)

        if os.path.isfile(img_fn) and ".fcsv" in img_fn:
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
        for i,spacing in enumerate(spacing_lst):
            images_path.append(scan_lst[i][n])
        data["images"] = images_path
        data["u"] = U_fcsv_lst[n]
        data["l"] = L_fcsv_lst[n]
        data["cb"] = CB_fcsv_lst[n]

        data_lst.append(data)

    # print(data_lst)

    
    environement_lst = []
    for data in data_lst:
        print("Generating Environement for :" , os.path.dirname(data["images"][0]))
        env = Environement(
            data["images"],
            np.array(agent_FOV)/2,
            verbose=True
            )
        for fcsv in landmarks:
            env.LoadLandmarks(data[fcsv])

        environement_lst.append(env)

    agent_lst = []
    for fcsv in landmarks:
        for label in LABELS[fcsv]:
            print("Generating Agent for the lamdmark :" , label)
            agt = agent_type(
                targeted_landmark=label,
                # models=DRLnet,
                env_dim = dim,
                FOV=agent_FOV,
                start_pos_radius = 40,
                verbose = True
            )
            agent_lst.append(agt)

    print("Number of Environement generated :",len(environement_lst))
    print("Number of Agent generated :",len(agent_lst))

    return environement_lst,agent_lst



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

    print("Reading:", filepath)
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
        

