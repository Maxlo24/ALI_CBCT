import os
import itk
import numpy as np
import shutil
import SimpleITK as sitk
import csv
import pandas

from numpy.ma.core import getdata

# #####################################
#  label list
# #####################################

U_labels = ['PNS','ANS','A','UR6apex','UR3apex','U1apex','UL3apex','UL6apex','UR6d','UR6m','UR3tip','UItip','UL3tip','UL6m','UL6d']
L_labels = ['RCo','RGo','LR6apex','LR7apex','L1apex','Me','Gn','Pog','B','LL6apex','LL7apex','LGo','LCo','LR6d','LR6m','LItip','LL6m','LL6d']
CB_labels = ['Ba','S','N']

# #####################################
#  SetFile spacing
# #####################################

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


def SetSpacingFromRef(file,refFile,outpath=-1):


    img = itk.imread(file)

    ref = itk.imread(refFile)

    img_sp = np.array(img.GetSpacing()) 
    img_size = np.array(itk.size(img))

    ref_sp = np.array(ref.GetSpacing())
    ref_size = np.array(itk.size(ref))

    if not (np.array_equal(img_sp,ref_sp) and np.array_equal(img_size,ref_size)):
        ref_info = itk.template(ref)[1]
        pixel_type = ref_info[0]
        pixel_dimension = ref_info[1]

        VectorImageType = itk.Image[pixel_type, pixel_dimension]

        # print(VectorImageType)

        if True in [seg in os.path.basename(file) for seg in ["seg","Seg"]]:
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        else:
            InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Scan with spacing :", output_spacing)

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,ref_size.tolist(),ref_sp,ref.GetOrigin(),ref.GetDirection(),interpolator,VectorImageType)

        if outpath != -1:
            itk.imwrite(resampled_img, outpath)
        return resampled_img

    else:
        # print("Already at the wanted spacing")
        if outpath != -1:
            itk.imwrite(img, outpath)
        return img




def SetSpacing(filepath,output_spacing=[0.5, 0.5, 0.5],outpath=-1):
    r"""
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


# #####################################
#  Keep the landmark of the segmentation
# #####################################

def RemoveLabel(filepath,outpath,labelToRemove = [1,5,6], label_radius = 4):
    print("Reading:", filepath)
    input_img = sitk.ReadImage(filepath) 
    img = sitk.GetArrayFromImage(input_img)

    range = np.max(img)-np.min(img)

    for i in labelToRemove:
        img = np.where(img == i, 0,img)

    img = np.where(img > 0, 1,img)
    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())

    output = sitk.BinaryDilate(output, [label_radius] * output.GetDimension())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output

# #####################################
#  Generate landmark from .fcsv files
# #####################################

def CorrectCSV(filePath):
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
                row[11] = row[11].replace("-1","")
                row[11] = row[11].replace(" ","")

                if "RGo_LGo" in row[11] or "RCo_LCo" in row[11] or "LCo_RCo" in row[11] or "LGo_RGo" in row[11]:
                    keep = False

            if(keep):
                writer.writerow(row)

def ReadFCSV(filePath):
    r"""
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

def GetSphereMaskCoord(h,w,l,center,rad):
    X, Y, Z = np.ogrid[:h, :w, :l]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)
    mask = dist_from_center <= rad

    return np.array(np.where(mask))

def CreateNewImage(size,origin,spacing,direction):
    image = sitk.Image(size.tolist(), sitk.sitkInt16)
    image.SetOrigin(origin.tolist())
    image.SetSpacing(spacing.tolist())
    image.SetDirection(direction.tolist())

    return image

def GenSeperateLabels(filePath,refImg,outpath,rad,label_lst):
    r"""
    Generate a label image from a fiducial file ".fcsv".
    The generated image will match with the reference image. 

    Parameters
    ----------
    filePath
     path of the .fcsv file 
    refImg
     reference image to use to generate the label image
    outpath
     path to save the generated image
    rad
     landmarks radius
    label_lst
     landmarks labeks list
     """

    print("Generating landmarks image at : ", outpath)

    ref = sitk.ReadImage(refImg)
    ref_size = np.array(ref.GetSize())
    ref_origin = np.array(ref.GetOrigin())
    ref_spacing = np.array(ref.GetSpacing())
    ref_direction = np.array(ref.GetDirection())

    image_3D = CreateNewImage(ref_size,ref_origin,ref_spacing,ref_direction)

    physical_origin = abs(ref_origin/ref_spacing)

    lm_lst = ReadFCSV(filePath)
    for lm in lm_lst :
        lm_ph_coord = np.array([float(lm["x"]),float(lm["y"]),float(lm["z"])])
        lm_ph_coord = lm_ph_coord/ref_spacing+physical_origin
        lm_coord = lm_ph_coord.astype(int)
        maskCoord = GetSphereMaskCoord(ref_size[0],ref_size[1],ref_size[2],lm_coord,rad)
        maskCoord=maskCoord.tolist()

        lm_label = label_lst.index(lm['label']) + 1
        for i in range(0,len(maskCoord[0])):
            image_3D.SetPixel([maskCoord[0][i],maskCoord[1][i],maskCoord[2][i]],lm_label)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(image_3D)

def GenerateUpLowLabels(upfilePath,lowfilePath,refImg,outpath,rad):
    r"""
    Generate a label image from a fiducial file ".fcsv".
    The generated image will match with the reference image. 

    Parameters
    ----------
    upfilePath
     path of the upper.fcsv file 
    lowfilePath
     path of the lower.fcsv file 
    refImg
     reference image to use to generate the label image
    outpath
     path to save the generated image
    rad
     landmarks radius
     """

    print("Generating landmarks image at : ", outpath)

    ref = sitk.ReadImage(refImg)
    ref_size = np.array(ref.GetSize())
    ref_origin = np.array(ref.GetOrigin())
    ref_spacing = np.array(ref.GetSpacing())
    ref_direction = np.array(ref.GetDirection())

    image_3D = CreateNewImage(ref_size,ref_origin,ref_spacing,ref_direction)

    physical_origin = abs(ref_origin/ref_spacing)

    ulm_lst = ReadFCSV(upfilePath)
    llm_lst = ReadFCSV(lowfilePath)


    # Upper landmarks
    for lm in ulm_lst :
        lm_ph_coord = np.array([float(lm["x"]),float(lm["y"]),float(lm["z"])])
        lm_ph_coord = lm_ph_coord/ref_spacing+physical_origin
        lm_coord = lm_ph_coord.astype(int)
        maskCoord = GetSphereMaskCoord(ref_size[0],ref_size[1],ref_size[2],lm_coord,rad)
        maskCoord=maskCoord.tolist()

        for i in range(0,len(maskCoord[0])):
            image_3D.SetPixel([maskCoord[0][i],maskCoord[1][i],maskCoord[2][i]],1)

    # Lower landmarks
    for lm in llm_lst :
        lm_ph_coord = np.array([float(lm["x"]),float(lm["y"]),float(lm["z"])])
        lm_ph_coord = lm_ph_coord/ref_spacing+physical_origin
        lm_coord = lm_ph_coord.astype(int)
        maskCoord = GetSphereMaskCoord(ref_size[0],ref_size[1],ref_size[2],lm_coord,rad)
        maskCoord=maskCoord.tolist()

        for i in range(0,len(maskCoord[0])):
            image_3D.SetPixel([maskCoord[0][i],maskCoord[1][i],maskCoord[2][i]],2)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(image_3D)


