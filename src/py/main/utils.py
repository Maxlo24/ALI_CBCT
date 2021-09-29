import os
import itk
import tempfile
import datetime
from itk.support.types import Offset
import numpy as np
import SimpleITK as sitk
import csv
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import glob
import shutil
import sys

# ----- MONAI ------

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    AddChannel,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityd,
    ScaleIntensity,
    Spacingd,
    Spacing,
    RandRotate90d,
    ToTensord,
    ToTensor,
    SaveImaged,
    SaveImage,
    RandCropByLabelClassesd,
    Lambdad,
    CastToTyped,
    SpatialCrop
)

from monai.config import print_config
from monai.metrics import DiceMetric

from monai.data import (
    DataLoader,
    CacheDataset,
    SmartCacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

# #####################################
#  Global variables
# #####################################

U_labels = ['PNS','ANS','A','UR6apex','UR3apex','U1apex','UL3apex','UL6apex','UR6d','UR6m','UR3tip','UItip','UL3tip','UL6m','UL6d']
L_labels = ['RCo','RGo','LR6apex','LR7apex','L1apex','Me','Gn','Pog','B','LL6apex','LL7apex','LGo','LCo','LR6d','LR6m','LItip','LL6m','LL6d']
CB_labels = ['Ba','S','N']

data_type = torch.float32

######## ########     ###    ##    ##  ######  ########  #######  ########  ##     ##  ######  
   ##    ##     ##   ## ##   ###   ## ##    ## ##       ##     ## ##     ## ###   ### ##    ## 
   ##    ##     ##  ##   ##  ####  ## ##       ##       ##     ## ##     ## #### #### ##       
   ##    ########  ##     ## ## ## ##  ######  ######   ##     ## ########  ## ### ##  ######  
   ##    ##   ##   ######### ##  ####       ## ##       ##     ## ##   ##   ##     ##       ## 
   ##    ##    ##  ##     ## ##   ### ##    ## ##       ##     ## ##    ##  ##     ## ##    ## 
   ##    ##     ## ##     ## ##    ##  ######  ##        #######  ##     ## ##     ##  ######  

def CreateROITrainTransform(wanted_spacing = [2,2,2],CropSize = [64,64,64],outdir="Out"):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "landmarks"]),
            AddChanneld(keys=["image", "landmarks"]),
            Spacingd(
                keys=["image", "landmarks"],
                pixdim=wanted_spacing,
                mode=("bilinear", "nearest"),
            ),
            # Orientationd(keys=["image", "landmarks"], axcodes="RAI"),
            ScaleIntensityd(
                keys=["image"],minv = 0.0, maxv = 1.0, factor = None
            ),
            # CropForegroundd(keys=["image", "landmarks"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "landmarks"],
                label_key="landmarks",
                spatial_size=CropSize,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "landmarks"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "landmarks"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "landmarks"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "landmarks"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "landmarks"]),
            CastToTyped(keys=["image", "landmarks"], dtype=[data_type,torch.int16])

        ]
    )

    return train_transforms

def CreateALITrainTransformWithROIScan(wanted_spacing = [0.5,0.5,0.5],CropSize = [64,64,64],outdir="Out"):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "landmarks","label"]),
            AddChanneld(keys=["image", "landmarks","label"]),
            Spacingd(
                keys=["image", "landmarks", "label"],
                pixdim=wanted_spacing,
                mode=("bilinear", "nearest", "nearest"),
            ),
            # Orientationd(keys=["image", "landmarks", "label"], axcodes="RAI"),
            ScaleIntensityd(
                keys=["image"],minv = 0.0, maxv = 1.0, factor = None
            ),
            CropForegroundd(keys=["image", "landmarks", "label"], source_key="image"),
            RandCropByLabelClassesd(
                keys=["image", "landmarks"],
                label_key="label",
                spatial_size=CropSize,
                ratios=[1,1],
                num_classes=2,
                num_samples=4,
            ),
            RandFlipd(
                keys=["image", "landmarks"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "landmarks"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "landmarks"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "landmarks"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "landmarks","label"]),
            CastToTyped(keys=["image", "landmarks"], dtype=[data_type,torch.int16])
        ]
    )

    return train_transforms

def test(x,cropSize):
    print(x,cropSize)

def CreateALITrainTransform():

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "landmarks"]),
            AddChanneld(keys=["image", "landmarks"]),
            # Spacingd(
            #     keys=["image", "landmarks"],
            #     pixdim=wanted_spacing,
            #     mode=("bilinear", "nearest"),
            # ),
            # Orientationd(keys=["image", "landmarks", "label"], axcodes="RAI"),
            ScaleIntensityd(
                keys=["image"],minv = 0.0, maxv = 1.0, factor = None
            ),
            # CropForegroundd(keys=["image", "landmarks"], source_key="image"),
            # Lambdad(
            #     keys=["image", "landmarks"],
            #     func = test,
            #     overwrite=False
            # ),
            RandFlipd(
                keys=["image", "landmarks"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "landmarks"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "landmarks"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "landmarks"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "landmarks"]),
        ]
    )

    return train_transforms

def CreateValidationTransform(wanted_spacing = [0.5,0.5,0.5],outdir="Out"):

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "landmarks"]),
            AddChanneld(keys=["image", "landmarks"]),
            # Spacingd(
            #     keys=["image", "landmarks"],
            #     pixdim=wanted_spacing,
            #     mode=("bilinear", "nearest"),
            # ),
            # Orientationd(keys=["image", "landmarks"], axcodes="RAI"),
            ScaleIntensityd(
                keys=["image"],minv = 0.0, maxv = 1.0, factor = None
            ),
            # CropForegroundd(keys=["image", "landmarks"], source_key="image"),
            ToTensord(keys=["image", "landmarks"]),
            # CastToTyped(keys=["image", "landmarks"], dtype=[data_type,torch.int16])
        ]
    )

    return val_transforms

def CreatePredictTransform(data):

    pre_transforms = Compose(
        [AddChannel(),ScaleIntensity(minv = 0.0, maxv = 1.0, factor = None)]
    )

    input_img = sitk.ReadImage(data) 
    img = sitk.GetArrayFromImage(input_img)
    pre_img = torch.from_numpy(pre_transforms(img))
    pre_img = pre_img.type(data_type)
    return pre_img,input_img

def SavePrediction(data,input_img, outpath):

    print("Saving prediction to : ", outpath)

    # print(data)

    img = data.numpy()[0][:]
    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)

######## ########     ###    #### ##    ## #### ##    ##  ######   
   ##    ##     ##   ## ##    ##  ###   ##  ##  ###   ## ##    ##  
   ##    ##     ##  ##   ##   ##  ####  ##  ##  ####  ## ##        
   ##    ########  ##     ##  ##  ## ## ##  ##  ## ## ## ##   #### 
   ##    ##   ##   #########  ##  ##  ####  ##  ##  #### ##    ##  
   ##    ##    ##  ##     ##  ##  ##   ###  ##  ##   ### ##    ##  
   ##    ##     ## ##     ## #### ##    ## #### ##    ##  ######   
 
def train(inID, outID, data_model, global_step, epoch_loss_values, max_iterations, train_loader, ):
    
    model = data_model["model"]
    model.train()
    epoch_loss = 0
    steps = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        steps += 1
        x, y = (batch[inID].cuda(), batch[outID].cuda())
        logit_map = model(x)
        loss = data_model["loss_f"](logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        data_model["optimizer"].step()
        data_model["optimizer"].zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step+steps, max_iterations, loss)
        )
        data_model["model"] = model

    epoch_loss /= steps
    epoch_loss_values.append(epoch_loss)

    return steps

def validation(inID, outID,model,cropSize, post_label, post_pred, dice_metric, global_step, epoch_iterator_val):
    model.eval()
    dice_vals = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch[inID].to(device), batch[outID].to(device))
            val_outputs = sliding_window_inference(val_inputs, cropSize, 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice = IoU_Dice(y_true_lst=val_output_convert, y_pred_lst=val_labels_convert)
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)

    return mean_dice_val


def validate(inID, outID,data_model,val_loader, cropSize, global_step, metric_values, dice_val_best, global_step_best, dice_metric, post_label, post_pred):
    model = data_model["model"]
    epoch_loss = 0
    step = 0

    epoch_iterator_val = tqdm(
        val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
    )
    dice_val = validation(
        inID=inID,
        outID = outID,
        model=model,
        cropSize=cropSize,
        global_step=global_step,
        epoch_iterator_val=epoch_iterator_val,
        dice_metric=dice_metric,
        post_label=post_label,
        post_pred=post_pred
    )
    metric_values.append(dice_val)
    if dice_val > dice_val_best:
        dice_val_best = dice_val
        global_step_best = global_step
        save_path = os.path.join(data_model["dir"],data_model["name"]+"_"+datetime.datetime.now().strftime("%Y_%d_%m")+"_E_"+str(global_step)+".pth")
        torch.save(
            model.state_dict(), save_path
        )
        data_model["best"] = save_path
        print("Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
    else:
        print("Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
    
    # global_step += 1
    data_model["model"] = model

    return dice_val_best, global_step_best

##     ## ######## ######## ########  ####  ######  
###   ### ##          ##    ##     ##  ##  ##    ## 
#### #### ##          ##    ##     ##  ##  ##       
## ### ## ######      ##    ########   ##  ##       
##     ## ##          ##    ##   ##    ##  ##       
##     ## ##          ##    ##    ##   ##  ##    ## 
##     ## ########    ##    ##     ## ####  ######  


def IoU_Dice(y_true_lst, y_pred_lst):

    for i in range(len(y_true_lst)):

        y_true = y_true_lst[i]
        y_pred = y_pred_lst[i]
        print(y_pred.size())

        num_classes = y_true.size()[0]
        
        y_true = torch.reshape(y_true, (num_classes,-1))
        y_pred = torch.reshape(y_pred, (num_classes,-1))

        print(y_pred)


        print(y_pred.size())

        intersection = 2.0*torch.sum(y_true * y_pred, dim=0) + 1.
        print(intersection.size())
        print(intersection)

        union = torch.sum(y_true,dim=0) + torch.sum(y_pred,dim=0) + 1.
        print(union)

        iou = 1.0 - intersection / union
        print(iou)
        print(torch.sum(iou))

        return torch.sum(iou)


########  #######   #######  ##        ######  
   ##    ##     ## ##     ## ##       ##    ## 
   ##    ##     ## ##     ## ##       ##       
   ##    ##     ## ##     ## ##        ######  
   ##    ##     ## ##     ## ##             ## 
   ##    ##     ## ##     ## ##       ##    ## 
   ##     #######   #######  ########  ######  


# #####################################
#  Setup Training
# #####################################

def GetDataList(dirDict):
    """
    Go through each dirPath directory to generate a list of dictionary 
    each dictionary contain the key-n : filepath of the sorted key-n : dirPath directorys

    Parameters
    ----------
    dirDict
     dictionary of key : dirPath 
    """
    scan_lst = []
    label_lst = []
    datalist = []

    listDict = {}

    for key,dirPath in dirDict.items():
        listDict[key] = []
        nbr_of_file = 0
        scan_normpath = os.path.normpath("/".join([dirPath, '**', '']))
        for img_fn in sorted(glob.iglob(scan_normpath, recursive=True)):
            #  print(img_fn)
            if os.path.isfile(img_fn): #and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                listDict[key].append(img_fn)
                nbr_of_file +=1

    
    # if len(scan_lst) != len(label_lst):
    #     print("ERROR : Not the same number of file in the different folders")
    #     return

    for file_id in range(0,nbr_of_file):
        data = {}
        for key, value in listDict.items():
            data[key] = value[file_id]
        datalist.append(data)

    return datalist


def CropImageFromROIfile(image,seg,roi_file,outpath,cropSize,radius=4):
    """
    Crop part of the image and the seg file based on the regions of interest (ROI) center 

    Parameters
    ----------
    image
     path of the image file 
    seg
     path of the seg file
    roi_file
     path of the ROI file
    outpath
     path to save the new images
    cropSize
     size of the crop
    radius
     minimum distance (pixels) between 2 ROI 
    """

    print("Cropping :", os.path.basename(image), os.path.basename(seg))

    transform = AddChannel()

    input_img = sitk.ReadImage(image)
    img_ar = sitk.GetArrayFromImage(input_img)
    img_tens = transform(torch.from_numpy(img_ar))

    input_seg = sitk.ReadImage(seg)
    seg_ar = sitk.GetArrayFromImage(input_seg)
    seg_tens = transform(torch.from_numpy(seg_ar))

    ROI_img = sitk.ReadImage(roi_file) 
    ROI_ar = sitk.GetArrayFromImage(ROI_img)
    # print(np.shape(ROI_ar))


    mask = ROI_ar == 1
            
    label_pos = np.array(np.where(mask),dtype='int')
    label_pos = label_pos.tolist()

    offset = np.array([2,1,1])

    ROI_coord = [np.array([label_pos[0][0],label_pos[1][0],label_pos[2][0]]) + offset]
    for i in range(1,len(label_pos[0])):
        ci = np.array([label_pos[0][i],label_pos[1][i],label_pos[2][i]]) + offset
        dist_list = np.array([np.linalg.norm(cn-ci) for cn in ROI_coord])
        if all([dist > radius for dist in dist_list]):
            ROI_coord.append(ci)

    # print(ROI_coord)

    crop_centers = []
    factor = np.array(ROI_img.GetSpacing()) / np.array(input_img.GetSpacing())
    # print(factor)
    for coord in ROI_coord:
        crop_centers.append(coord*factor)

    print(len(crop_centers), "ROI found")
    # print(crop_centers)

    scan_name = os.path.basename(image).split('.')
    ext = ""
    for e in scan_name[1:]:
       ext += "." + e 
    scan_outpath = os.path.normpath("/".join([outpath, 'Scans', scan_name[0]]))
    if not os.path.exists(scan_outpath):
        os.makedirs(scan_outpath)

    seg_name = os.path.basename(seg).split('.')
    seg_outpath = os.path.normpath("/".join([outpath, 'Segs', seg_name[0]]))
    if not os.path.exists(seg_outpath):
        os.makedirs(seg_outpath)


    for i,center_coord in enumerate(crop_centers):
        cropTransform = SpatialCrop(center_coord,cropSize)
        cr_img = cropTransform(img_tens)
        cr_seg = cropTransform(seg_tens)
        # print(img.size())

        out_img = cr_img.numpy()[0][:]
        output_img = sitk.GetImageFromArray(out_img)
        output_img.SetSpacing(input_img.GetSpacing())
        output_img.SetDirection(input_img.GetDirection())
        output_img.SetOrigin(input_img.GetOrigin())

        out_seg = cr_seg.numpy()[0][:]
        output_seg = sitk.GetImageFromArray(out_seg)
        output_seg.SetSpacing(input_seg.GetSpacing())
        output_seg.SetDirection(input_seg.GetDirection())
        output_seg.SetOrigin(input_seg.GetOrigin())

        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(scan_outpath,scan_name[0] + "_" +str(i) + ext))
        writer.Execute(output_img)

        writer.SetFileName(os.path.join(seg_outpath,seg_name[0] + "_" +str(i) + ext))
        writer.Execute(output_seg)



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


def SetSpacingFromRef(file,refFile,interpolator = "NearestNeighbor",outpath=-1):
    """
    Set the spacing of the image the same as the reference image 

    Parameters
    ----------
    filePath
     path of the image file 
    refFile
     path of the reference image 
    interpolator
     Type of interpolation 'NearestNeighbor' or 'Linear'
    outpath
     path to save the new image
    """

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

        if interpolator == "NearestNeighbor":
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        elif interpolator == "Linear":
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


# #####################################
#  Keep the landmark of the segmentation
# #####################################

def RemoveLabel(filepath,outpath,labelToRemove = [1,5,6], label_radius = 4):
    """
    Remove the unwanted labels from a file and make the other one bigger  

    Parameters
    ----------
    filePath
     path of the image file 
    labelToRemove
     list of the labels to remove from the image 
    label_radius
     radius of the dilatation to apply to the remaining labels
    outpath
     path to save the new image
    """


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

def GetSphereMaskCoord(h,w,l,center,rad):
    X, Y, Z = np.ogrid[:h, :w, :l]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)
    mask = dist_from_center <= rad

    return np.array(np.where(mask))

def GetImageInfo(filepath):
    ref = sitk.ReadImage(filepath)
    ref_size = np.array(ref.GetSize())
    ref_spacing = np.array(ref.GetSpacing())
    ref_origin = np.array(ref.GetOrigin())
    ref_direction = np.array(ref.GetDirection())

    return ref_size,ref_spacing,ref_origin,ref_direction

def GetSegLabelNbr(filepath):
    input_seg = sitk.ReadImage(filepath)
    seg_ar = sitk.GetArrayFromImage(input_seg)
    return np.max(seg_ar)-np.min(seg_ar) + 1

def CreateNewImage(size,origin,spacing,direction):
    image = sitk.Image(size.tolist(), sitk.sitkInt16)
    image.SetOrigin(origin.tolist())
    image.SetSpacing(spacing.tolist())
    image.SetDirection(direction.tolist())

    return image

def GenSeperateLabels(filePath,refImg,outpath,rad,label_lst):
    """
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

    ref_size,ref_spacing,ref_origin,ref_direction = GetImageInfo(refImg)

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

def GenerateUpLowCBLabels(upfilePath,lowfilePath,cbPath,refImg,outpath,rad):
    """
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

    ref_size,ref_spacing,ref_origin,ref_direction = GetImageInfo(refImg)

    image_3D = CreateNewImage(ref_size,ref_origin,ref_spacing,ref_direction)

    physical_origin = abs(ref_origin/ref_spacing)

    ulm_lst = ReadFCSV(upfilePath)
    llm_lst = ReadFCSV(lowfilePath)
    cblm_lst = ReadFCSV(cbPath)



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
            p_val = image_3D.GetPixel([maskCoord[0][i],maskCoord[1][i],maskCoord[2][i]])
            if p_val == 0:
                image_3D.SetPixel([maskCoord[0][i],maskCoord[1][i],maskCoord[2][i]],2)
            elif p_val == 1:
                image_3D.SetPixel([maskCoord[0][i],maskCoord[1][i],maskCoord[2][i]],4)

    # CB landmarks
    for lm in cblm_lst :
        lm_ph_coord = np.array([float(lm["x"]),float(lm["y"]),float(lm["z"])])
        lm_ph_coord = lm_ph_coord/ref_spacing+physical_origin
        lm_coord = lm_ph_coord.astype(int)
        maskCoord = GetSphereMaskCoord(ref_size[0],ref_size[1],ref_size[2],lm_coord,rad)
        maskCoord=maskCoord.tolist()

        for i in range(0,len(maskCoord[0])):
            image_3D.SetPixel([maskCoord[0][i],maskCoord[1][i],maskCoord[2][i]],3)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(image_3D)



def GenerateROIfile(lab_scan,outpath,labels = [1]):
    """
    Generate a file with the physical coordonate of the center of the ROI in a ".xlsx" file .
    It only go through the selected labels and sperate the RAI by the radius value. 

    Parameters
    ----------
    lab_scan
     file with labels
    outpath
     path to save the generated .xlsx file
    labels
     list of labels to go through
    radius
     minimum space between 2 ROI
     """

    print("Generating ROI file at : ", outpath)

    ref_size,ref_spacing,ref_origin,ref_direction = GetImageInfo(lab_scan)
    physical_origin = abs(ref_origin/ref_spacing)

    # print("Reading:", filepath)
    input_img = sitk.ReadImage(lab_scan) 
    img = sitk.GetArrayFromImage(input_img)

    for lab in labels:
        img = np.where(img==lab, -1,img)

    img = np.where(img!=-1, 0,img)
    img = np.where(img==-1, 1,img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())

    # output = sitk.BinaryDilate(output, [label_radius] * output.GetDimension())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)


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
