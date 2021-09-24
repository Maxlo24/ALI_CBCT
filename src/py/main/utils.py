
import os
import shutil
import tempfile
import datetime

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    RandCropByLabelClassesd
)

from monai.config import print_config
from monai.metrics import DiceMetric

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

from sklearn.model_selection import train_test_split

import torch
import glob

# #####################################
#  Setup Training
# #####################################

def setupTrain(dirDict,test_percentage,dir_model):
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
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
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

    trainingSet, validationSet = train_test_split(datalist, test_size=test_percentage/100, random_state=len(datalist))  

    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory

    print("WORKING IN : ", root_dir)

    return trainingSet, validationSet, root_dir

# #####################################
#  Transforms
# #####################################

def createROITrainTransform(wanted_spacing = [2,2,2],CropSize = [64,64,64],outdir="Out"):

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
        ]
    )

    return train_transforms

def createALITrainTransform(wanted_spacing = [0.5,0.5,0.5],CropSize = [64,64,64],outdir="Out"):

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
        ]
    )

    return train_transforms

def createValidationTransform(wanted_spacing = [0.5,0.5,0.5],outdir="Out"):

    val_transforms = Compose(
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
            ToTensord(keys=["image", "landmarks"]),
        ]
    )

    return val_transforms

def createPredictTransform(data):

    pre_transforms = Compose(
        [AddChannel(),ScaleIntensity(minv = 0.0, maxv = 1.0, factor = None)]
    )

    input_img = sitk.ReadImage(data) 
    img = sitk.GetArrayFromImage(input_img)
    pre_img = torch.from_numpy(pre_transforms(img))
    pre_img = pre_img.type(torch.DoubleTensor)
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

# #####################################
#  Training
# #####################################

def validation(inID, outID,model,cropSize, post_label, post_pred, dice_metric, global_step, epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch[inID].cuda(), batch[outID].cuda())
            val_outputs = sliding_window_inference(val_inputs, cropSize, 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)

    return mean_dice_val

 
def train(inID, outID, data_model, cropSize, global_step, eval_num, max_iterations, train_loader, 
        val_loader, epoch_loss_values, metric_values, dice_val_best, global_step_best, dice_metric, post_label, post_pred):
    
    model = data_model["model"]
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch[inID].cuda(), batch[outID].cuda())
        logit_map = model(x)
        loss = data_model["loss_f"](logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        data_model["optimizer"].step()
        data_model["optimizer"].zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
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
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                save_path = os.path.join(data_model["dir"],data_model["name"]+"_"+datetime.datetime.now().strftime("%Y_%d_%m")+"_E_"+str(global_step)+".pth")
                torch.save(
                    model.state_dict(), save_path
                )
                data_model["best"] = save_path
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
        data_model["model"] = model

    return global_step, dice_val_best, global_step_best