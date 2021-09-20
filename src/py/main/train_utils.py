
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
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SaveImaged,
)

from monai.config import print_config
from monai.metrics import DiceMetric


from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch

# #####################################
#  Transforms
# #####################################

def createTrainTransform(wanted_spacing = [0.5,0.5,0.5],CropSize = [64,64,64],outdir="Out"):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=wanted_spacing,
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAI"),
            ScaleIntensityd(
                keys=["image"],minv = 0.0, maxv = 1.0, factor = None
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=CropSize,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transforms


def createValidationTransform(wanted_spacing = [0.5,0.5,0.5],outdir="Out"):

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=wanted_spacing,
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAI"),
            ScaleIntensityd(
                keys=["image"],minv = 0.0, maxv = 1.0, factor = None
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    return val_transforms

def createPredictTransform(wanted_spacing = [0.5,0.5,0.5],outdir="out"):

    pre_transforms = Compose(
        [
            LoadImaged(keys="image"),
            AddChanneld(keys="image"),
            Spacingd(
                keys="image",
                pixdim=wanted_spacing,
                mode="bilinear",
            ),
            Orientationd(keys="image", axcodes="RAI"),
            ScaleIntensityd(
                keys=["image"],minv = 0.0, maxv = 1.0, factor = None
            ),
            CropForegroundd(keys="image", source_key="image"),
            ToTensord(keys="image"),
            SaveImaged(
                keys="image",
                meta_keys="image_meta_dict", 
                output_dir=outdir, output_postfix="Input", 
                resample=False
            ),
        ]
    )

    return pre_transforms


def SavePrediction(data, outpath):

    save_transform = Compose(
        [
            SaveImaged(
                keys="pred",
                meta_keys="image_meta_dict", 
                output_dir=outpath, output_postfix="Pred", 
                resample=False
            ),
        ]
    )
    
    save_transform(data)

# #####################################
#  Training
# #####################################

def validation(model,cropSize, post_label, post_pred, dice_metric, global_step, epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
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





def train(data_model, cropSize, global_step, eval_num, max_iterations, train_loader, val_loader, epoch_loss_values, metric_values, dice_val_best, global_step_best, dice_metric, post_label, post_pred):
    
    model = data_model["model"]
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
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
