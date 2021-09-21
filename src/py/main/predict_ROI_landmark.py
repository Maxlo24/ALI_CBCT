from model import*
from utils import*

import argparse

import logging
import os
import sys
import tempfile
import glob

import nibabel as nib
import numpy as np
import torch



def main(args):

    label_nbr = 3
    nbr_workers = 1
    spacing = args.spacing
    cropSize = args.crop_size

    scan_lst = []
    datalist = []

    scan_normpath = os.path.normpath("/".join([args.dir, '**', '']))
    for img_fn in sorted(glob.iglob(scan_normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            scan_lst.append(img_fn)

    for file_id in range(0,len(scan_lst)):
        data = {"image" : scan_lst[file_id]}
        datalist.append(data)

    # define pre transforms
    pre_transforms = createPredictTransform(wanted_spacing= args.spacing,outdir=args.out)

    print("Loading data from", args.dir)

    val_ds = CacheDataset(
        data=datalist,
        transform=pre_transforms,
        # cache_num=6, 
        cache_rate=1.0, 
        num_workers=nbr_workers
    )


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Create_UNETR(
        label_nbr=label_nbr,
        cropSize=cropSize
    ).to(device)

    print("Loading model", args.load_model)
    net.load_state_dict(torch.load(args.load_model,map_location=device))
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(val_ds):
            img = data["image"]
            val_inputs = torch.unsqueeze(img, 1)
            val_outputs = sliding_window_inference(
                inputs= val_inputs,
                roi_size = cropSize, 
                sw_batch_size= nbr_workers, 
                predictor= net, 
                overlap=0.8
            )

            data["pred"] = torch.argmax(val_outputs, dim=1).detach().cpu()
            SavePrediction(data,args.out)

    print("Done : " + str(len(val_ds)) + " scan segmented")

    # case_num = 0
    # slice_map = 40
    # with torch.no_grad():
    #     img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
    #     img = val_ds[case_num]["image"]
    #     val_inputs = torch.unsqueeze(img, 1)
    #     val_outputs = sliding_window_inference(
    #         val_inputs, cropSize, nbr_workers, net, overlap=0.8
    #     )
    #     plt.figure("check", (18, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.title("image")
    #     plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map], cmap="gray")
    #     plt.subplot(1, 2, 2)
    #     plt.title("output")
    #     plt.imshow(
    #         torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map]
    #     )
    #     plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans',default=None, required=True)
    input_group.add_argument('--load_model', type=str, help='Path of the model', default=None, required=True)
    input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)

    
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[2,2,2])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])

    args = parser.parse_args()
    
    main(args)
