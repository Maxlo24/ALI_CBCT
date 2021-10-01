from model import*
from utils import*

import argparse


def main(args):

    # #####################################
    #  Init_param
    # #####################################
    label_nbr = args.nbr_label
    nbr_workers = args.nbr_worker

    spacing = args.spacing
    cropSize = args.crop_size

    datalist = GetDataList(
        dirDict = {
            "image" : args.dir,
            "landmarks" : args.dir_test,
            # "fiducial" : args.fid
        }
    )

    for scan in datalist:
        input_img = sitk.ReadImage(scan["landmarks"]) 
        img = sitk.GetArrayFromImage(input_img)
        SaveFiducialFromArray(img,scan["image"],args.out,CB_labels)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans',default=None, required=True)
    input_group.add_argument('--dir_test', type=str, help='Input directory with the landmarks',default=None, required=True)
    # input_group.add_argument('--fid', type=str, help='Input directory with the fiducial',default=None, required=True)

    # input_group.add_argument('--load_model', type=str, help='Path of the model', default=None, required=True)
    input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)

    
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[2,2,2])
    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=5)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=1)

    args = parser.parse_args()
    
    main(args)