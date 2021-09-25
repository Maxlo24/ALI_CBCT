from utils import*
import argparse
import glob
import sys

def main(args):

    lab_scan_lst = []

    U_ROIOutpath = os.path.normpath("/".join([args.out,"U_ROI"]))
    L_ROIOutpath = os.path.normpath("/".join([args.out,"L_ROI"]))
    CB_ROIOutpath = os.path.normpath("/".join([args.out,"CB_ROI"]))

    print("Reading folder : ", args.input_dir)
    		
    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            baseName = os.path.basename(img_fn)
            if "or" in baseName :
                lab_scan_lst.append(img_fn)

    # print(lab_scan_lst)
    if not os.path.exists(U_ROIOutpath) and "u" in args.select_region:
        os.makedirs(U_ROIOutpath)
    
    if not os.path.exists(L_ROIOutpath) and "l" in args.select_region:
        os.makedirs(L_ROIOutpath)

    if not os.path.exists(CB_ROIOutpath) and "cb" in args.select_region:
        os.makedirs(CB_ROIOutpath)

    for lm_scan in lab_scan_lst:
        baseName = os.path.basename(lm_scan)
        baseName = baseName.split("or")[0]

        if "u" in args.select_region:
            GenerateROIfile(lm_scan,os.path.join(U_ROIOutpath,baseName+"U_ROI.nii.gz"),[1,4],args.box_dist)

        if "l" in args.select_region:
            GenerateROIfile(lm_scan,os.path.join(L_ROIOutpath,baseName+"L_ROI.nii.gz"),[2,4],args.box_dist)

        if "cb" in args.select_region:
            GenerateROIfile(lm_scan,os.path.join(CB_ROIOutpath,baseName+"CB_ROI.nii.gz"),[3],args.box_dist)
    
    # os.path.normpath("/".join([L_ROIOutpath,baseName]))

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', required=True)

    input_group.add_argument('-reg','--select_region',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: u l cb)", default=[])

    input_group.add_argument('-bd', '--box_dist', type=int, help='distance between 2 ROI', default=5)
    
    args = parser.parse_args()
    
    main(args)
