
from utils import*
import argparse
import glob
import sys
import os
from shutil import copyfile


def main(args):

    scan_lst = []
    U_fcsv_lst = []
    L_fcsv_lst = []
    CB_fcsv_lst = []
    

    print("Reading folder : ", args.input_dir)
    print("Selected spacings : ", args.spacing)
    		
    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            img_obj = {}
            img_obj["img"] = img_fn
            baseName = os.path.basename(img_fn)
            if True in [scan in baseName for scan in ["scan","Scan"]]:
                scan_lst.append(img_obj)

        if os.path.isfile(img_fn) and ".fcsv" in img_fn:
            img_obj = {}
            img_obj["file"] = img_fn
            baseName = os.path.basename(img_fn)
            if "_U." in baseName :
                U_fcsv_lst.append(img_obj)
            elif "_L." in baseName :
                L_fcsv_lst.append(img_obj)
            elif "_CB." in baseName :
                CB_fcsv_lst.append(img_obj)
                
            else:
                print("----> Unrecognise fiducial file found at :", img_fn)
        elif os.path.isfile(img_fn) and "fcsv" in img_fn:
            print("----> Not correct fiducial file found at :", img_fn)

            
    # if not os.path.exists(SegOutpath):
    #     os.makedirs(SegOutpath)
    

    if len(scan_lst) != len(U_fcsv_lst) or len(scan_lst) != len(L_fcsv_lst) or len(L_fcsv_lst) != len(CB_fcsv_lst):

        print("ERROR : folder dont have the same number of scans , _U.fcsv, _L.fcsv files and  _CB.fcsv.", file=sys.stderr)
        print("Lead : make sure the fiducial files end like this : '_U.fcsv and' , '_L.fcsv' (no space or missing '.' )")
        print('       Scan number : ',len(scan_lst))
        print('       _U.fcsv number : ',len(U_fcsv_lst))
        print('       _L.fcsv number : ',len(L_fcsv_lst))
        print('       _CB.fcsv number : ',len(L_fcsv_lst))
        
        raise 

    for n in range(0,len(scan_lst)):

        scan = scan_lst[n]

        # print(scan_basename)
        U_lm = U_fcsv_lst[n]
        L_lm = L_fcsv_lst[n]
        CB_lm = CB_fcsv_lst[n]
        
        CorrectCSV(U_lm["file"])
        CorrectCSV(L_lm["file"])
        CorrectCSV(CB_lm["file"])


        scan_dirname = os.path.dirname(scan["img"])
        scan_basename = os.path.basename(scan["img"])
        scan_name= scan_basename.split(".")

        ScanOutpath = os.path.normpath("/".join([args.out,os.path.basename(scan_dirname)]))

        if not os.path.exists(ScanOutpath):
            os.makedirs(ScanOutpath)

        for sp in args.spacing:
            new_name = ""

            for i,element in enumerate(scan_name):
                if i == 0:
                    new_name += element + "_" + str(sp)
                else:
                    new_name += "." + element
            
            SetSpacing(scan["img"],[sp,sp,sp],os.path.join(ScanOutpath,new_name))

        copyfile(U_lm["file"],os.path.join(ScanOutpath,os.path.basename(U_lm["file"])))
        copyfile(L_lm["file"],os.path.join(ScanOutpath,os.path.basename(L_lm["file"])))
        copyfile(CB_lm["file"],os.path.join(ScanOutpath,os.path.basename(CB_lm["file"])))



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', required=True)

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[2,1,0.3])

    args = parser.parse_args()
    
    main(args)
