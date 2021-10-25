
from utils import*
import argparse
import glob
import sys
import os
from shutil import copyfile


def main(args):

    print("Reading folder : ", args.input_dir)
    print("Selected spacings : ", args.spacing)

    patients = {}
    		
    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            file_name = basename.split(".")[0]
            elements = file_name.split("_")
            patient = elements[0] + "_" + elements[1]
            if patient not in patients.keys():
                patients[patient] = {"dir": os.path.dirname(img_fn)}

            if True in [scan in basename for scan in ["scan","Scan"]]:
                patients[patient]["scan"] = img_fn

        if True in [scan in img_fn for scan in [".fcsv",".mrk.json"]]:
            file_name = basename.split(".")[0]
            elements = file_name.split("_")
            patient = elements[0] + "_" + elements[1]
            if patient not in patients.keys():
                patients[patient] = {"dir": os.path.dirname(img_fn)}

            if True in [char in basename for char in ["_U.", "_U_","_max_","_Max_"]] :
                patients[patient]["U"] = img_fn
            elif True in [char in basename for char in ["_L.", "_L_","_mand_","_Mand_"]] :
                patients[patient]["L"] = img_fn
            elif True in [char in basename for char in ["_CB.", "_CB_"]] :
                patients[patient]["CB"] = img_fn
            else:
                print("----> Unrecognise fiducial file found at :", img_fn)
        elif os.path.isfile(img_fn) and "fcsv" in img_fn:
            print("----> Unrecognise file found at :", img_fn)

            
    # if not os.path.exists(SegOutpath):
    #     os.makedirs(SegOutpath)
    
    error = False
    for patient,data in patients.items():
        if "scan" not in data.keys():
            print("Missing scan for patient :",patient,"at",data["dir"])
            error = True
        if "U" not in data.keys():
            print("Missing U landmark for patient :",patient,"at",data["dir"])
            error = True
        if "L" not in data.keys():
            print("Missing L landmark for patient :",patient,"at",data["dir"])
            error = True
        if "CB" not in data.keys():
            print("Missing CB landmark for patient :",patient,"at",data["dir"])
            error = True

    if error:
        print("ERROR : folder have missing files", file=sys.stderr)
        raise

    # if len(scan_lst) != len(U_fcsv_lst) or len(scan_lst) != len(L_fcsv_lst) or len(L_fcsv_lst) != len(CB_fcsv_lst):

    #     print("ERROR : folder dont have the same number of scans , _U.fcsv, _L.fcsv files and  _CB.fcsv.", file=sys.stderr)
    #     print("Lead : make sure the fiducial files end like this : '_U.fcsv and' , '_L.fcsv' (no space or missing '.' )")
    #     print('       Scan number : ',len(scan_lst))
    #     print('       _U.fcsv number : ',len(U_fcsv_lst))
    #     print('       _L.fcsv number : ',len(L_fcsv_lst))
    #     print('       _CB.fcsv number : ',len(L_fcsv_lst))
        
    #     raise 

    # for patient,data in patients.items():

    #     scan = data["scan"]
    #     U_lm = data["U"]
    #     L_lm = data["L"]
    #     CB_lm = data["CB"]
        
    #     CorrectCSV(U_lm)
    #     CorrectCSV(L_lm)
    #     CorrectCSV(CB_lm)


    #     patient_dirname = os.path.basename(data["dir"]).split(" ")[0]
    #     ScanOutpath = os.path.normpath("/".join([args.out,patient_dirname]))

    #     if not os.path.exists(ScanOutpath):
    #         os.makedirs(ScanOutpath)


    #     scan_basename = os.path.basename(scan)
    #     scan_name = scan_basename.split(".")

    #     for sp in args.spacing:
    #         new_name = ""

    #         for i,element in enumerate(scan_name):
    #             if i == 0:
    #                 new_name = patient + "_scan_" + str(sp)
    #             else:
    #                 new_name += "." + element
            
    #         SetSpacing(scan,[sp,sp,sp],os.path.join(ScanOutpath,new_name))

    #     if ".fcsv" in U_lm:
    #         SaveJsonFromFcsv(U_lm,os.path.join(ScanOutpath,patient + "_lm_U.mrk.json"))

    #     copyfile(U_lm,os.path.join(ScanOutpath,patient + "_lm_U.fcsv"))
    #     copyfile(L_lm,os.path.join(ScanOutpath,patient + "_lm_L.fcsv"))
    #     copyfile(CB_lm,os.path.join(ScanOutpath,patient + "_lm_CB.fcsv"))



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', required=True)

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[2,0.3])

    args = parser.parse_args()
    
    main(args)
