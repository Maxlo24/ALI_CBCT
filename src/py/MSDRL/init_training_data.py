
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

    for patient,data in patients.items():

        scan = data["scan"]

        patient_dirname = os.path.basename(data["dir"]).split(" ")[0]
        ScanOutpath = os.path.normpath("/".join([args.out,patient_dirname]))

        if not os.path.exists(ScanOutpath):
            os.makedirs(ScanOutpath)

        scan_basename = os.path.basename(scan)
        scan_name = scan_basename.split(".")

        for sp in args.spacing:
            new_name = ""

            for i,element in enumerate(scan_name):
                if i == 0:
                    new_name = patient + "_scan_sp" + str(sp).replace(".","-")
                else:
                    new_name += "." + element
            
            SetSpacing(scan,[sp,sp,sp],os.path.join(ScanOutpath,new_name))

        for lm in ["U","L","CB"]:
            if ".fcsv" in data[lm]:
                CorrectCSV(data[lm])
                SaveJsonFromFcsv(data[lm],os.path.join(ScanOutpath,patient + "_lm_"+lm+".mrk.json"))
            else:
                copyfile(data[lm],os.path.join(ScanOutpath,patient + "_lm_"+lm+".mrk.json"))




if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', required=True)

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[2,1,0.3])

    args = parser.parse_args()
    
    main(args)
