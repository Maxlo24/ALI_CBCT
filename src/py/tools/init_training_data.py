from data_utils import*
import argparse
import glob
import sys

def main(args):

    scan_lst = []
    seg_lst = []
    U_fcsv_lst = []
    L_fcsv_lst = []
    CB_fcsv_lst = []
    
    SegOutpath = os.path.normpath("/".join([args.out,"Seg"]))
    ScanOutpath = os.path.normpath("/".join([args.out,"Scan_"+str(args.spacing[0])]))
    U_LMOutpath = os.path.normpath("/".join([args.out,"U_Landmark"]))
    L_LMOutpath = os.path.normpath("/".join([args.out,"L_Landmark"]))
    CB_LMOutpath = os.path.normpath("/".join([args.out,"CB_Landmark"]))

    Mixed_LMOutpath = os.path.normpath("/".join([args.out,"Mixed_Landmarks"]))

    print("Reading folder : ", args.input_dir)
    		
    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            img_obj = {}
            img_obj["img"] = img_fn
            baseName = os.path.basename(img_fn)
            if True in [seg in baseName for seg in ["seg","Seg"]]:
                img_obj["out"] = os.path.normpath("/".join([SegOutpath,baseName]))
                seg_lst.append(img_obj)

            else:
                img_obj["out"] = os.path.normpath("/".join([ScanOutpath,baseName]))
                scan_lst.append(img_obj)

        if os.path.isfile(img_fn) and ".fcsv" in img_fn:
            img_obj = {}
            img_obj["file"] = img_fn
            baseName = os.path.basename(img_fn)
            if "_U." in baseName :
                img_obj["out"] = U_LMOutpath
                U_fcsv_lst.append(img_obj)
                # print(img_obj, 1)
            elif "_L." in baseName :
                img_obj["out"] = L_LMOutpath
                L_fcsv_lst.append(img_obj)
                # print(img_obj, 2)
            elif "_CB." in baseName :
                img_obj["out"] = CB_LMOutpath
                CB_fcsv_lst.append(img_obj)
                # print(img_obj, 2)
                
            else:
                print("----> Unrecognise fiducial file found at :", img_fn)
        elif os.path.isfile(img_fn) and "fcsv" in img_fn:
            print("----> Not correct fiducial file found at :", img_fn)

            
    # if not os.path.exists(SegOutpath):
    #     os.makedirs(SegOutpath)

    if not os.path.exists(ScanOutpath):
        os.makedirs(ScanOutpath)

    if not os.path.exists(U_LMOutpath) and "u" in args.seperate_landmark:
        os.makedirs(U_LMOutpath)
    
    if not os.path.exists(L_LMOutpath) and "l" in args.seperate_landmark:
        os.makedirs(L_LMOutpath)

    if not os.path.exists(CB_LMOutpath) and "cb" in args.seperate_landmark:
        os.makedirs(CB_LMOutpath)

    if not os.path.exists(Mixed_LMOutpath) and args.mixed_landmark:
        os.makedirs(Mixed_LMOutpath)

    

    if len(scan_lst) != len(U_fcsv_lst) or len(scan_lst) != len(L_fcsv_lst) or len(seg_lst) != len(L_fcsv_lst) or len(L_fcsv_lst) != len(CB_fcsv_lst):

        print("ERROR : folder dont have the same number of scans , _U.fcsv and _L.fcsv files.", file=sys.stderr)
        print("Lead : make sure the fiducial files end like this : '_U.fcsv and' , '_L.fcsv' (no space or missing '.' )")
        print('       Scan number : ',len(scan_lst))
        print('       Seg number : ',len(seg_lst))
        print('       _U.fcsv number : ',len(U_fcsv_lst))
        print('       _L.fcsv number : ',len(L_fcsv_lst))
        
        raise 

    for n in range(0,len(scan_lst)):

        scan = scan_lst[n]

        scan_basename = os.path.basename(scan["img"])
        seg = seg_lst[n]
        U_lm = U_fcsv_lst[n]
        L_lm = L_fcsv_lst[n]
        CB_lm = CB_fcsv_lst[n]

        
        CorrectCSV(U_lm["file"])
        CorrectCSV(L_lm["file"])
        CorrectCSV(CB_lm["file"])

        SetSpacing(scan["img"],args.spacing,scan["out"])

        if "u" in args.seperate_landmark:
            GenSeperateLabels(U_lm["file"],scan["out"],os.path.normpath("/".join([U_lm["out"],"U_LM_" + scan_basename])),args.label_radius,U_labels)
        
        if "l" in args.seperate_landmark:
            GenSeperateLabels(L_lm["file"],scan["out"],os.path.normpath("/".join([L_lm["out"],"L_LM_" + scan_basename])),args.label_radius,L_labels)
        
        if "cb" in args.seperate_landmark:
            GenSeperateLabels(CB_lm["file"],scan["out"],os.path.normpath("/".join([CB_lm["out"],"CB_LM_" + scan_basename])),args.label_radius,CB_labels)

        if args.mixed_landmark:
            GenerateUpLowLabels(U_lm["file"],L_lm["file"],scan["out"],os.path.normpath("/".join([Mixed_LMOutpath,"M_LM_" + scan_basename])),args.label_radius)
        
        # RemoveLabel(seg["img"], seg["out"], args.label_to_remove)
        # # SetSpacing(seg["out"],args.spacing,seg["out"])
        # SetSpacingFromRef(seg["out"],scan["out"],seg["out"])

        

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='MD_reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', required=True)

    input_params = input_group.add_mutually_exclusive_group(required=True)
    input_params.add_argument('-slm','--seperate_landmark',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: u l cb)", default=[])
    # input_params.add_argument('-llm','--lower_landmark',action="store_true",help="Organise the data for uper landmark training")
    input_params.add_argument('-mlm','--mixed_landmark',action="store_true",help="Prepare the data for the low resolution uper and lower landmark training")

    if parser.parse_args().seperate_landmark:
        spacing = [0.5,0.5,0.5]
        radius = 2
    elif parser.parse_args().mixed_landmark:
        spacing = [2,2,2]
        radius = 2

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=spacing)
    input_group.add_argument('-lrem', '--label_to_remove', nargs="+", type=int, help='Label ID to remove', default=[1,5,6])
    input_group.add_argument('-lrad', '--label_radius', type=int, help='Label ID to remove', default=radius)


    args = parser.parse_args()
    
    main(args)
