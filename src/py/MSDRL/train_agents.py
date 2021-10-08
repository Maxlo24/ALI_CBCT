

from sklearn.model_selection import train_test_split

from classes import *
from utils import *

import glob
import argparse
import random

import logging
import sys

def main(args):

    # #####################################
    #  Init_param
    # #####################################
    nbr_workers = args.nbr_worker

    spacing_lst = args.spacing
    # FOV = args.agent_FOV

    scan_lst = []
    for spacing in spacing_lst:
       scan_lst.append([])

    U_fcsv_lst = []
    L_fcsv_lst = []
    CB_fcsv_lst = []
    

    print("Reading folder : ", args.dir_scans)
    print("Selected spacings : ", args.spacing)
    		
    normpath = os.path.normpath("/".join([args.dir_scans, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            baseName = os.path.basename(img_fn)
            if True in [scan in baseName for scan in ["scan","Scan"]]:
                for i,spacing in enumerate(spacing_lst):
                    if "_"+str(spacing) in baseName:
                        scan_lst[i].append(img_fn)

        if os.path.isfile(img_fn) and ".fcsv" in img_fn:
            baseName = os.path.basename(img_fn)
            if "_U." in baseName :
                U_fcsv_lst.append(img_fn)
            elif "_L." in baseName :
                L_fcsv_lst.append(img_fn)
            elif "_CB." in baseName :
                CB_fcsv_lst.append(img_fn)


    data_lst = []
    for n in range(0,len(scan_lst[0])):
        data = {}

        images_path = []
        for i,spacing in enumerate(spacing_lst):
            images_path.append(scan_lst[i][n])
        data["images"] = images_path
        data["u"] = U_fcsv_lst[n]
        data["l"] = L_fcsv_lst[n]
        data["cb"] = CB_fcsv_lst[n]

        data_lst.append(data)

    # print(data_lst)

    
    environement_lst = []
    for data in data_lst:
        print("Generating Environement for :" , os.path.dirname(data["images"][0]))
        env = Environement(data["images"],np.array(args.agent_FOV)/2)
        for fcsv in args.landmarks:
            env.LoadLandmarks(data[fcsv])

        environement_lst.append(env)

    agent_lst = []
    for fcsv in args.landmarks:
        for label in Label_dic[fcsv]:
            print("Generating Agent for the lamdmark :" , label)
            agt = TrainingAgent(
                targeted_landmark=label,
                # models=DRLnet,
                FOV=args.agent_FOV,
                verbose=True
            )
            agent_lst.append(agt)


    

    # a = agent_lst[1]
    # # a = TrainingAgent()
    # env = environement_lst[0]

    # a.SetEnvironement(env)

    # a.GoToScale(1)
    # a.SetRandomPos()

    # for i in range(100):
    #     a.Move(0)

    # img = a.GetState()
    # output = sitk.GetImageFromArray(img[0][:])
    # writer = sitk.ImageFileWriter()
    # writer.SetFileName("test.nii.gz")
    # writer.Execute(output)

    # if not os.path.exists("crop"):
    #     os.makedirs("crop")

    # for epoch in range(10):
    #     print("Epoch :", epoch+1)
    #     a.SetEnvironement(random.choice(environement_lst))
    #     a.Train(5)
    #     img = a.GetState()
    #     output = sitk.GetImageFromArray(img[0][:])
    #     writer = sitk.ImageFileWriter()
    #     writer.SetFileName(f"crop/test_{epoch}.nii.gz")
    #     writer.Execute(output)

    # a.Validate(5)
        

    # img = a.GetState()
    # output = sitk.GetImageFromArray(img[0][:])
    # writer = sitk.ImageFileWriter()
    # writer.SetFileName("test.nii.gz")
    # writer.Execute(output)


    # trainingSet, validationSet = train_test_split(data_lst, test_size=args.test_percentage/100, random_state=len(data_lst))  




# #####################################
#  Args
# #####################################

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    # input_group.add_argument('--dir_project', type=str, help='Directory with all the project',default='/Users/luciacev-admin/Documents/Projects/ALI_benchmark')
    # input_group.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/data')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans')#,default=parser.parse_args().dir_data+'/Scans')

    # input_group.add_argument('--dir_cash', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/Cash')
    # input_group.add_argument('--dir_model', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/ALI_models')


    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: u l cb)", default=["cb"])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[2,1,0.3])
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mi', '--max_iterations', type=int, help='Number of training epocs', default=25000)
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=20)
    input_group.add_argument('-mn', '--model_name', type=str, help='Name of the model', default="ALI_model")
    # input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=19)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=2)

    args = parser.parse_args()
    
    main(args)


