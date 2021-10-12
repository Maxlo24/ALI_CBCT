

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


