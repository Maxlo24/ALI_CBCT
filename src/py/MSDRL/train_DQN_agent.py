# from torch._C import device
from utils import (
    GetEnvironementsAgents,
    PlotAgentPath
)

import SimpleITK as sitk
import os
import torch

from GlobalVar import*
from Models_class import (Brain,DQN)
from Agents_class import (DQNAgent)
from Environement_class import (Environement)
from TrainingManager_class import (TrainingMaster)

import argparse


def main(args):

    # #####################################
    #  Init_param
    # #####################################
    nbr_workers = args.nbr_worker

    spacing_lst = args.spacing
    agent_FOV = args.agent_FOV
    dim = len(spacing_lst)

    batch_size = 100
    data_size = 100000

    environement_lst, agent_lst = GetEnvironementsAgents(args.dir_scans,spacing_lst,DQNAgent,agent_FOV,args.landmarks)
    # agent_lst = [agent_lst[0]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for agent in agent_lst:
        dir_path = os.path.join(args.dir_model,agent.target)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        agent.SetBrain(Brain(
            network_type = DQN,
            network_nbr = dim,
            model_dir = dir_path,
            model_name = agent.target,
            device = device,
            in_channels = 1,
            out_channels = 26,
            learning_rate = 1e-4,
            batch_size= batch_size,
            verbose=True
            ))

    Master = TrainingMaster(
        environement_lst= environement_lst,
        agent_lst = agent_lst, 
        max_train_memory_size = data_size,
        max_val_memory_size= data_size,
        val_percentage = 0.2,
        env_dim = dim,
        num_worker = nbr_workers,
        batch_size = batch_size,
        )

    Master.GenerateAllDataset(data_size)
    Master.GenerateAllDataloaders()

    Master.Train(
        max_epoch = args.max_epoch,
        data_size = int(data_size/2),
        data_update_freq = 1,
        val_freq = 1)

    # a = agent_lst[0]
    # e = environement_lst[1]

    # a.SetEnvironement(e)

    # a.verbose = False
    # for dim in range(2):
    #     a.GoToScale(dim)  
    #     for i in range(50):
    #         a.SetRandomPos()
    #         a.Search(80)

    # for i in range(50):

    # a.SetRandomPos()
    # a.Search(100)
    # a.GoToScale(0)
    # a.brain.LoadModels(["/Users/luciacev-admin/Desktop/MSDRL_models/Ba_2021_14_10_E_2.pth","/Users/luciacev-admin/Desktop/Ba_2021_13_10_E_30.pth","/Users/luciacev-admin/Desktop/Ba_2021_12_10_E_14.pth"])
    # for i in range(50):
    #     print("Reset")
    #     a.SetRandomPos()
    #     a.Search()

    # a.GoToScale(1)
    # a.SetRandomPos()
    # a.Search(30)
    # a.GoToScale(2)
    # a.SetRandomPos()
    # a.Search(30)

    # PlotAgentPath(a)

    # if not os.path.exists("crop"):
    #     os.makedirs("crop")

    # for key,value in Master.dataset.items():
    #     for k,v in value.items():
    #         for n,dq in enumerate(v):
    #             for obj in dq:
                    # print(obj)
                    # output = sitk.GetImageFromArray(obj["state"][0][:])
                    # writer = sitk.ImageFileWriter()
                    # writer.SetFileName(f"crop/test_{key}_{n}.nii.gz")
                    # writer.Execute(output)


    # Master.GenerateTrainingDataset(1000)
    # Master.GenerateValidationDataset(200)
    # Master.GenerateTrainDataLoader()
    # Master.GenerateValidationDataLoader()
    # Master.TrainAgents(1,2)



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
    input_group.add_argument('--dir_model', type=str, help='Output directory of the training',default='ALI_models') # parser.parse_args().dir_data+

    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: u l cb)", default=["u","l","cb"])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[2,0.3])
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mi', '--max_epoch', type=int, help='Number of training epocs', default=10)
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=20)
    input_group.add_argument('-mn', '--model_name', type=str, help='Name of the model', default="ALI_model")
    # input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=19)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=2)

    args = parser.parse_args()
    
    main(args)




