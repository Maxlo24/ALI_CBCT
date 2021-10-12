from utils import (
    GetEnvironementsAgents,
)

import SimpleITK as sitk
import os

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

    batch_size = 25

    environement_lst, agent_lst = GetEnvironementsAgents(args.dir_scans,spacing_lst,DQNAgent,agent_FOV,args.landmarks)

    for agent in agent_lst:
        dir_path = os.path.join(args.dir_model,agent.target)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        agent.SetBrain(Brain(
            network_type = DQN,
            network_nbr = len(spacing_lst),
            model_dir = dir_path,
            model_name = agent.target,
            in_channels = 1,
            out_channels = 6,
            learning_rate = 1e-5,
            batch_size= batch_size,
            verbose=True
            ))

    Master = TrainingMaster(
        environement_lst= environement_lst,
        agent_lst = agent_lst, 
        max_memory_size = 1000,
        val_percentage = 0.2,
        env_dim = 3,
        num_worker = nbr_workers,
        batch_size = batch_size,
        )

    Master.Train(
        max_epoch = args.max_epoch,
        data_size = 1000,
        data_update_freq = 5,
        val_freq = 2)

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

    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: u l cb)", default=["cb"])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[2,1,0.3])
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mi', '--max_epoch', type=int, help='Number of training epocs', default=100)
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=20)
    input_group.add_argument('-mn', '--model_name', type=str, help='Name of the model', default="ALI_model")
    # input_group.add_argument('-nl', '--nbr_label', type=int, help='Number of label', default=19)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=2)

    args = parser.parse_args()
    
    main(args)




