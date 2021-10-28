# from torch._C import device
from utils import (
    GetTrainingEnvironementsAgents,
    PlotAgentPath,
    CheckCrops
)

import SimpleITK as sitk
import os
import torch
import datetime


from GlobalVar import*
from Models_class import (Brain,DQN,MaxDQN)
from Agents_class import (DQNAgent)
from Environement_class import (Environement)
from TrainingManager_class import (TrainingMaster)

import argparse


def main(args):

    # #####################################
    #  Init_param
    # #####################################

    dim = len(args.spacing)
    movements = MOVEMENTS[args.movement]

    batch_size = args.batch_size
    data_size = args.data_size

    environments_param = {
        "type" : Environement,
        "dir" : args.dir_scans,
        "spacings" : args.spacing,
        "verbose" : False
    }

    agents_param = {
        "type" : DQNAgent,
        "FOV" : args.agent_FOV,
        "landmarks" : args.landmarks,
        "movements" : movements,
        "spawn_rad" : args.spawn_radius,
        "dim" : dim,
        "verbose" : True
    }

    environement_lst, agent_lst = GetTrainingEnvironementsAgents(environments_param,agents_param)
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
            in_size = args.agent_FOV,
            out_channels = len(movements["id"]),
            learning_rate = args.learning_rate,
            batch_size= batch_size,
            generate_tensorboard=True,
            verbose=True
            ))

    Master = TrainingMaster(
        environement_lst= environement_lst,
        agent_lst = agent_lst, 
        max_train_memory_size = data_size,
        max_val_memory_size= data_size*2,
        val_percentage = args.test_percentage/100,
        env_dim = dim,
        num_worker = args.nbr_worker,
        batch_size = batch_size,
        )

    Master.Train(
        max_epoch = args.max_epoch,
        val_freq = args.val_freq,
        data_update_freq = args.data_update_freq,
        data_update_ratio= args.data_update_ratio
        )

    # agent = agent_lst[0]
    # CheckCrops(Master,agent)
    # e = environement_lst[0]

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
    # a.brain.LoadModels(["/Users/luciacev-admin/Desktop/MSDRL_models/Ba_2021_18_10_E_2.pth","/Users/luciacev-admin/Desktop/MSDRL_models/Ba_2021_18_10_E_2.pth"])
    # for i in range(10):
    #     print("Reset")
    # a.Search()
    
    # PlotAgentPath(a)









# #####################################
#  Args
# #####################################

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_project', type=str, help='Directory with all the project',default='/Users/luciacev-admin/Documents/Projects/MSDRL_benchmark')
    input_group.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/data')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default=parser.parse_args().dir_data+'/patients')

    # input_group.add_argument('--dir_cash', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/Cash')
    input_group.add_argument('--dir_model', type=str, help='Output directory of the training',default= parser.parse_args().dir_data+'/ALI_CNN_models_'+datetime.datetime.now().strftime("%Y_%d_%m"))

    #Environment
    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: U L CB)", default=["CB"])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[1,0.3])
    
    #Agent
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mvt','--movement', type=str, help='Number of posssible agent movement',default='6') # parser.parse_args().dir_data+
    input_group.add_argument('-sr', '--spawn_radius', type=int, help='Wanted crop size', default=30)


    #Training data
    input_group.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=250)
    input_group.add_argument('-ds', '--data_size', type=int, help='Size of the dataset', default=50000)
    input_group.add_argument('-duf', '--data_update_freq', type=int, help='Data update frequency', default=1)
    input_group.add_argument('-dur', '--data_update_ratio', type=float, help='Ratio of data to update', default=0.5)
    #Training param
    input_group.add_argument('-mi', '--max_epoch', type=int, help='Number of training epocs', default=1000)
    input_group.add_argument('-vf', '--val_freq', type=int, help='Validation frequency', default=2)
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=20)
    input_group.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=1e-4)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=4)

    args = parser.parse_args()
    
    main(args)




