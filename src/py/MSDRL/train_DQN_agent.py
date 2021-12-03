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
from Models_class import (Brain,ADL,DQN,MaxDQN,Gen121DensNet)
from Agents_class import (DQNAgent)
from Environement_class import (Environement)
from TrainingManager_class import (TrainingMaster)

import argparse
from resnet2p1d import *

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
        "padding" : np.array(args.agent_FOV)/2+1,
        "landmarks" : args.landmarks,
        "verbose" : False,
        "rotated" : False
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

    trainsitionLayerSize = 2048

    # featNet = Gen121DensNet(
    #     i_channels=1,
    #     o_channels=trainsitionLayerSize
    # ).to(DEVICE)

    featNet = generate_model(
        model_depth = 18,
        n_input_channels=1,
        n_classes=trainsitionLayerSize
    ).to(DEVICE)

    for agent in agent_lst:
        dir_path = os.path.join(args.dir_model,agent.target)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        agent.SetBrain(Brain(
            network_type = ADL,
            network_nbr = dim,
            model_dir = dir_path,
            model_name = agent.target,
            device = DEVICE,
            in_channels = trainsitionLayerSize,
            out_channels = len(movements["id"]),
            feature_extract_net=featNet,
            pretrained_featNet=False,
            learning_rate = args.learning_rate,
            batch_size= batch_size,
            generate_tensorboard=True,
            verbose=True
            ))

    Master = TrainingMaster(
        environement_lst= environement_lst,
        agent_lst = agent_lst, 
        featNet = featNet,
        model_dir = args.dir_model,
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
    # e = environement_lst[1]
    # e.SetRandomRotation()
    # e.SetRandomRotation()
    # e.SetRandomRotation()

    # e.SaveCBCT(1,"/Users/luciacev-admin/Desktop/test/test.nii.gz")
    # print(e.dim_landmarks[1]["Ba"])
    # e.SaveCBCT(1,"/Users/luciacev-admin/Desktop/test")

    # e.SaveEnvironmentState()



# #####################################
#  Args
# #####################################

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_project', type=str, help='Directory with all the project',default='/Users/luciacev-admin/Documents/Projects/Benchmarks/MSDRL_benchmark')
    input_group.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/data')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default=parser.parse_args().dir_data+'/patients')
    # input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default='/Users/luciacev-admin/Desktop/Agent_Training_data')


    # input_group.add_argument('--dir_cash', type=str, help='Output directory of the training',default=parser.parse_args().dir_data+'/Cash')
    input_group.add_argument('--dir_model', type=str, help='Output directory of the training',default= parser.parse_args().dir_data+'/ALI_CNN_models_'+datetime.datetime.now().strftime("%Y_%d_%m"))

    #Environment
    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: U L CB)", default=["CB"])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[1,0.3])
    
    #Agent
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mvt','--movement', type=str, help='Number of posssible agent movement',default='6') # parser.parse_args().dir_data+
    input_group.add_argument('-sr', '--spawn_radius', type=int, help='spawning radius around landmark', default=30)


    #Training data
    input_group.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=2)
    input_group.add_argument('-ds', '--data_size', type=int, help='Size of the dataset', default=10)
    input_group.add_argument('-duf', '--data_update_freq', type=int, help='Data update frequency', default=1)
    input_group.add_argument('-dur', '--data_update_ratio', type=float, help='Ratio of data to update', default=0.5)
    #Training param
    input_group.add_argument('-mi', '--max_epoch', type=int, help='Number of training epocs', default=1000)
    input_group.add_argument('-vf', '--val_freq', type=int, help='Validation frequency', default=4)
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=15)
    input_group.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=1e-4)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=0)

    args = parser.parse_args()
    
    main(args)




