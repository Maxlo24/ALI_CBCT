# from torch._C import device
from utils import (
    GetAgentLst,
    PlotAgentPath,
    CheckCrops,
    GetEnvironmentLst
)

import datetime

import GlobalVar as GV

from Environement_class import (Environement)
from Agents_class import (Agent)
from Models_class import (Brain,DNet,RNet)
from TrainingManager_class import (TrainingMaster)

import argparse
from resnet2p1d import *

import numpy as np
import os

def main(args):

    # #####################################
    #  Init_param
    # #####################################

    scale_spacing = args.scale_spacing  #Number of scale to use for the multi-scale environment

    image_dim = len(args.agent_FOV) # Dimention of the images 2D or 3D
    agent_FOV = args.agent_FOV # FOV of the agent

    batch_size = args.batch_size # Batch size for the training
    dataset_size = args.dataset_size # Size of the dataset to generate for the training

    GV.SCALE_KEYS = [str(scale).replace('.','-') for scale in scale_spacing]

    environments_param = {
        "type" : Environement,
        "dir" : args.dir_scans,
        "scale_spacing" : scale_spacing,
        "padding" : np.array(agent_FOV)/2 + 1,
        "device" : GV.DEVICE,
        "verbose" : False,
    }

    environement_lst = GetEnvironmentLst(environments_param)

    # environement_lst[0].SavePredictedLandmarks(multi_scale_keys[0])

    # return

    agents_param = {
        "type" : Agent,
        "FOV" : agent_FOV,
        "movements" : GV.MOVEMENTS,
        "scale_keys" : GV.SCALE_KEYS,
        "spawn_rad" : args.spawn_radius,
        "speed_per_scale" : args.speed_per_scale,
        "verbose" : True
    }

    agent_lst = GetAgentLst(agents_param, GV.LABELS_TO_TRAIN)

    # environement_lst, agent_lst = GetTrainingEnvironementsAgents(environments_param,agents_param)

    trainsitionLayerSize = 1024

    for agent in agent_lst:
        target = agent.target
        print(f"{GV.bcolors.OKCYAN}Generating brain for {GV.bcolors.OKBLUE}{target}{GV.bcolors.ENDC}{GV.bcolors.OKCYAN} agent.{GV.bcolors.ENDC}")
        dir_path = os.path.join(args.dir_model,target)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        agent.SetBrain(Brain(
            network_type = DNet,
            network_scales = GV.SCALE_KEYS,
            model_dir = dir_path,
            model_name = target,
            device = GV.DEVICE,
            in_channels = trainsitionLayerSize,
            out_channels = len(GV.MOVEMENTS["id"]),
            learning_rate = args.learning_rate,
            batch_size= batch_size,
            generate_tensorboard=True,
            verbose=True
            ))

    print()

    training_scales = [GV.SCALE_KEYS[scale] for scale in args.training_scales]

    print('Training on scales : ',training_scales)

    Master = TrainingMaster(
        environement_lst= environement_lst,
        agent_lst = agent_lst, 
        model_dir = args.dir_model,
        max_train_memory_size = dataset_size,
        max_val_memory_size= dataset_size*2,
        val_percentage = args.test_percentage/100,
        env_scales = training_scales,
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
    # CheckCrops(Master,agent,1)
    # e = environement_lst[1]
    # e.SetRandomRotation()
    # e.SaveCBCT(0,"/Users/luciacev-admin/Desktop/test/test.nii.gz")
    # print(e.dim_landmarks[0]["Ba"])
    # e.SaveEnvironmentState()



# #####################################
#  Args
# #####################################

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_project', type=str, help='Directory with all the project',default='/home/luciacev/Desktop/Maxime_Gillot/Trainings/ALI_CBCT', required=False)
    input_group.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/data')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default=parser.parse_args().dir_data+'/Patients')
    input_group.add_argument('--dir_model', type=str, help='Output directory of the training',default= parser.parse_args().dir_data+'/ALI_models_'+datetime.datetime.now().strftime("%Y_%d_%m"))

    #Environment
    # input_group.add_argument('-lm','--landmark_group',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: U L CB CI)", default=["CI"])
    input_group.add_argument('-sp', '--scale_spacing', nargs="+", type=float, help='Spacing of the different scales', default=[0.3,0.08])
    input_group.add_argument('-ts', '--training_scales', nargs="+", type=float, help='Scale to train', default=[0,1])


    #Agent
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-sps', '--speed_per_scale', nargs="+", type=int, help='Speed for each environment scale', default=[1,1])
    input_group.add_argument('-sr', '--spawn_radius', type=int, help='spawning radius around landmark', default=30)
    input_group.add_argument('-fr', '--focus_radius', type=int, help='focus radius around landmark', default=4)


    #Training data

    input_group.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=150)
    input_group.add_argument('-ds', '--dataset_size', type=int, help='Size of the randomly generated dataset', default=12000)
    # input_group.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=10)
    # input_group.add_argument('-ds', '--dataset_size', type=int, help='Size of the randomly generated dataset', default=100)
    input_group.add_argument('-duf', '--data_update_freq', type=int, help='Data update frequency', default=1)
    input_group.add_argument('-dur', '--data_update_ratio', type=float, help='Ratio of data to update', default=0.5)
    #Training param
    input_group.add_argument('-mi', '--max_epoch', type=int, help='Number of training epocs', default=1000)
    input_group.add_argument('-vf', '--val_freq', type=int, help='Validation frequency', default=1)
    input_group.add_argument('-tp', '--test_percentage', type=int, help='Percentage of data to keep for validation', default=15)
    input_group.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=1e-4)
    input_group.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker (CPU)', default=5)

    args = parser.parse_args()
    
    main(args)




