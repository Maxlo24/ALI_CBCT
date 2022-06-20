from utils import (
    GetAgentLst,
    GetEnvironmentLst,
    PlotAgentPath,
    GetBrain,
    ReslutAccuracy,
    ResultDiscretAccuracy,
    PlotResults
)


import time

import GlobalVar as GV

from Environement_class import (Environement)
from Agents_class import (Agent)
from Models_class import (Brain,DNet,RNet)
from TrainingManager_class import (TrainingMaster)

import argparse
from resnet2p1d import *

import numpy as np
import os
import argparse

def main(args):

    # #####################################
    #  Init_param
    # #####################################

    scale_spacing = args.spacing  #Number of scale to use for the multi-scale environment

    image_dim = len(args.agent_FOV) # Dimention of the images 2D or 3D
    agent_FOV = args.agent_FOV # FOV of the agent

    # batch_size = args.batch_size # Batch size for the training
    # dataset_size = args.dataset_size # Size of the dataset to generate for the training

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

    agent_lst = GetAgentLst(agents_param)


    # agent_lst = GetAgentLst(agents_param)
    brain_lst = GetBrain(args.dir_model)
    # print( brain_lst)
    # environement_lst = [environement_lst[0]]
    # agent_lst = [agent_lst[0]]

    trainsitionLayerSize = 1024

    # featNet = Gen121DensNet(
    #     i_channels=1,
    #     o_channels=trainsitionLayerSize
    # ).to(DEVICE)
    featNet = None

    # print("Loading Feature Net" , args.feat_extract_model)
    # featNet.load_state_dict(torch.load(args.feat_extract_model,map_location=DEVICE)) 

    for agent in agent_lst:
        brain = Brain(
            network_type = DNet,
            network_scales = GV.SCALE_KEYS,
            # model_dir = dir_path,
            # model_name = target,
            device = GV.DEVICE,
            in_channels = trainsitionLayerSize,
            out_channels = len(GV.MOVEMENTS["id"]),
            batch_size= 1,
            generate_tensorboard=False,
            verbose=True
            )
        brain.LoadModels(brain_lst[agent.target])
        agent.SetBrain(brain)

    start_time = time.time()

    tot_step = 0
    for environment in environement_lst:
        print(environment.patient_id)
        # print(environment)
        for agent in agent_lst:
            agent.SetEnvironement(environment)
            tot_step += agent.Search()
            # PlotAgentPath(agent)
        environment.SavePredictedLandmarks(GV.SCALE_KEYS[-1])
    
    print("Total steps:",tot_step)    
    end_time = time.time()
    print('prediction time :' , end_time-start_time)
        

    # for e in environement_lst:
    #     for key in args.landmarks:
    #         for lm in LABELS[key]:
    #             if e.LandmarkIsPresent(lm):
    #                 e.AddPredictedLandmark(lm,e.GetLandmarkPos(1,lm))
    #                 e.SavePredictedLandmarks()

    data_result = ReslutAccuracy(environement_lst,GV.SCALE_KEYS[-1])
    PlotResults(data_result)

    # data_discret_result = ResultDiscretAccuracy(environement_lst,args.spacing[-1])
    # PlotResults(data_discret_result)



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default='/Users/luciacev-admin/Documents/Projects/MSDRL_benchmark/data/test')
    input_group.add_argument('--dir_model', type=str, help='Directory of the trained models',default= '/Users/luciacev-admin/Desktop/MSDRL_models/ALI_CNN_models_2021_26_10')
    input_group.add_argument('-fem','--feat_extract_model', type=str, help='Directory of the trained feature extraction models',default= '/Users/luciacev-admin/Desktop/MSDRL_models/ALI_CNN_models_2021_26_10')

    #Environment
    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: U L CB)", default=["U","L","CB"])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[1,0.3])
    input_group.add_argument('-sps', '--speed_per_scale', nargs="+", type=int, help='Speed for each environment scale', default=[1,1])
    input_group.add_argument('-sr', '--spawn_radius', type=int, help='spawning radius around landmark', default=10)


    #Agent
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mvt','--movement', type=str, help='Number of posssible agent movement',default='6') # parser.parse_args().dir_data+
    
    args = parser.parse_args()
    main(args)