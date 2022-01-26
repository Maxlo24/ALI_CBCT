from utils import (
    GenPredictEnvironment,
    GetAgentLst,
    PlotAgentPath,
    GetBrain,
    GetTrainingEnvironementsAgents,
    ReslutAccuracy,
    ResultDiscretAccuracy,
    PlotResults
)

import SimpleITK as sitk
import os
import torch

from GlobalVar import*
from Models_class import (Brain,DNet,RNet,ADL,DQN,Gen121DensNet)
from Agents_class import (DQNAgent,)
from Environement_class import (Environement)

import argparse

def main(args):

    # #####################################
    #  Init_param
    # #####################################

    dim = len(args.spacing)
    movements = MOVEMENTS[args.movement]

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
        "spawn_rad" : 10,
        "dim" : dim,
        "verbose" : True
    }

    # environement_lst = GenPredictEnvironment(environments_param,agents_param)
    # environement_lst, agent_lst = GetTrainingEnvironementsAgents(environments_param,agents_param)

    # agent_lst = GetAgentLst(agents_param)
    brain_lst = GetBrain(args.dir_model)
    # environement_lst = [environement_lst[0]]
    # agent_lst = [agent_lst[0]]

    trainsitionLayerSize = 2048

    # featNet = Gen121DensNet(
    #     i_channels=1,
    #     o_channels=trainsitionLayerSize
    # ).to(DEVICE)
    featNet = None

    # print("Loading Feature Net" , args.feat_extract_model)
    # featNet.load_state_dict(torch.load(args.feat_extract_model,map_location=DEVICE)) 

    # for agent in agent_lst:
    #     brain = Brain(
    #         network_type = DNet,
    #         network_nbr = dim,
    #         device = DEVICE,
    #         in_channels = trainsitionLayerSize,
    #         out_channels = len(movements["id"]),
    #         feature_extract_net=featNet,
    #         pretrained_featNet=True,
    #         )
    #     brain.LoadModels(brain_lst[agent.target])
    #     agent.SetBrain(brain)

    # for environment in environement_lst:
    #     print(environment.images_path[0])
    #     for agent in agent_lst:
    #         agent.SetEnvironement(environment)
    #         agent.Search()
    #         # PlotAgentPath(agent)
    #     environment.SavePredictedLandmarks()
        

    # for e in environement_lst:
    #     for key in args.landmarks:
    #         for lm in LABELS[key]:
    #             if e.LandmarkIsPresent(lm):
    #                 e.AddPredictedLandmark(lm,e.GetLandmarkPos(1,lm))
    #                 e.SavePredictedLandmarks()

    data_result = ReslutAccuracy(args.dir_scans)
    # PlotResults(data_result)

    # data_discret_result = ResultDiscretAccuracy(environement_lst,args.spacing[-1])
    # PlotResults(data_discret_result)



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default='/Users/luciacev-admin/Documents/Projects/MSDRL_benchmark/data/test')
    input_group.add_argument('--dir_model', type=str, help='Directory of the trained models',default= '/Users/luciacev-admin/Desktop/MSDRL_models/ALI_CNN_models_2021_26_10')
    input_group.add_argument('-fem','--feat_extract_model', type=str, help='Directory of the trained feature extraction models',default= '/Users/luciacev-admin/Desktop/MSDRL_models/ALI_CNN_models_2021_26_10')

    #Environment
    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: U L CB)", default=["U"])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[1,0.3])
    
    #Agent
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mvt','--movement', type=str, help='Number of posssible agent movement',default='6') # parser.parse_args().dir_data+
    
    args = parser.parse_args()
    main(args)