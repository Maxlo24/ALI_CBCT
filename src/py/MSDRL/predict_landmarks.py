from utils import (
    GenPredictEnvironment,
    GetAgentLst,
    PlotAgentPath,
    GetBrain,
    GetTrainingEnvironementsAgents,
    ReslutAccuracy,
    PlotResults
)

import SimpleITK as sitk
import os
import torch

from GlobalVar import*
from Models_class import (Brain,DQN)
from Agents_class import (DQNAgent)
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
        "verbose" : False
    }

    agents_param = {
        "type" : DQNAgent,
        "FOV" : args.agent_FOV,
        "landmarks" : args.landmarks,
        "movements" : movements,
        "spawn_rad" : 20,
        "dim" : dim,
        "verbose" : True
    }

    # environement_lst = GenPredictEnvironment(environments_param,agents_param)
    environement_lst, agent_lst = GetTrainingEnvironementsAgents(environments_param,agents_param)

    agent_lst = GetAgentLst(agents_param)
    brain_lst = GetBrain(args.dir_model)
    # environement_lst = [environement_lst[0]]
    # agent_lst = [agent_lst[0]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for agent in agent_lst:
        brain = Brain(
            network_type = DQN,
            network_nbr = dim,
            device = device,
            in_channels = 1,
            in_size = args.agent_FOV,
            out_channels = len(movements["id"]),
            )
        brain.LoadModels(brain_lst[agent.target])
        agent.SetBrain(brain)

    for environment in environement_lst:
        print(environment.images_path[0])
        for agent in agent_lst:
            agent.SetEnvironement(environment)
            agent.Search()
            # PlotAgentPath(agent)
        environment.SavePredictedLandmarks()
        


    # for e in environement_lst:
    #     for key in args.landmarks:
    #         for lm in LABELS[key]:
    #             if e.LandmarkIsPresent(lm):
    #                 e.AddPredictedLandmark(lm,e.GetLandmarkPos(1,lm))
    #                 e.SavePredictedLandmarks()

    data_result = ReslutAccuracy(args.dir_scans)
    PlotResults(data_result)



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default='/Users/luciacev-admin/Documents/Projects/MSDRL_benchmark/data/test')
    input_group.add_argument('--dir_model', type=str, help='Directory of the trained models',default= '/Users/luciacev-admin/Desktop/MSDRL_models/ALI_CNN_models_2021_26_10')

    #Environment
    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: U L CB)", default=["U","L","CB"])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[1,0.3])
    
    #Agent
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mvt','--movement', type=str, help='Number of posssible agent movement',default='6') # parser.parse_args().dir_data+
    
    args = parser.parse_args()
    main(args)