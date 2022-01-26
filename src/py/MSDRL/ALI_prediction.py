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

class PredictMaster:
    def __init__(
        self,
        scan_lst,
        lm_lst,
        spacings = [1,0.3],
        agent_FOV = [64,64,64],
        brain_type = DNet,
        transition_layer = 2048

        ) -> None:
        self.scan_lst = scan_lst
        self.lm_lst = lm_lst
        self.spacings = spacings
        self.dim = len(spacings)
        self.agent_FOV = agent_FOV
        self.brain_type = brain_type
        self.transition_layer = transition_layer
        self.GenerateEnvironments()

    def GenerateEnvironments(self):
        scan_lst = self.scan_lst

        environment_lst = []
        for scan in scan_lst:
            print("Generating Environement for :" , scan)
            env = Environement(
                padding = np.array(self.agent_FOV)/2+1,
                device=DEVICE,
                verbose=False
                )
            env.GenerateImages(scan,self.spacings)
            environment_lst.append(env)
        self.environement_lst = environment_lst

    def GenerateAgents(self,agents_param,brain_lst):
        agent_lst =[]
        for label in self.lm_lst:
            print("Generating Agent for the lamdmark :" , label)
            agt = agents_param["type"](
                targeted_landmark=label,
                # models=DRLnet,
                movements = agents_param["movements"],
                env_dim = agents_param["dim"],
                FOV=agents_param["FOV"],
                start_pos_radius = agents_param["spawn_rad"],
                verbose = agents_param["verbose"]
            )
            agent_lst.append(agt)

        for agent in agent_lst:
            brain = Brain(
                network_type = self.brain_type,
                network_nbr = self.dim,
                device = DEVICE,
                in_channels = self.transition_layer,
                out_channels = len(agent.movement_id),
                feature_extract_net=None,
                pretrained_featNet=True,
                )
            brain.LoadModels(brain_lst[agent.target])
            agent.SetBrain(brain)
        
        self.agent_lst = agent_lst

    def Process(self):
        for environment in self.environement_lst:
            print(environment.images_path[0])
            for agent in self.agent_lst:
                agent.SetEnvironement(environment)
                agent.Search()
                # PlotAgentPath(agent)
            environment.SavePredictedLandmarks()

def main(args):

    # #####################################
    #  Init_param
    # #####################################

    dim = len(args.spacings)
    movements = MOVEMENTS[args.movement]


    agents_param = {
        "type" : DQNAgent,
        "FOV" : args.agent_FOV,
        "landmarks" : args.landmarks,
        "movements" : movements,
        "spawn_rad" : 10,
        "dim" : dim,
        "verbose" : False
    }

    brain_lst = GetBrain(args.dir_model)
    Pred = PredictMaster(
        scan_lst=args.scans,
        lm_lst=args.landmarks,
        spacings=args.spacings,
        agent_FOV=args.agent_FOV,
        brain_type=DNet,
        transition_layer=2048
    )

    Pred.GenerateAgents(agents_param,brain_lst)
    Pred.Process()



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('-s','--scans',nargs="+",type=str,help="Scans to predict", default=['/Users/luciacev-admin/Desktop/TEST_MODULE/01_T0_scan_or.nii.gz'])
    input_group.add_argument('--dir_model', type=str, help='Directory of the trained models',default= '/Users/luciacev-admin/Desktop/TEST_DENSENET')

    #Environment
    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Which landmark to predict", default=["Gn","S"])
    input_group.add_argument('-sp', '--spacings', nargs="+", type=float, help='Spacing of the different scales', default=[1,0.3])
    
    #Agent
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mvt','--movement', type=str, help='Number of posssible agent movement',default='6') # parser.parse_args().dir_data+
    
    args = parser.parse_args()
    main(args)