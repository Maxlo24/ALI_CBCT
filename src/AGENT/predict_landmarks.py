# from utils import (
#     GetAgentLst,
#     GetEnvironmentLst,
#     PlotAgentPath,
#     GetBrain,
#     ReslutAccuracy,
#     ResultDiscretAccuracy,
#     PlotResults
# )

from utils import*
import glob
import sys
from shutil import copyfile


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
import shutil

def main(args):

    print("Reading folder : ",args.dir_scans)
    print("Selected spacings : ", args.spacing)

    scale_spacing = args.spacing  #Number of scale to use for the multi-scale environment


    patients = {}
    		
    normpath = os.path.normpath("/".join([args.dir_scans, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:

            if basename not in patients.keys():
                patients[basename] = {"scan": img_fn, "scans":{}}

            
    temp_fold = os.path.join(args.dir_temp, "temp")
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)


    
    for patient,data in patients.items():

        scan = data["scan"]

        scan_name = patient.split(".")

        tempPath = os.path.join(temp_fold, patient)

        if not os.path.exists(tempPath):
            CorrectHisto(scan, tempPath,0.01, 0.99)


        for sp in scale_spacing:
            new_name = ""
            spac = str(sp).replace(".","-")
            for i,element in enumerate(scan_name):
                if i == 0:
                    new_name = scan_name[0] + "_scan_sp" + spac
                else:
                    new_name += "." + element
            
            outpath = os.path.join(temp_fold,new_name)
            if not os.path.exists(outpath):
                SetSpacing(tempPath,[sp,sp,sp],outpath)
            patients[patient]["scans"][spac] = outpath


    print("Patients : ",patients)



    # #####################################
    #  Init_param
    # #####################################


    image_dim = len(args.agent_FOV) # Dimention of the images 2D or 3D
    agent_FOV = args.agent_FOV # FOV of the agent

    # batch_size = args.batch_size # Batch size for the training
    # dataset_size = args.dataset_size # Size of the dataset to generate for the training

    GV.SCALE_KEYS = [str(scale).replace('.','-') for scale in scale_spacing]

    # environments_param = {
    #     "type" : Environement,
    #     "dir" : args.dir_scans,
    #     "scale_spacing" : scale_spacing,
    #     "padding" : np.array(agent_FOV)/2 + 1,
    #     "device" : GV.DEVICE,
    #     "verbose" : False,
    # }


    environement_lst = GenEnvironmentLst(patient_dic = patients,env_type = Environement, padding =  np.array(agent_FOV)/2 + 1, device = GV.DEVICE)
    # environement_lst = GetEnvironmentLst(environments_param)

    # environement_lst[0].SavePredictedLandmarks(multi_scale_keys[0])

    # return

    agents_param = {
        "type" : Agent,
        "FOV" : agent_FOV,
        "movements" : GV.MOVEMENTS,
        "scale_keys" : GV.SCALE_KEYS,
        "spawn_rad" : args.spawn_radius,
        "speed_per_scale" : args.speed_per_scale,
        "focus_radius" : args.focus_radius,
        "verbose" : True
    }

    agent_lst = GetAgentLst(agents_param, args.landmarks)


    # agent_lst = GetAgentLst(agents_param)
    brain_lst = GetBrain(args.dir_models)
    # print( brain_lst)
    # environement_lst = [environement_lst[0]]
    # agent_lst = [agent_lst[0]]

    trainsitionLayerSize = 1024


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
    fails = {}
    for environment in environement_lst:
        print(environment.patient_id)
        # print(environment)
        for agent in agent_lst:
            agent.SetEnvironement(environment)
            search_result = agent.Search()
            if search_result == -1:
                fails[agent.target] = fails.get(agent.target,0) + 1
            else:
                tot_step += search_result
            # PlotAgentPath(agent)
        outPath = os.path.dirname(patients[environment.patient_id]["scan"])
        environment.SavePredictedLandmarks(GV.SCALE_KEYS[-1],outPath)
    
    print("Total steps:",tot_step)    
    end_time = time.time()
    print('prediction time :' , end_time-start_time)
        

    for lm, nbr in fails.items():
        print(f"Fails for {lm} : {nbr}/{len(environement_lst)}")

    if args.clear_temp:
        try:
            shutil.rmtree(temp_fold)
        except OSError as e:
            print("Error: %s : %s" % (temp_fold, e.strerror))


    # data_discret_result = ResultDiscretAccuracy(environement_lst,args.spacing[-1])
    # PlotResults(data_discret_result)



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training for Automatic Landmarks Identification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('dir')
    input_group.add_argument('--dir_scans', type=str, help='Input directory with the scans',default='/app/data/scans')
    input_group.add_argument('--dir_models', type=str, help='Directory of the trained models',default= '/app/data/models')
    
    input_group.add_argument('--clear_temp', type=bool, help='Temp directory',default= True)
    input_group.add_argument('--dir_temp', type=str, help='Temp directory',default= '..')


    #Environment
    input_group.add_argument('-lm','--landmarks',nargs="+",type=str,help="Prepare the data for uper and/or lower landmark training (ex: U L CB)", default=['Ba', 'S'])
    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Spacing of the different scales', default=[1,0.3])
    input_group.add_argument('-sps', '--speed_per_scale', nargs="+", type=int, help='Speed for each environment scale', default=[1,1])
    input_group.add_argument('-sr', '--spawn_radius', type=int, help='spawning radius around landmark', default=10)
    input_group.add_argument('-fr', '--focus_radius', type=int, help='focus radius around landmark', default=4)


    #Agent
    input_group.add_argument('-fov', '--agent_FOV', nargs="+", type=float, help='Wanted crop size', default=[64,64,64])
    input_group.add_argument('-mvt','--movement', type=str, help='Number of posssible agent movement',default='6') # parser.parse_args().dir_data+
    
    args = parser.parse_args()
    main(args)




