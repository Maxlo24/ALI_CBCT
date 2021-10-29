
import torch
from torch._C import dtype
from torch.utils import data
from Models_class import DRLnet
from Agents_class import (
    DQNAgent,
    RLAgent
)
from Environement_class import Environement

from collections import deque
from sklearn.model_selection import train_test_split

import numpy as np
import os
from tqdm.std import tqdm
import time

# ----- MONAI ------
# from monai.losses import DiceCELoss
# from monai.inferers import sliding_window_inference

from monai.transforms import(
    RandShiftIntensityd,
    Compose,
    ScaleIntensityd
)
from monai.data import (
    DataLoader,
    CacheDataset,
    SmartCacheDataset,
    Dataset,
    decollate_batch,
)

class TrainingMaster :
    def __init__(
        self,
        environement_lst,
        agent_lst,
        max_train_memory_size = 100,
        max_val_memory_size = 100,
        val_percentage = 0.2,
        env_dim = 2,
        num_worker = 1,
        batch_size = 20,
    ) -> None:

        self.environements = environement_lst
        self.env_dim = env_dim
        # self.val_percentage = val_percentage
        self.SplitTrainValData(val_percentage)

        self.agents = agent_lst
        self.target_lst = []

        data_dic = {}
        for agent in agent_lst:
            target = agent.target
            self.target_lst.append(target)
            data_dic[target] = {"train" : [], "val" : []}
            for dim in range(self.env_dim):
                data_dic[agent.target]["train"].append({})
                data_dic[agent.target]["val"].append({})

        self.crop_dataset = {"train":deque(maxlen=max_train_memory_size), "val":deque(maxlen=max_val_memory_size)}

        self.pos_dataset = data_dic

        # self.data_transform = Compose([ScaleIntensityd(keys=["state"],minv = 0.0, maxv = 1.0,factor = None),RandShiftIntensityd(keys=["state"],offsets=0.10,prob=0.50,)])
        self.data_transform = RandShiftIntensityd(keys=["state"],offsets=0.10,prob=0.50)

        self.num_worker = num_worker
        self.batch_size = batch_size

        self.max_train_memory_size = max_train_memory_size
        self.max_val_memory_size = max_val_memory_size


    # ENVIRONEMENT MANAGEMENT

    def AddEnvironement(self,env):
        self.environements.append(env)

    def ResetEnvironements(self,env_lst = []):
        self.environements = env_lst

    # AGENT MANAGEMENT

    def AddAgent(self,agt):
        self.agents.append(agt)

    def ResetAgents(self,agt_lst = []):
        self.agents = agt_lst


    # DATA MANAGEMENT

    def SplitTrainValData(self,val_percentage = 0.2):
        train_env, val_env = train_test_split(self.environements, test_size=val_percentage, random_state=len(self.environements))
        self.s_env = {"train":train_env,"val":val_env}
        for key in self.s_env.keys():
            print(key,"environments :")
            lst = []
            for env in self.s_env[key]:
                lst.append(os.path.basename(os.path.dirname(env.images_path[0])))
            print(lst)

    def GeneratePosDataset(self,key,size):
        for agent in self.agents:
            nbr_generated_data = 0
            valid_env = []
            for env in self.s_env[key]:
                if env.LandmarkIsPresent(agent.target):
                    valid_env.append(env)

            # print("start")
            # start_time = time.time()
            # # pos_dataset = tqdm(range(size),desc="Generating "+key+" dataset for agent " + agent.target)
            # for i in range(size) :
            #     for env in valid_env:
            #         for dim in range(self.env_dim):
            #             self.pos_dataset[agent.target][key][dim].append({"env":env,"coord":env.GetRandomPos(dim,agent.target,agent.start_pos_radius)})
            # print("--- %s seconds ---" % (time.time() - start_time))

            target = agent.target
            start_pos_radius = agent.start_pos_radius

            
            start_time = time.time()
            print("Generating "+key+" dataset for agent " + agent.target,end="\r",flush=True)
            for dim in range(self.env_dim):
                data_per_env = int(size/len(valid_env))+1 
                for env in valid_env:
                    self.pos_dataset[target][key][dim][env] = env.GetRandomPoses(dim,target,start_pos_radius,data_per_env)
            print("Generating "+key+" dataset for agent " + agent.target+" : done in %2.1f seconds" % (time.time() - start_time))

    def GenerateDataLoader(self,key,agent,dim):
        dataset = []
        
        # pos_dataset = tqdm(
        #     self.pos_dataset[agent.target][key][dim],
        #     desc="Loading "+key+" crops for agent " + agent.target + " at scale "+ str(dim)
        #     )

        # for pos in self.pos_dataset[agent.target][key][dim]:            
        #     dataset.append(pos["env"].GetSample(dim,agent.target,pos["coord"],agent.FOV,agent.movement_matrix))

        start_time = time.time()
        print("Generating "+key+" crops for agent " + agent.target + " at scale "+ str(dim),end="\r",flush=True)
        target = agent.target
        FOV = agent.FOV
        mov_mat = agent.movement_matrix
        # get_sample = lambda pos: pos["env"].GetSample(dim,target,pos["coord"],FOV,mov_mat)

        # dataset = list(map(get_sample,self.pos_dataset[agent.target][key][dim]))        
        # print("Loading "+key+" crops for agent " + agent.target + " at scale "+ str(dim)+" : done in %2.1f seconds" % (time.time() - start_time))
        dataset = self.crop_dataset[key]
        for env,pos_lst in self.pos_dataset[agent.target][key][dim].items():
            dataset += env.GetSampleFromPoses(dim,target,pos_lst,FOV,mov_mat)
        print("Generating "+key+" crops for agent " + agent.target + " at scale "+ str(dim)+" : done in %2.1f seconds" % (time.time() - start_time))
        train_ds = CacheDataset(
            data=dataset,
            transform=self.data_transform,
            # cache_rate=1.0,
            # num_workers=self.num_worker,
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_worker)
        return train_loader
        
    # TOOLS
    def Train(self,max_epoch,val_freq,data_update_freq,data_update_ratio):
        epoch_ctr = 0
        val_ctr = 0
        self.GeneratePosDataset("train",self.max_train_memory_size)
        self.GeneratePosDataset("val",self.max_val_memory_size)
        while epoch_ctr < max_epoch:
            val_done = False
            for agent in self.agents:
                for dim in range(self.env_dim):
                    data_loader = self.GenerateDataLoader("train",agent,dim)
                    val = val_ctr
                    for i in range(data_update_freq):
                        val += 1
                        agent.Train(data_loader,dim)
                        if val >= val_freq:
                            val_data_loader = self.GenerateDataLoader("val",agent,dim)
                            agent.Validate(val_data_loader,dim)
                            val = 0
                            val_done = True

            val_ctr += data_update_freq
            if val_done:
                val_ctr = 0
            epoch_ctr += data_update_freq
            self.GeneratePosDataset("train",int(self.max_train_memory_size*data_update_ratio))

        print("End of training")
        for agent in self.agents:
            for dim in range(self.env_dim):
                val_data_loader = self.GenerateDataLoader("val",agent,dim)
                agent.Validate(val_data_loader,dim)

