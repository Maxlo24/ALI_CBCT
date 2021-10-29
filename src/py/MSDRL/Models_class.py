
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import os

import torchvision
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from tqdm.std import tqdm
# from torchvision import models

class Brain:
    def __init__(
        self,
        network_type,
        network_nbr,
        device, 
        in_channels,
        in_size,
        out_channels,
        model_dir = "",
        model_name = "",
        run_dir = "",
        learning_rate = 1e-4,
        batch_size = 10,
        generate_tensorboard = False,
        verbose = False
    ) -> None:
        self.network_type = network_type
        self.in_channels = in_channels
        self.in_size = in_size
        self.out_channels = out_channels

        self.verbose = verbose
        self.generate_tensorboard = generate_tensorboard
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        networks = []
        global_epoch = []
        epoch_losses = []
        validation_metrics = []
        models_dirs = []

        writers = []
        optimizers = []
        best_metrics = []
        best_epoch = []

        for n in range(network_nbr):
            net = network_type(
                in_channels = in_channels,
                in_size = in_size,
                out_channels = out_channels,
            )
            net.to(self.device)
            networks.append(net)
            optimizers.append(optim.Adam(net.parameters(), lr=learning_rate))
            epoch_losses.append([0])
            validation_metrics.append([])
            best_metrics.append(0)
            global_epoch.append(0)
            best_epoch.append(0)

            if not model_dir == "":
                dir_path = os.path.join(model_dir,str(n))
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                models_dirs.append(dir_path)
            
            if self.generate_tensorboard:
                run_path = os.path.normpath("/".join([os.path.dirname(os.path.dirname(model_dir)),str(os.path.basename(os.path.dirname(model_dir)))+"_Runs",os.path.basename(model_dir),str(n)]))
                if not os.path.exists(run_path):
                    os.makedirs(run_path)
                writers.append(SummaryWriter(run_path))


        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizers = optimizers
        self.writers = writers

        self.networks = networks
        # self.networks = [networks[0]]
        self.epoch_losses = epoch_losses
        self.validation_metrics = validation_metrics
        self.best_metrics = best_metrics
        self.global_epoch = global_epoch
        self.best_epoch = best_epoch

        self.model_dirs = models_dirs
        self.model_name = model_name


    def ResetNet(self,n):
        net = self.network_type(
            in_channels = self.in_channels,
            in_size = self.in_size,
            out_channels = self.out_channels,
        )
        net.to(self.device)
        self.networks[n] = net
        self.optimizers[n] = optim.Adam(net.parameters(), lr=self.learning_rate)
        self.epoch_losses[n] = [0]
        self.validation_metrics[n] = []
        self.best_metrics[n] = 0
        self.global_epoch[n] = 0
        self.best_epoch[n] = 0


    def Predict(self,dim,state):
        network = self.networks[dim]
        network.eval()
        with torch.no_grad():
            input = torch.unsqueeze(state,0).type(torch.float32).to(self.device)
            x = network(input)
        return torch.argmax(x)

    def Train(self,data,n):
        # print(data)
        # for n,network in enumerate(self.networks):
        network = self.networks[n]
        self.global_epoch[n] += 1   
        if self.verbose:
            print("training epoch:",self.global_epoch[n],"for network :",n)

        network.train()
        epoch_loss = 0
        epoch_good_move = 0
        epoch_iterator = tqdm(
            data, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        optimizer = self.optimizers[n]
        step=0
        for step, batch in enumerate(epoch_iterator):
            # print(batch["state"].size(),batch["target"].size())
            # print(torch.min(batch["state"]),torch.max(batch["state"]) , batch["state"].type())
            input,target = batch["state"].type(torch.float32).to(self.device),batch["target"].to(self.device)
            
            
            # img_grid = torchvision.utils.make_grid(batch["state"])
            # self.writer.add_image('Crop of network '+str(n)+' at epoch ' + str(self.global_epoch[n]),img_grid)
            
            optimizer.zero_grad()
            y = network(input) 
            loss = self.loss_fn(y,target)
            loss.backward()
            optimizer.step()
            epoch_loss +=loss.item()
            for i in range(self.batch_size):
                if torch.eq(torch.argmax(y[i]),target[i]):
                    epoch_good_move +=1
            epoch_iterator.set_description(
                "Training loss=%2.5f" % (loss)
            )

        epoch_loss /= step+1
        metric = epoch_good_move/((step+1)*self.batch_size)
        
        if self.epoch_losses[n][-1] == epoch_loss: #If learning is stuck
            self.ResetNet(n)
            print()
            print("Stuck at Loss :",epoch_loss)
            print("Net reset")
            print("-------------------------------------------------")
        else:
            self.epoch_losses[n].append(epoch_loss)
            if self.verbose:
                print()
                print("Average epoch Loss :",epoch_loss)
                print("Porcentage of good moves :",metric*100,"%")
                print("-------------------------------------------------")

            if self.generate_tensorboard:
                writer = self.writers[n]
                writer.add_scalar("Training loss",epoch_loss,self.global_epoch[n])
                writer.add_scalar("Training accuracy",metric,self.global_epoch[n])
                writer.close()


    def Validate(self,data,n):
        # print(data)
        # for n,network in enumerate(self.networks):
        if self.verbose:
            print("validating network :",n)
        
        network = self.networks[n]
        network.eval()
        with torch.no_grad():
            running_loss = 0
            good_move = 0
            epoch_iterator = tqdm(
                data, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True
            )
            for step, batch in enumerate(epoch_iterator):
                
                # print(batch["state"].size(),batch["target"].size())
                # print(torch.min(batch["state"]),torch.max(batch["state"]))
                input,target = batch["state"].type(torch.float32).to(self.device),batch["target"].to(self.device)

                y = network(input)
                loss = self.loss_fn(y,target)

                for i in range(self.batch_size):
                    if torch.eq(torch.argmax(y[i]),target[i]):
                        good_move +=1

                running_loss +=loss.item()
                epoch_iterator.set_description(
                    "Validating loss=%2.5f)" % (loss)
                )

            # running_loss /= step+1
            metric = good_move/((step+1)*self.batch_size)

            self.validation_metrics[n].append(metric)

            if self.verbose:
                print()
                print("Porcentage of good moves :",metric*100,"%")

            if metric > self.best_metrics[n]:
                self.best_metrics[n] = metric
                self.best_epoch[n] = self.global_epoch[n]
                save_path = os.path.join(self.model_dirs[n],self.model_name+"_Net_"+str(n)+".pth")
                torch.save(
                    network.state_dict(), save_path
                )
                # data_model["best"] = save_path
                print("Model Was Saved ! Current Best Avg. metric: {} Current Avg. metric: {}".format(self.best_metrics[n], metric))
            else:
                print("Model Was Not Saved ! Current Best Avg. metric: {} Current Avg. metric: {}".format(self.best_metrics[n], metric))
        print("-------------------------------------------------")
        if self.generate_tensorboard:
            writer = self.writers[n]
            writer.add_graph(network,input)
            writer.add_scalar("Validation accuracy",metric,self.global_epoch[n])
            writer.close()

    def LoadModels(self,model_lst):
        for n,net in enumerate(self.networks):
            print("Loading model", model_lst[n])
            net.load_state_dict(torch.load(model_lst[n],map_location=self.device))

# #####################################
#  Networks
# #####################################

class DQN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        in_size = [64,64,64],
        out_channels: int = 6,
        dropout_rate: float = 0.0,
    ) -> None:
        super(DQN, self).__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        # self.norm = nn.LayerNorm(in_size)
        self.conv1 = nn.Conv3d(
            in_channels, 
            out_channels = 32, 
            kernel_size = 4, 
        )
        self.pool1 = nn.AvgPool3d(2)
        self.conv2 = nn.Conv3d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = 3,
        )
        self.pool2 = nn.AvgPool3d(2)
        self.conv3 = nn.Conv3d(
            in_channels = 64, 
            out_channels = 128, 
            kernel_size = 2,
        )
        self.pool3 = nn.AvgPool3d(2)

        self.fc0 = nn.Linear(128*6*6*6,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_channels)


        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self,x):
        # print(x.size())
        # x = self.norm(x)
        x=self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x=self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x=self.pool3(F.relu(self.conv3(x)))
        # print(x.size())
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = x #F.softmax(self.fc3(x), dim=1)
        return output



class MaxDQN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        in_size = [64,64,64],
        out_channels: int = 6,
        dropout_rate: float = 0.0,
    ) -> None:
        super(MaxDQN, self).__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        # self.norm = nn.LayerNorm(in_size)
        self.conv1 = nn.Conv3d(
            in_channels, 
            out_channels = 32, 
            kernel_size = 4, 
        )
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = 3,
        )
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(
            in_channels = 64, 
            out_channels = 128, 
            kernel_size = 2,
        )
        self.pool3 = nn.MaxPool3d(2)

        self.fc0 = nn.Linear(128*6*6*6,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_channels)

    def forward(self,x):
        # print(x.size())
        # x = self.norm(x)
        x=self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x=self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x=self.pool3(F.relu(self.conv3(x)))
        # print(x.size())
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = x #F.softmax(self.fc3(x), dim=1)
        return output



class DRLnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super(DRLnet, self).__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.conv1 = nn.Conv3d(
            in_channels, 
            out_channels = 32, 
            kernel_size = 4, 
        )
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(
            in_channels = 32, 
            out_channels = 42, 
            kernel_size = 3, 
        )
        self.pool2 = nn.MaxPool3d(2)

        self.fc0 = nn.Linear(115248,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)

        # self.pool1 = nn.MaxPool3d(kernel_size = 2)

    def forward(self,x):
        # print(x.size())
        x=self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x=self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output


