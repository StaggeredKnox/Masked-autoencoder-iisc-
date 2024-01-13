import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt

import numpy as np

import random

from einops import rearrange, repeat

from sab_data_load import plot_vel_field, GiveMe_loaders, load_vel, VelocityDataset

from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining

from timm_mae import train_timm_mae as tmTRAIN
from timm_mae import test_reconstruction_plot as tmPLOT

from fbMae_finetuning import train_timm_mae as fbTRAIN
from fbMae_finetuning import test_reconstruction_plot as fbPLOT

from get_data_names import get_names

import wandb

from datetime import datetime

from cfg.config_fbMae import config as fbconfig 
from cfg.config_timm import config as tmconfig

from src.Class_optimsAndScheds import Optims_Scheds

import argparse
ARGS_MODE = None




def main():

    a = ARGS_MODE=="finetuned"
    b = ARGS_MODE=="trained"

    assert(a | b)
    if a:
        config = fbconfig
    elif b:
        config = tmconfig


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st = datetime.now().replace(microsecond=0)
    st = st.strftime("%d-%m %H-%M")
    exp_name = "Continuing_"+config["continue_model"]+"_training " + st 

    wandb.login()
    wandb.init(
            project= "s'jar",
            config= config,
            name= exp_name,
        )
    cfg = wandb.config


    # loading data
    datpaths = get_names(data_path = cfg.Train_data_path)
    train_dataloader, val_dataloader, data_sets, test_dat = GiveMe_loaders(*datpaths, 
                                       batch_size=cfg.batch_size, path=cfg.cont_path, plot_dat=cfg.plot_dat, cfg=cfg)
    

    # model and state loading
    dummy_model_path = cfg.cont_path+cfg.continue_model+".pt"
    checkpoint_path = cfg.cont_path+cfg.checkpoints_name+".pt"
    model = torch.load(dummy_model_path).to(device)
    checkpoint = torch.load(checkpoint_path)


    # defining optimizer and its scheduler
    lr = checkpoint['learning_rate']
    obj_OS = Optims_Scheds(model=model, optim=cfg.optimizer, sched=cfg.scheduler, cfg=cfg)
    obj_OS.assign()
    optimizer = obj_OS.optimizer()
    scheduler = obj_OS.scheduler()


    # assigning model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.param_groups[0]["lr"] = lr
    scheduler.load_state_dict(checkpoint['scheduler'])
    st_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    TLoss = checkpoint['TLoss']
    VLoss = checkpoint['VLoss']

    print(f"continuing model from --> \n Epoch:{st_epoch} \n Train_loss:{TLoss} \t Validation_loss:{VLoss} \n Best_Val_loss: {best_val_loss} \t Best_Epoch: {best_epoch} \n learning_rate:{lr}")


    # training of model
    model.train()


    if ARGS_MODE=="trained":
        train_loss_epochwise, val_loss_epochwise = tmTRAIN(model, train_dataloader, val_dataloader, data_sets, 
                                                              eps=cfg.epochs, nth=cfg.nth, path=cfg.cont_path,
                                                              optimizer=optimizer, scheduler=scheduler, 
                                                              start_eps=st_epoch+1, BestValLoss=best_val_loss, BestEpoch=best_epoch)
        # After train, testing on entirely different dataset
        model_name = cfg.cont_path+cfg.load_best_model+".pt"
        bestModel= torch.load(model_name).to(device)
        bestModel.eval()

        with torch.no_grad():

            vel_field_data = load_vel(cfg.aftrain_testDatpath, cfg)
            dataset = VelocityDataset(vel_field_data)
            plt.clf()
            plot_vel_field(dataset.__getitem__(0)[0], dataset.__getitem__(0)[1], flow_name="aftrain_original", path=cfg.cont_path)
            wandb.log({"aftrain_original_sample": wandb.Image(cfg.cont_path+"aftrain_original.png")})
            tmPLOT(dataset, bestModel, n_samples=cfg.n_samples, fname="timm_aftrain_test_sample", path=cfg.cont_path)
            
            plt.clf()
            plot_vel_field(test_dat.__getitem__(0)[0], test_dat.__getitem__(0)[1], flow_name="test_original", path=cfg.cont_path)
            wandb.log({"test_original_sample": wandb.Image(cfg.cont_path+"test_original.png")})
            tmPLOT(test_dat, bestModel, n_samples=cfg.n_samples, fname="timm_test_sample", path=cfg.cont_path)

    else:
        
        train_loss_epochwise, val_loss_epochwise = fbTRAIN(model, train_dataloader, val_dataloader, data_sets, 
                                                              eps=cfg.epochs, nth=cfg.nth, path=cfg.cont_path,
                                                              optimizer=optimizer, scheduler=scheduler, 
                                                              start_eps=st_epoch+1, BestValLoss=best_val_loss, BestEpoch=best_epoch)
        # After train, testing on entirely different dataset
        model_name = cfg.cont_path+cfg.load_best_model+".pt"
        bestModel= torch.load(model_name).to(device)
        bestModel.eval()

        with torch.no_grad():

            vel_field_data = load_vel(cfg.aftrain_testDatpath, cfg)
            dataset = VelocityDataset(vel_field_data)
            plt.clf()
            plot_vel_field(dataset.__getitem__(0)[0], dataset.__getitem__(0)[1], flow_name="aftrain_original", path=cfg.cont_path)
            wandb.log({"aftrain_original_sample": wandb.Image(cfg.cont_path+"aftrain_original.png")})
            fbPLOT(dataset, bestModel, n_samples=cfg.n_samples, fname="fbMae_aftrain_test_sample", path=cfg.cont_path)

            plt.clf()
            plot_vel_field(test_dat.__getitem__(0)[0], test_dat.__getitem__(0)[1], flow_name="test_original", path=cfg.cont_path)
            wandb.log({"test_original_sample": wandb.Image(cfg.cont_path+"test_original.png")})
            fbPLOT(test_dat, bestModel, n_samples=cfg.n_samples, fname="fbMae_test_sample", path=cfg.cont_path)






if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='finetuned')
    args = parser.parse_args()

    ARGS_MODE = args.mode

    main()


# END

