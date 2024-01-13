"""
For loading trained and saved models and tesing on arbitrary datasets
"""

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from src.Class_fbMae_preTrained import fbMae_model
from src.Class_VitMae_MOD import VitMae_model

from sab_data_load import VelocityDataset, load_vel, plot_vel_field
from fbMae_finetuning import test_reconstruction_plot as fbPLOT
from timm_mae import test_reconstruction_plot as tmPLOT

import wandb

from datetime import datetime

from cfg.config_testing import config

import random

device= torch.device("cuda" if (torch.cuda.is_available()) else "cpu")



if __name__ == "__main__":

    st = datetime.now().replace(microsecond=0)
    st = st.strftime("%d-%m %H-%M")
    exp_name = "Testing_mae " + st 

    wandb.login()
    wandb.init(
            project= "s'jar",
            config= config,
            name= exp_name,
        )
    
    cfg = wandb.config

    data_path = cfg.test_on_datapath

    vel_field_data = load_vel(data_path, cfg)

    dataset = VelocityDataset(vel_field_data)

    mae= torch.load(cfg.test_model_path).to(device)

    plt.clf()
    path=cfg.test_path
    plot_vel_field(dataset.__getitem__(0)[0], dataset.__getitem__(0)[1], flow_name="original", path=path)
    wandb.log({"TEST_reconstructed_velocity_field": wandb.Image(path+"original.png")})

    if cfg.test_model_class=="finetuned":
        fbPLOT(dataset, mae, n_samples=cfg.test_n_samples, fname="fbMae_TESTING_", path=cfg.test_path)
    elif cfg.test_model_class=="trained":
        tmPLOT(dataset, mae, n_samples=cfg.test_n_samples, fname="vitMae_TESTING_", path=cfg.test_path)


# END

