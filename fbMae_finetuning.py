"""
for finetuning fbMAE pretrained model
"""

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

from src.Class_fbMae_preTrained import fbMae_model

from src.Class_optimsAndScheds import Optims_Scheds

import wandb

from datetime import datetime

from cfg.config_fbMae import hpt_config
from cfg.config_fbMae import config as sr_config

from get_data_names import get_names


import argparse
ARGS_MODE = None


wandb.login()

MODEL_NAME = ""  # to save best model name


def unmasked_ids(mask_space):
    Len = len(mask_space)
    un_ids = []
    for i in range(Len):
        if mask_space[i]==0:
            un_ids.append(i)
    un_ids = np.array(un_ids)
    un_ids = torch.from_numpy(un_ids)
    un_ids.unsqueeze_(0)
    return un_ids



def test_reconstruction_plot(dataset, mae, n_samples=10, fname="fbMae_test_sample_", path=""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    szImg= wandb.config.image_size
    p12= wandb.config.patch_size  # patch size (our case always remains square so total pixels=p12*p12*c)
    hw= szImg//p12  #number of such (p12 X p12) images along height and width
    
    for i in range(n_samples):
        r= random.randint(0, len(dataset)-1)
        image_r = dataset.__getitem__(r)
        image_r = torch.reshape(image_r,(1,2,szImg,szImg))

        shp = image_r.shape   # padding a ZERO-PIXELLED image channel
        zero_pad = torch.zeros(shp[0], 1, *shp[2:])
        image_r = torch.cat((image_r, zero_pad), dim=1)

        image_patch = rearrange(image_r,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', h = hw, w = hw, p1 = p12, p2 = p12, c = 3)
        unmasked_img = np.zeros((1, hw*hw, p12*p12*3))


        # as we just need to get inferences out of our model
        mae.eval()
        with torch.no_grad():
            outputs = mae(image_r.to(device), padded=True)
            re_image_r = outputs.logits  #b.n.p1*p2*c    (1.256.512)
            unmasked_ind = unmasked_ids(outputs.mask[0])


        re_image_r = re_image_r.cpu().detach().numpy()
        ind_list = unmasked_ind[0].cpu().numpy()  


        unmasked_img = np.zeros((1, hw*hw, p12*p12*3))
        for ind in ind_list:
            re_image_r[:,ind,:] = image_patch[:,ind,:]
            unmasked_img[:,ind,:] = image_patch[:,ind,:]


        re_img = rearrange(re_image_r, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 3)
        unmasked_img = rearrange(unmasked_img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 3)    

        re_img = rearrange(re_image_r, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 3)

        plt.clf()
        plot_vel_field(re_img[0][0], re_img[0][1], flow_name=fname+str(i), path=path)
        wandb.log({"TEST_reconstructed_velocity_field": wandb.Image(path+fname+str(i)+".png")})



def train_reconstruction_plot(dataset, mae, i, sth=0, fname="fbMae_Validation_flow_(", path="", ep=99, st_eps=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    szImg= wandb.config.image_size
    p12= wandb.config.patch_size  # patch size (our case always remains square so total pixels=p12*p12*c)
    hw= szImg//p12  #number of such (p12 X p12) images along height and width
    
    image_r = dataset.__getitem__(sth)
    image_r = torch.reshape(image_r, (1, 2, szImg, szImg))

    shp = image_r.shape   # padding a ZERO-PIXELLED image channel
    zero_pad = torch.zeros(shp[0], 1, *shp[2:])
    image_r = torch.cat((image_r, zero_pad), dim=1)

    image_patch = rearrange(image_r,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', h = hw, w = hw, p1 = p12, p2 = p12, c = 3)
    unmasked_img = np.zeros((1, hw*hw, p12*p12*3))


    # as we just need to get inferences out of our model
    mae.eval()
    with torch.no_grad():
        outputs = mae(image_r.to(device), padded=True)
        re_image_r = outputs.logits  #b.n.p1*p2*c    (1.256.512)
        unmasked_ind = unmasked_ids(outputs.mask[0])
        # print(f"{re_image_r.shape}      {unmasked_ind.shape}")
        # print(outputs.mask.shape)
        a = outputs.hidden_states
        #print(f"{len(a)}     {a[0].shape}      {a[1].shape}      {a[2].shape}     {a[3].shape}     {a[4].shape}")
        #print(a[0])


    re_image_r = re_image_r.cpu().detach().numpy()
    ind_list = unmasked_ind[0].cpu().numpy()  


    unmasked_img = np.zeros((1, hw*hw, p12*p12*3))
    for ind in ind_list:
        re_image_r[:,ind,:] = image_patch[:,ind,:]
        unmasked_img[:,ind,:] = image_patch[:,ind,:]


    re_img = rearrange(re_image_r, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 3)
    unmasked_img = rearrange(unmasked_img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 3)    


    re_img = rearrange(re_image_r, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = hw, w = hw, p1 = p12, p2 = p12, c = 3)
    
    # Save at every 5th epoch
    plt.clf()
    tname = fname+str(i)+")dataset_("+str(sth)+")sample"

    if ep==st_eps:
        plot_vel_field(dataset.__getitem__(sth)[0], dataset.__getitem__(sth)[1], flow_name=tname, path=path)
        wandb.log({"TRAIN_original_velocity_field_("+str(i)+")dataset": wandb.Image(path+tname+".png")})
        
    plot_vel_field(re_img[0][0], re_img[0][1], flow_name=tname, path=path)
    wandb.log({"TRAIN_reconstructed_velocity_field_("+str(i)+")dataset": wandb.Image(path+tname+".png")})






def train_timm_mae(mae, train_dataloader, val_dataloader, data_sets, eps=5, nth=0, path="", optimizer=0, scheduler=0, start_eps=0, BestValLoss=10e10, BestEpoch=-9):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optimizer
    scheduler = scheduler

    mae.to(device)
    print("FineTuning the model (PreTrained on ImageNet-1K) --->")
    if device.type == "cuda":
        print("Using CUDA")
    else:
        print("Not using CUDA")
        
    train_loss_epochwise = []
    val_loss_epochwise = []

    bestValLoss = BestValLoss   # must be min of all 
    bestEpoch = BestEpoch

    for epoch in range(start_eps, eps):
        TLoss= 0
        N= 0
        for _, batch in enumerate(train_dataloader):
            images = batch.to(device)
            optimizer.zero_grad()
            outputs = mae(images)
            train_loss = outputs.loss
            train_loss.backward()
            nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                N= N+1
                TLoss += train_loss
        scheduler.step(train_loss)
        TLoss /= N

        VLoss= 0
        N= 0
        for _, batch in enumerate(val_dataloader):
            images = batch.to(device)
            with torch.no_grad():
                outputs = mae(images)
                val_loss = outputs.loss
                N= N+1
                VLoss += val_loss
        VLoss /= N

        global MODEL_NAME
        if bestValLoss>VLoss:    # best val loss computation
            bestValLoss=VLoss
            bestEpoch=epoch
            MODEL_NAME = path+"Bestuned_fbMae_Model_"+str(len(data_sets))+"D_"+str(eps)+"E"+".pt"
            torch.save(mae, MODEL_NAME)

        train_loss_epochwise.append(TLoss.item())
        val_loss_epochwise.append(VLoss.item())
        lr = optimizer.param_groups[0]["lr"]

        print(f"epoch: {epoch}    Train Loss: {TLoss.item()}    Validation Loss: {VLoss.item()}    Learning Rate: {lr}")
        #print(f"{a}   {b}   {c}   {d}")

        logDict= {
            "Epoch": epoch,
            "Train Loss": TLoss.item(),
            "Validation Loss": VLoss.item(),
            "Best Val Loss": bestValLoss,
            "learning rate": lr, #if scheduler.is_available() else lr
            "Best epoch": bestEpoch
        }
        wandb.log(logDict)

        model_dict = {
            "epoch": epoch,
            "model_state_dict": mae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "learning_rate": lr,
            "scheduler": scheduler.state_dict(),
            "best_val_loss": bestValLoss,
            "best_epoch": bestEpoch,
            "VLoss": VLoss,
            "TLoss": TLoss
        }
        torch.save(model_dict, path+"checkpoints_fbMae_model.pt")

        #for i in range(len(data_sets)):
        if epoch%nth==0 or epoch==eps:
            train_reconstruction_plot(data_sets, mae, 0, sth=0, path=path, ep=epoch, st_eps=start_eps)
            train_reconstruction_plot(data_sets, mae, 1, sth=len(data_sets)>>1, path=path, ep=epoch, st_eps=start_eps)
            train_reconstruction_plot(data_sets, mae, 2, sth=len(data_sets)-1, path=path, ep=epoch, st_eps=start_eps)


    train_loss_epochwise = [loss/(max(train_loss_epochwise)) for loss in train_loss_epochwise]
    val_loss_epochwise = [loss/(max(val_loss_epochwise)) for loss in val_loss_epochwise]

    plt.clf()
    plt.plot([i+1 for i in range(eps-start_eps)], train_loss_epochwise, color='g', label='Train Loss')
    plt.plot([i+1 for i in range(eps-start_eps)], val_loss_epochwise, color='r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Epochwise')

    plt.savefig(path+"fbMAE_tv_loss_"+str(len(data_sets))+"D_"+str(eps-start_eps)+"E"+".png")

    return train_loss_epochwise, val_loss_epochwise






def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    st = datetime.now().replace(microsecond=0)
    st = st.strftime("%d-%m %H-%M")

    if ARGS_MODE == 'single_run':        # for SINGLE run
        config = sr_config
        exp_name = "fbMae_fineTuning_single_88m " + st 
        wandb.init(
            project= "s'jar",
            name= exp_name,
            config=config
        )
    else:                                # for SWEEP run
        exp_name = "fbMae_fineTuning_multiple_88m " + st 
        wandb.init(
            name= exp_name,
        )
    
    
    cfg=wandb.config
    
    datpaths = get_names(data_path = cfg.Train_data_path)
    train_dataloader, val_dataloader, data_sets, test_dat = GiveMe_loaders(*datpaths, 
                                       batch_size=cfg.batch_size, path=cfg.fbMae_path, plot_dat=cfg.plot_dat, cfg=cfg)



    # Initializing a model 
    model = fbMae_model( 
        pixel_scale=cfg.scl
        )

    # Accessing the model configuration
    model_config = model.config
    #print(model_config)


    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params


    # train_param
    obj_OS = Optims_Scheds(model=model, optim=cfg.optimizer, sched=cfg.scheduler, cfg=cfg)
    obj_OS.assign()
    optim = obj_OS.optimizer()
    schedul = obj_OS.scheduler()

    train_loss_epochwise, val_loss_epochwise = train_timm_mae(model, train_dataloader, val_dataloader, data_sets, 
                                                              eps=cfg.epochs, nth=cfg.nth, path=cfg.fbMae_path,
                                                              optimizer=optim, scheduler=schedul)


    # After train, testing on entirely different dataset
    global MODEL_NAME
    bestModel= torch.load(MODEL_NAME).to(device)
    bestModel.eval()

    with torch.no_grad():

        vel_field_data = load_vel(cfg.aftrain_testDatpath, cfg)
        dataset = VelocityDataset(vel_field_data)
        plt.clf()
        plot_vel_field(dataset.__getitem__(0)[0], dataset.__getitem__(0)[1], flow_name="aftrain_original", path=cfg.fbMae_path)
        wandb.log({"aftrain_original_sample": wandb.Image(cfg.fbMae_path+"aftrain_original.png")})
        test_reconstruction_plot(dataset, bestModel, n_samples=cfg.n_samples, fname="fbMae_aftrain_test_sample", path=cfg.fbMae_path)

        plt.clf()
        plot_vel_field(test_dat.__getitem__(0)[0], test_dat.__getitem__(0)[1], flow_name="test_original", path=cfg.fbMae_path)
        wandb.log({"test_original_sample": wandb.Image(cfg.fbMae_path+"test_original.png")})
        test_reconstruction_plot(test_dat, bestModel, n_samples=cfg.n_samples, fname="fbMae_test_sample", path=cfg.fbMae_path)





if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single_run')
    args = parser.parse_args()

    ARGS_MODE = args.mode

    if ARGS_MODE=='single_run':
        main()
    else:
        sweep_id = wandb.sweep(sweep=hpt_config, project="s'jar")
        wandb.agent(sweep_id, function=main, count=2)


# END        

