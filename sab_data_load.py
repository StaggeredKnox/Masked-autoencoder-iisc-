"""
loading velocity datasets and sampling time and rzn slices at regular intervals
includes multi gpu datasampling as well


"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

from torch.utils.data.distributed import DistributedSampler

import numpy as np

import matplotlib.pyplot as plt

nT___ = -1
nR___ = -1
IMSZ = 0

def load_vel(data_path, config):
    nt=config.nt
    nr=config.nr

    all_u_mat = np.load(data_path +'all_u_mat.npy')
    
    tts = all_u_mat.shape[0]  # total time steps
    a = int(config.frac_1*(tts/config.kf))
    b = int(config.frac_2*(tts/config.kf))
    gt = int((b-a)/nt)  # gaps for time steps slicing
    if gt==0:
        gt=1
    
    all_u_mat = all_u_mat[a:b:gt]
    all_ui_mat = (np.load(data_path +'all_ui_mat.npy'))[a:b:gt]
    all_v_mat = (np.load(data_path +'all_v_mat.npy'))[a:b:gt]
    all_vi_mat = (np.load(data_path +'all_vi_mat.npy'))[a:b:gt]
    
    all_Yi = np.load(data_path +'all_Yi.npy')
    
    trzn = all_Yi.shape[1]  # total realizations
    gr = int(trzn/nr)  # gaps for rzn steps slicing

    all_Yi = all_Yi[a:b:gt, :trzn:gr] 

    global nT___, nR___
    nT___ = all_u_mat.shape[0]
    nR___ = all_Yi.shape[1]

    vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]

    global IMSZ
    IMSZ = config.image_size

    return vel_field_data


def extract_velocity(vel_field_data, t, rzn, as_image=True):
    nmodes =  vel_field_data[2].shape[1]
    vx = vel_field_data[0][t,:,:].copy()
    vy = vel_field_data[1][t,:,:].copy() 

    # vx1= 0
    # vy1= 0
    # vx1= vx1 + vx
    # vy1= vy1 + vy

    for m in range(nmodes):
        vx += vel_field_data[2][t, m, :, :]*vel_field_data[4][t, rzn,m]
        vy += vel_field_data[3][t, m, :, :] * vel_field_data[4][t, rzn,m]

    if as_image:
        im = np.stack([vx, vy], axis=0)
        return im
    else:
        return vx, vy
    
    
class VelocityDataset(Dataset):
    def __init__(self, vel_field_data, nr=-1, nt=-1):
        self.vel_field_data = vel_field_data
        global nT___, nR___, IMSZ
        self.IMSZ = IMSZ
        self.nt = nT___ if nT___!=-1 else nt
        self.nr = nR___ if nR___!=-1 else nr
        assert(self.nt!=-1 and self.nr!=-1)
        
    
    def __len__(self):
        return self.nt * self.nr
    
    def __getitem__(self, idx):
        rzn = idx // self.nt
        t = idx % self.nt
        im = extract_velocity(self.vel_field_data, t, rzn)
        
        sz= self.IMSZ   # image size  (256)
        im_tensor = torch.tensor(im)
        im_tensor = im_tensor.unsqueeze(0)  # Add batch dimension
        im_tensor = F.interpolate(im_tensor, size=(sz, sz), mode='bilinear', align_corners=False)     # 1,2,256,256
        im_tensor = im_tensor.squeeze(0)  # Remove batch dimension              
        
        return im_tensor   
    
    
def plot_vel_field(vx_grid, vy_grid, g_strmplot_lw=1, g_strmplot_arrowsize=1, flow_name="", path=""):
    # Make modes the last axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    vx_grid = np.flipud(vx_grid)
    vy_grid = np.flipud(vy_grid)
    xlim, ylim = vx_grid.shape
    Xs = np.arange(0,xlim) + 0.5
    Ys = np.arange(0,ylim) + 0.5
    X,Y = np.meshgrid(Xs, Ys)
    plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    im = plt.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.9, zorder = -1e5)

    plt.savefig(path+flow_name+".png")
    return im  
    
    
def GiveMe_loaders(*args, batch_size=16, path="", plot_dat=True, multi_gpu=False, cfg=0):
    N= len(args)
    data_sets= []
    
    for i in range(N):
        vel_field_data_i = load_vel(args[i], cfg)
        #print(f"{vel_field_data_i[0].shape}       {vel_field_data_i[1].shape}        {vel_field_data_i[2].shape}         {vel_field_data_i[3].shape}          {vel_field_data_i[4].shape}")
        dataset_i = VelocityDataset(vel_field_data_i)
        data_sets.append(dataset_i)
        if plot_dat:
            plot_vel_field(dataset_i.__getitem__(0)[0], dataset_i.__getitem__(0)[1], flow_name="loader_flow_"+str(i), path=path)
        
    dataset = ConcatDataset(data_sets)
    Len= len(dataset)
    
    assert((Len*(cfg.val_size + cfg.test_size + cfg.train_size) - Len)==0)
    
    g1 = torch.Generator().manual_seed(cfg.random_seed)
    
    train_dat, val_dat, test_dat= random_split(dataset, [cfg.train_size, cfg.val_size, cfg.test_size], generator=g1)

    if multi_gpu:
        # shuffle has to be kept false because shuffling is done by Dsampler
        train_dataloader = DataLoader(train_dat, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dat))
        val_dataloader = DataLoader(val_dat, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(val_dat))
    else:
        train_dataloader = DataLoader(train_dat, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dat, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, val_dataloader, val_dat, test_dat

    

# if __name__ == "__main__":

#     a = "/home/rohit/Documents/Research/data_prep/HDD_data/GenHW/"
#     data_path_1 = a+"GenHW_TV_DNV_dl_c83_m10_s20_f2_A1/"
#     data_path_2 = a+"GenHW_TV_DNV_dl_c-36_m20_s20_f2_A1/"
#     data_path_3 = a+"GenHW_TV_DNV_dl_c-17_m10_s20_f2_A1/"
#     data_path_4 = a+"GenHW_TV_DNV_dl_c0_m0_s20_f2_A1/"
#     data_path_5 = a+"GenHW_TV_DNV_dl_c50_m0_s20_f2_A1/"
#     data_path_6 = a+"GenHW_TV_DNV_dl_c64_m20_s20_f2_A1/"
#     data_path_7 = a+"GenHW_TV_DNV_dl_c14_m20_s20_f2_A1/"
#     data_path_8 = a+"GenHW_TV_DNV_dl_c33_m10_s20_f2_A1/"
#     data_path_9 = a+"GenHW_TV_DNV_dl_c42_m5_s20_f2_A1/"
#     data_path_10 = a+"GenHW_TV_DNV_dl_c54_m25_s20_f2_A1/"

#     loader_path = config["loader_path"]

#     # m= -80, +80, -45, +45, 0
#     # c= 10, 90, 35, 65

#     T_loader, V_loader = GiveMe_loaders(data_path_1, data_path_2, data_path_3, data_path_4, data_path_5, data_path_6, data_path_7, data_path_8, data_path_9, data_path_10,
#                                          val_size=0.05, batch_size=16, path=loader_path)



# END    