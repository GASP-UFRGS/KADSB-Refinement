from collections import OrderedDict
from copy import deepcopy
import sys
from sys import stderr
import time
from pathlib import Path
import gc
from functools import reduce
import operator as op

import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


def grad_gauss(x, m, var):
    xout = (x - m) / var
    return -xout


def ornstein_ulhenbeck(x, gradx, gamma):
    xout = x + gamma * gradx + \
        torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
    return xout


def get_timestep_embedding(timesteps, embedding_dim=128):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)

    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb


def load_data(file_path_gflash, file_path_g4, normalize_energy=True, shuffle=True, plotting=False):

    energy_voxel_g4 = np.load(file_path_g4)[:, 0:100].astype(np.float32)
    energy_voxel_gflash  = np.load(file_path_gflash)[:, 0:100].astype(np.float32)

    energy_particle_g4 = np.load(file_path_g4)[:, 200:201].astype(np.float32)/10000.0
    energy_particle_gflash  = np.load(file_path_gflash)[:, 200:201].astype(np.float32)/10000.0

    if shuffle:
        # sort by incident energy to define pairs
        mask_energy_particle_g4 = np.argsort(energy_particle_g4, axis=0)[:,0]
        mask_energy_particle_gflash = np.argsort(energy_particle_gflash, axis=0)[:,0]

        energy_particle_g4 = energy_particle_g4[mask_energy_particle_g4]
        energy_particle_gflash = energy_particle_gflash[mask_energy_particle_gflash]

        energy_voxel_g4 = energy_voxel_g4[mask_energy_particle_g4]
        energy_voxel_gflash = energy_voxel_gflash[mask_energy_particle_gflash]

        # reshuffle consistently
        mask_shuffle = np.random.permutation(energy_particle_g4.shape[0])

        energy_particle_g4 = energy_particle_g4[mask_shuffle]
        energy_particle_gflash = energy_particle_gflash[mask_shuffle]

        energy_voxel_g4 = energy_voxel_g4[mask_shuffle]
        energy_voxel_gflash = energy_voxel_gflash[mask_shuffle]

    if plotting:
        if energy_particle_gflash.shape[1] == 0:
            energy_particle_g4 = np.ones((energy_voxel_g4.shape[0], 1)).astype(np.float32)/10000.0*50000.0
            energy_particle_gflash  = np.ones((energy_particle_gflash.shape[0], 1)).astype(np.float32)/10000.0*50000.0

    energy_g4 = np.sum(energy_voxel_g4, 1, keepdims=True)
    energy_gflash = np.sum(energy_voxel_gflash, 1, keepdims=True)

    energy_voxel_g4 = np.reshape(energy_voxel_g4, (-1, 1, 10, 10))
    energy_voxel_gflash = np.reshape(energy_voxel_gflash, (-1, 1, 10, 10))

    energy_voxel_g4 = energy_voxel_g4/np.tile(np.reshape(energy_g4, (-1, 1, 1, 1)), (1, 1, 10, 10))
    energy_voxel_gflash = energy_voxel_gflash/np.tile(np.reshape(energy_gflash, (-1, 1, 1, 1)), (1, 1, 10, 10))

    #--------------------------------------------------------------------------------------------------------#

    shifter_energy_fullrange_g4 = np.mean(energy_g4, 0)
    shifter_energy_fullrange_gflash = np.mean(energy_gflash, 0)
    scaler_energy_fullrange_g4 = np.std(energy_g4)
    scaler_energy_fullrange_gflash = np.std(energy_gflash, 0)

    if normalize_energy:
        energy_g4 = energy_g4/energy_particle_g4
        energy_gflash = energy_gflash/energy_particle_gflash

    shifter_g4 = np.mean(energy_voxel_g4, 0)
    shifter_gflash = np.mean(energy_voxel_gflash, 0)
    scaler_g4 = np.std(energy_voxel_g4, 0)
    scaler_gflash = np.std(energy_voxel_gflash, 0)

    energy_voxel_g4 = (energy_voxel_g4 - shifter_g4)/scaler_g4
    energy_voxel_gflash = (energy_voxel_gflash - shifter_gflash)/scaler_gflash

    shifter_energy_g4 = np.mean(energy_g4, 0)
    shifter_energy_gflash = np.mean(energy_gflash, 0)
    scaler_energy_g4 = np.std(energy_g4, 0)
    scaler_energy_gflash = np.std(energy_gflash, 0)

    energy_g4 = (energy_g4 - shifter_energy_g4)/scaler_energy_g4
    energy_gflash = (energy_gflash - shifter_energy_gflash)/scaler_energy_gflash

    return {"energy_gflash":energy_gflash, "energy_particle_gflash":energy_particle_gflash, "energy_voxel_gflash":energy_voxel_gflash,
            "energy_g4":energy_g4, "energy_particle_g4":energy_particle_g4, "energy_voxel_g4":energy_voxel_g4,
            "shifter_gflash":shifter_gflash, "scaler_gflash":scaler_gflash, "shifter_g4":shifter_g4, "scaler_g4":scaler_g4,
            "shifter_energy_gflash":shifter_energy_gflash, "scaler_energy_gflash":scaler_energy_gflash,
            "shifter_energy_g4":shifter_energy_g4, "scaler_energy_g4":scaler_energy_g4,
            "shifter_energy_fullrange_gflash":shifter_energy_fullrange_gflash, "scaler_energy_fullrange_gflash":scaler_energy_fullrange_gflash,
            "shifter_energy_fullrange_g4":shifter_energy_fullrange_g4, "scaler_energy_fullrange_g4":scaler_energy_fullrange_g4}


def load_data_plots(file_path_g4, file_path_gflash):

    energy_voxel_g4 = np.load(file_path_g4)[:, 0:100].astype(np.float32)
    energy_voxel_gflash  = np.load(file_path_gflash)[:, 0:100].astype(np.float32)

    energy_particle_g4 = np.load(file_path_g4)[:, 200:201].astype(np.float32)/10000.0
    energy_particle_gflash  = np.load(file_path_gflash)[:, 200:201].astype(np.float32)/10000.0

    if energy_particle_gflash.shape[1] == 0:
        energy_particle_g4 = np.ones((energy_voxel_g4.shape[0], 1)).astype(np.float32)/10000.0*50000.0
        energy_particle_gflash  = np.ones((energy_particle_gflash.shape[0], 1)).astype(np.float32)/10000.0*50000.0

    energy_g4 = np.sum(energy_voxel_g4, 1, keepdims=True)
    energy_gflash = np.sum(energy_voxel_gflash, 1, keepdims=True)

    energy_voxel_g4 = np.reshape(energy_voxel_g4, (-1, 1, 10, 10))
    energy_voxel_gflash = np.reshape(energy_voxel_gflash, (-1, 1, 10, 10))

    energy_voxel_g4 = energy_voxel_g4/np.tile(np.reshape(energy_g4, (-1, 1, 1, 1)), (1, 1, 10, 10))
    energy_voxel_gflash = energy_voxel_gflash/np.tile(np.reshape(energy_gflash, (-1, 1, 1, 1)), (1, 1, 10, 10))

    energy_g4 = energy_g4/energy_particle_g4
    energy_gflash = energy_gflash/energy_particle_gflash

    return {"energy_voxel_g4":energy_voxel_g4, "energy_voxel_gflash":energy_voxel_gflash,
            "energy_g4":energy_g4, "energy_gflash":energy_gflash}


### https://www.zijianhu.com/post/pytorch/ema/
class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        
        if self.training:
            return self.model(*args, **kwargs)
        else:
            return self.shadow(*args, **kwargs)
        

class CacheLoader(Dataset):
    def __init__(self, nets, device, dls, gammas, npar, batch_size, num_steps, d, dy, mean_final, var_final,
                 forward_or_backward='f', forward_or_backward_rev='b', first=False, sample=False):
        super().__init__()
        self.num_batches = int(npar/batch_size)

        self.data = torch.zeros((self.num_batches, batch_size*num_steps, 2, *d)).to(device)  # .cpu()
        self.y_data = torch.zeros((self.num_batches, batch_size*num_steps, *dy)).to(device)  # .cpu()
        self.steps_data = torch.zeros((self.num_batches, batch_size*num_steps, 1)).to(device)  # .cpu() # steps



        for b, dat in enumerate(dls[forward_or_backward]):    
            #print(b, self.num_batches)
            
            if b == self.num_batches:
                break

            x = dat[0].float().to(device)
            x_orig = x.clone().to(device)
            y = dat[1].float().to(device)
            steps = torch.arange(num_steps).to(device)
            time = torch.cumsum(gammas, 0).to(device).float()


            N = x.shape[0]
            steps = steps.reshape((1, num_steps, 1)).repeat((N, 1, 1))
            time = time.reshape((1, num_steps, 1)).repeat((N, 1, 1))
            #gammas_new = gammas.reshape((1, num_steps, 1)).repeat((N, 1, 1))
            steps = time

            x_tot = torch.Tensor(N, num_steps, *d).to(x.device)
            y_tot = torch.Tensor(N, num_steps, *dy).to(y.device)
            out = torch.Tensor(N, num_steps, *d).to(x.device)
            store_steps = steps
            num_iter = num_steps
            steps_expanded = time

            with torch.no_grad():
                if first:

                    for k in range(num_iter):
                        gamma = gammas[k]
                        gradx = grad_gauss(x, mean_final, var_final)

                        t_old = x + gamma * gradx
                        z = torch.randn(x.shape, device=x.device)
                        x = t_old + torch.sqrt(2 * gamma)*z
                        gradx = grad_gauss(x, mean_final, var_final)

                        t_new = x + gamma * gradx

                        x_tot[:, k, :] = x
                        y_tot[:, k, :] = y

                        out[:, k, :] = (t_old - t_new)  # / (2 * gamma)


                else:
                    for k in range(num_iter):
                        gamma = gammas[k]
                        t_old = x + nets[forward_or_backward_rev](x, steps[:, k, :], y, x_orig)

                        if sample & (k == num_iter-1):
                            x = t_old
                        else:
                            z = torch.randn(x.shape, device=x.device)
                            x = t_old + torch.sqrt(2 * gamma) * z
                        t_new = x + nets[forward_or_backward_rev](x, steps[:, k, :], y, x_orig)

                        x_tot[:, k, :] = x
                        y_tot[:, k, :] = y
                        
                        
                        out[:, k, :] = (t_old - t_new)

                x_tot = x_tot.unsqueeze(2)
                out = out.unsqueeze(2)

                batch_data = torch.cat((x_tot, out), dim=2)
                flat_data = batch_data.flatten(start_dim=0, end_dim=1)
                self.data[b] = flat_data
                
                
                y_tot = y_tot.unsqueeze(1)
                
                flat_y_data = y_tot.flatten(start_dim=0, end_dim=1)
                self.y_data[b] = flat_y_data.flatten(start_dim=0, end_dim=1)


                flat_steps = steps_expanded.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = flat_steps

        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.y_data = self.y_data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

        print('Cache size: {0}'.format(self.data.shape))

    def __getitem__(self, index):
        item = self.data[index]
        x = item[0]
        out = item[1]
        steps = self.steps_data[index]
        y = self.y_data[index]
        
        return x, out, y, steps

    def __len__(self):
        return self.data.shape[0]
    

def iterate_ipf(nets, opts, device, dls, gammas, npar, batch_size, num_steps, d, dy, T, mean_final, var_final,
                n_iter=200, forward_or_backward='f', forward_or_backward_rev='b', first=False, sample=False):

    nets['iter_loss'] = []
    nets['iter_et'] = []

    CL = CacheLoader(nets=nets,
                     device=device,
                     dls=dls,
                     gammas=gammas,
                     npar=npar,
                     batch_size=batch_size,
                     num_steps=num_steps,
                     d=d,
                     dy=dy,
                     mean_final=mean_final,
                     var_final=var_final,
                     forward_or_backward=forward_or_backward,
                     forward_or_backward_rev=forward_or_backward_rev,
                     first=first,
                     sample=sample)
    
    CL = DataLoader(CL, batch_size=batch_size, shuffle=False)

    for i_iter in range(n_iter):
        ttrain = 0
        t0 = time.time()
        for (i, data_iter) in enumerate(CL):
            (x, out, y, steps_expanded) = data_iter
            x = x.to(device)
            x_orig = x.clone().to(device)
            y = y.to(device)
            out = out.to(device)
            steps_expanded = steps_expanded.to(device)
            eval_steps = T - steps_expanded

            t1 = time.time()
            #---------------------------------------------------------
            pred = nets[forward_or_backward](x, eval_steps, y, x_orig)
            loss = F.mse_loss(pred, out)
            loss.backward()
            opts[forward_or_backward].step()
            opts[forward_or_backward].zero_grad()
            #---------------------------------------------------------
            ttrain += (time.time()-t1)
        
        nets['iter_loss'].append(loss)
        nets['iter_et'].append(time.time()-t0)
        print(f"{i_iter} - loss: {loss:.6f} ---- elapsed time: {time.time()-t0:.2f} ---- training time: {ttrain:.2f}")
        #EMA update
        nets[forward_or_backward].update()


def sample_data(dls, data, netsEnergy, netsConv, de, d, num_steps_voxel, num_steps_energy, gammas_voxel, gammas_energy, device,
                forward_or_backward = 'f', forward_or_backward_rev = 'b', full_sample = True):
    
    shifter_energy_g4 = data['shifter_energy_g4']
    shifter_energy_gflash = data['shifter_energy_gflash']
    scaler_energy_g4 = data['scaler_energy_g4']
    scaler_energy_gflash = data['scaler_energy_gflash']
    shifter_energy_fullrange_g4 = data['shifter_energy_fullrange_g4']
    shifter_energy_fullrange_gflash = data['shifter_energy_fullrange_gflash']
    scaler_energy_fullrange_g4 = data['scaler_energy_fullrange_g4']
    scaler_energy_fullrange_gflash = data['scaler_energy_fullrange_gflash']
    shifter_g4 = data['shifter_g4']
    shifter_gflash = data['shifter_gflash']
    scaler_g4 = data['scaler_g4']
    scaler_gflash = data['scaler_gflash']

    data_orig = []
    data_energy_particle = []
    data_x = []
    data_y = []
    iteration = -1

    netsEnergy_ts = []
    netsConv_ts = []
    
    for b, dat in enumerate(dls[forward_or_backward]):    
        x, y = dat
        x = x.float().to(device)
        y = y.float().to(device)

        x_orig = x.clone()

        N = x.shape[0]
       
        steps_voxel = torch.arange(num_steps_voxel).to(device)
        time_voxel = torch.cumsum(gammas_voxel, 0).to(device).float()
        steps_voxel = steps_voxel.reshape((1, num_steps_voxel, 1)).repeat((N, 1, 1))
        time_voxel = time_voxel.reshape((1, num_steps_voxel, 1)).repeat((N, 1, 1))
        steps_voxel = time_voxel
        num_iter_voxel = num_steps_voxel

        steps_energy = torch.arange(num_steps_energy).to(device)
        time_energy = torch.cumsum(gammas_energy, 0).to(device).float()
        steps_energy = steps_energy.reshape((1, num_steps_energy, 1)).repeat((N, 1, 1))
        time_energy = time_energy.reshape((1, num_steps_energy, 1)).repeat((N, 1, 1))
        steps_energy = time_energy
        num_iter_energy = num_steps_energy

        energy__shower_tot = torch.Tensor(N, num_steps_energy, *de).to(x.device)
        energy__shower_out = torch.Tensor(N, num_steps_energy, *de).to(x.device)
        
        x_tot = torch.Tensor(N, num_steps_voxel, *d).to(x.device)
        out = torch.Tensor(N, num_steps_voxel, *d).to(x.device)
        
        shifter_energy_g4_tensor = torch.tensor(shifter_energy_g4).to(x.device)
        shifter_energy_gflash_tensor = torch.tensor(shifter_energy_gflash).to(x.device)
        scaler_energy_g4_tensor = torch.tensor(scaler_energy_g4).to(x.device)
        scaler_energy_gflash_tensor = torch.tensor(scaler_energy_gflash).to(x.device)
        
        shifter_energy_fullrange_g4_tensor = torch.tensor(shifter_energy_fullrange_g4).to(x.device)
        shifter_energy_fullrange_gflash_tensor = torch.tensor(shifter_energy_fullrange_gflash).to(x.device)
        scaler_energy_fullrange_g4_tensor = torch.tensor(scaler_energy_fullrange_g4).to(x.device)
        scaler_energy_fullrange_gflash_tensor = torch.tensor(scaler_energy_fullrange_gflash).to(x.device)
        
        shifter_g4_tensor = torch.tensor(shifter_g4).to(x.device)
        shifter_gflash_tensor = torch.tensor(shifter_gflash).to(x.device)
        scaler_g4_tensor = torch.tensor(scaler_g4).to(x.device)
        scaler_gflash_tensor = torch.tensor(scaler_gflash).to(x.device)
        
        y_current = y.clone()
        energy__shower_start = y_current[:,0:1].clone()
        energy__shower_target = y_current[:,1:2].clone()
        energy__particle = y_current[:,2:3].clone()
        
        energy__shower_orig = energy__shower_start.clone().view(-1, 1, 1, 1)

        energy__shower_orig = (energy__shower_orig * scaler_energy_gflash_tensor) + shifter_energy_gflash_tensor
        energy__shower_orig = energy__shower_orig * energy__particle.view(-1, 1, 1, 1)
                            
        with torch.no_grad():
            for k in range(num_iter_energy):
                gamma = gammas_energy[k]

                t0 = time.time()
                #---------------------------------------------------------------------------------------------------#
                t_old = energy__shower_start + netsEnergy[forward_or_backward_rev](energy__shower_start, 
                                                                             steps_energy[:, k, :],
                                                                             energy__particle, energy__shower_start)
                if k == num_iter_energy-1:
                    energy__shower_start = t_old
                else:
                    z = torch.randn(energy__shower_start.shape, device=x.device)
                    energy__shower_start = t_old + torch.sqrt(2 * gamma) * z

                t_new = energy__shower_start + netsEnergy[forward_or_backward_rev](energy__shower_start, 
                                                                             steps_energy[:, k, :],
                                                                             energy__particle, energy__shower_start)
                energy__shower_tot[:, k, :] = energy__shower_start
                energy__shower_out[:, k, :] = (t_old - t_new)
                #---------------------------------------------------------------------------------------------------#
                netsEnergy_ts.append(time.time()-t0)
            
        energy__shower_tot = (energy__shower_tot * scaler_energy_g4_tensor) + shifter_energy_g4_tensor
        energy__shower_tot = energy__shower_tot * energy__particle.view(-1, 1, 1)
        energy__shower_tot = (energy__shower_tot - shifter_energy_fullrange_g4_tensor) / scaler_energy_fullrange_g4_tensor
            
        energy__shower_start = (energy__shower_start * scaler_energy_gflash_tensor) + shifter_energy_gflash_tensor
        energy__shower_start = energy__shower_start * energy__particle.view(-1, 1)
            
        y_current[:,1:2] = energy__shower_tot[:, iteration]
        
        if full_sample:
            with torch.no_grad():
                for k in range(num_iter_voxel):
                    gamma = gammas_voxel[k]

                    t0 = time.time()
                    #---------------------------------------------------------------------------------------------------#
                    t_old = x + netsConv[forward_or_backward_rev](x, steps_voxel[:, k, :], y_current, x)

                    if k == num_iter_voxel-1:
                        x = t_old
                    else:
                        z = torch.randn(x.shape, device=x.device)
                        x = t_old + torch.sqrt(2 * gamma) * z

                    t_new = x + netsConv[forward_or_backward_rev](x, steps_voxel[:, k, :], y_current,x )
                    
                    x_tot[:, k, :] = x
                    out[:, k, :] = (t_old - t_new)
                    #---------------------------------------------------------------------------------------------------#
                    netsConv_ts.append(time.time()-t0)
        

        x_orig = x_orig * scaler_gflash_tensor + shifter_gflash_tensor
        x_orig = x_orig * energy__shower_orig

        data_orig.append(x_orig.cpu().numpy())
        del x_orig

        energy__shower_tot = (energy__shower_tot * scaler_energy_fullrange_g4_tensor) + shifter_energy_fullrange_g4_tensor

        y_current[:,1:2] = energy__shower_tot[:, iteration]
        y_current[:,0:1] = energy__shower_start

        data_y.append(y_current.cpu().numpy())
        del y_current

        data_energy_particle.append(energy__particle.cpu().numpy()*10.0)
        del energy__particle


        if full_sample:
            x_tot = (x_tot * scaler_g4_tensor) + shifter_g4_tensor
            sum_old = torch.sum(x_tot, (2,3,4)).view(-1, x_tot.size(1), 1, 1, 1)
            sum_new = energy__shower_tot[:,iteration].view(-1, 1, 1, 1, 1)
            x_tot = x_tot / sum_old * sum_new
            data_x.append(x_tot.cpu().numpy())
            del x_tot
        else:
            data_x = [np.zeros((2,2)),np.zeros((2,2))]

        gc.collect()

    sample = {"energy_voxel_gflash_orig":np.concatenate(data_orig, 0), # used for En plot
              "energy_voxel_gflash_trafo":np.concatenate(data_x, 0),
              "energy_gflash_trafo":np.concatenate(data_y, 0), # used for En plot
              "energy_particle":np.concatenate(data_energy_particle, 0), # used for En plot
              "netsEnergy_ts":np.array(netsEnergy_ts),
              "netsConv_ts":np.array(netsConv_ts)}

    # # print(torch.cuda.memory_summary())
    
    # del x
    # del y
    # del steps_voxel
    # del time_voxel
    # del num_iter_voxel
    # del steps_energy
    # del time_energy
    # del num_iter_energy
    # del energy__shower_tot
    # del energy__shower_out
    # del out
    # del shifter_energy_g4_tensor
    # del shifter_energy_gflash_tensor
    # del scaler_energy_g4_tensor
    # del scaler_energy_gflash_tensor
    # del shifter_energy_fullrange_g4_tensor
    # del shifter_energy_fullrange_gflash_tensor
    # del scaler_energy_fullrange_g4_tensor
    # del scaler_energy_fullrange_gflash_tensor
    # del shifter_g4_tensor
    # del shifter_gflash_tensor
    # del scaler_g4_tensor
    # del scaler_gflash_tensor
    # del energy__shower_start
    # del energy__shower_target
    # del energy__shower_orig
    # del t_old
    # del z
    # del t_new
    # del gamma
    # del t0
    # del sum_old
    # del sum_new
    # del data_x
    # gc.collect()
    # torch.cuda.empty_cache()

    # print(torch.cuda.memory_summary())

    return sample



        ##----- Original -----##
        # x_orig = x_orig * scaler_gflash_tensor + shifter_gflash_tensor
        # x_orig = x_orig * energy__shower_orig

        # energy__shower_tot = (energy__shower_tot * scaler_energy_fullrange_g4_tensor) + shifter_energy_fullrange_g4_tensor

        # x_tot = (x_tot * scaler_g4_tensor) + shifter_g4_tensor

        # sum_old = torch.sum(x_tot, (2,3,4)).view(-1, x_tot.size(1), 1, 1, 1)
        # sum_new = energy__shower_tot[:,iteration].view(-1, 1, 1, 1, 1)
        
        # x_tot = x_tot / sum_old * sum_new
        
        # y_current[:,1:2] = energy__shower_tot[:, iteration]
        # y_current[:,0:1] = energy__shower_start

        # data_orig.append(x_orig.cpu().numpy())
        # data_y.append(y_current.cpu().numpy())
        
        # data_x.append(x_tot.cpu().numpy())
        # data_energy_particle.append(energy__particle.cpu().numpy()*10.0)


def train_en_network(modelEnergy_type, enc_layers_dim, pos_dim, n_iter, abs_path = '/media/marcelomd/HDD2/UFRGS/TCC/Dados'):
    
    ## ----------------------------------------------------------------------------------------------------
    ## Define Models
    ## ----------------------------------------------------------------------------------------------------

    ## Energy
    if modelEnergy_type == "SQuIRELS":
        from score_models import SquirelsScoreNetwork as ScoreNetworkEnergy
    elif modelEnergy_type == "Bernstein":
        from score_models import BernScoreKAN as ScoreNetworkEnergy
    elif modelEnergy_type == "Bottleneck":
        from score_models import BottleneckScoreKAGN as ScoreNetworkEnergy
    elif modelEnergy_type == "Chebyshev":
        from score_models import ChebyScoreKAN as ScoreNetworkEnergy
    elif modelEnergy_type == "Fast":
        from score_models import FastScoreKAN as ScoreNetworkEnergy
    elif modelEnergy_type == "Gram":
        from score_models import GramScoreKAN as ScoreNetworkEnergy
    elif modelEnergy_type == "Jacobi":
        from score_models import JacobiScoreKAN as ScoreNetworkEnergy
    elif modelEnergy_type == "Lagrange":
        from score_models import LagrangeScoreKAN as ScoreNetworkEnergy
    elif modelEnergy_type == "ReLU":
        from score_models import ReluScoreKAN as ScoreNetworkEnergy
    elif modelEnergy_type == "Wav":
        from score_models import WavScoreKAN as ScoreNetworkEnergy
    else:
        sys.exit("Selected energy model does not exist.")


    CUDA = True
    device = torch.device("cuda" if CUDA else "cpu")

    suffix = 'GFlash_Energy'

    num_steps = 20
    n = num_steps//2
    batch_size = 1024*16
    lr = 1e-5
    n_iter_glob = 50

    gamma_max = 0.001
    gamma_min = 0.001
    gamma_half = np.linspace(gamma_min, gamma_max, n)
    gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
    gammas = torch.tensor(gammas).to(device)
    T = torch.sum(gammas)

    # encoder_layers=[256,256]
    # pos_dim=128
    # decoder_layers=[256,256]

    normalize_energy=True
    # model_version = f"_{encoder_layers[0]}_{pos_dim}_{decoder_layers[0]}_"

    data_dir_path = f"{abs_path}/datasets/SB_Refinement/"
    models_dir_path = f"{abs_path}/repos/sb_ref_kan/models/Energy/{modelEnergy_type}/{enc_layers_dim}_{pos_dim}_{enc_layers_dim}"
    logs_path = f"{abs_path}/repos/sb_ref_kan/models/Energy/EnergyLogs"

    Path(models_dir_path).mkdir(parents=True, exist_ok=True)


    file_path_gflash = data_dir_path + 'run_GFlash01_100k_10_100GeV_full.npy'
    file_path_g4 = data_dir_path + 'run_Geant_100k_10_100GeV_full.npy'

    data = load_data(file_path_gflash, file_path_g4, normalize_energy=True, shuffle=True, plotting=False)

    energy_gflash = data["energy_gflash"]
    energy_particle_gflash = data["energy_particle_gflash"]
    energy_voxel_gflash = data["energy_voxel_gflash"]
    energy_g4 = data["energy_g4"]
    energy_particle_g4 = data["energy_particle_g4"]
    energy_voxel_g4 = data["energy_voxel_g4"]

    npar = int(energy_voxel_g4.shape[0])
                
    X_init = energy_gflash
    Y_init = energy_particle_gflash
    init_sample = torch.tensor(X_init)#.view(X_init.shape[0], 1, 10, 10)
    init_lable = torch.tensor(Y_init)
    init_ds = TensorDataset(init_sample, init_lable)
    init_dl = DataLoader(init_ds, batch_size=batch_size, shuffle=False)
    #init_dl = repeater(init_dl)
    # print(init_sample.shape)

    X_final = energy_g4
    Y_final = energy_particle_g4
    final_sample = torch.tensor(X_final)#.view(X_final.shape[0], 1, 10, 10)
    final_label = torch.tensor(Y_final)
    final_ds = TensorDataset(final_sample, final_label)
    final_dl = DataLoader(final_ds, batch_size=batch_size, shuffle=False)
    #final_dl = repeater(final_dl)

    #mean_final = torch.tensor(0.)
    #var_final = torch.tensor(1.*10**3) #infty like

    mean_final = torch.zeros(final_sample.size(-1)).to(device)
    var_final = 1.*torch.ones(final_sample.size(-1)).to(device)

    # print(final_sample.shape)
    # print(mean_final.shape)
    # print(var_final.shape)

    dls = {'f': init_dl, 'b': final_dl}

    # from score_models import FastScoreKAN as ScoreNetworkEnergy

    i1 = enc_layers_dim
    i2 = pos_dim
    encoder_layers=[i1,i1]
    pos_dim=i2
    decoder_layers=[i1,i1]

    model_version = f"{encoder_layers[0]}_{pos_dim}_{decoder_layers[0]}"

    model_f = ScoreNetworkEnergy(encoder_layers=encoder_layers,
                                pos_dim=pos_dim,
                                decoder_layers=decoder_layers,
                                n_cond = init_lable.size(1)).to(device)

    print(f"{modelEnergy_type}{model_version[:-1]}: {sum(p.numel() for p in model_f.parameters())} parameters")

    model_f = ScoreNetworkEnergy(encoder_layers=encoder_layers,
                                pos_dim=pos_dim,
                                decoder_layers=decoder_layers,
                                n_cond = init_lable.size(1)).to(device)

    model_b = ScoreNetworkEnergy(encoder_layers=encoder_layers,
                                pos_dim=pos_dim,
                                decoder_layers=decoder_layers,
                                n_cond = init_lable.size(1)).to(device)

    model_name = str(model_f.__class__)[21:-2]

    model_f = torch.nn.DataParallel(model_f)
    model_b = torch.nn.DataParallel(model_b)

    opt_f = torch.optim.Adam(model_f.parameters(), lr=lr)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr)

    net_f = EMA(model=model_f, decay=0.95).to(device)
    net_b = EMA(model=model_b, decay=0.95).to(device)

    nets  = {'f': net_f, 'b': net_b, 'iter_loss': [], 'iter_et': [] }
    opts  = {'f': opt_f, 'b': opt_b }

    nets['f'].train()
    nets['b'].train()


    d = init_sample[0].shape  # shape of object to diffuse
    dy = init_lable[0].shape  # shape of object to diffuse
    # print(d)
    # print(dy)

    #print(net_f)


    f = open(f"{logs_path}/{model_name}_{model_version}_.txt", 'w', encoding="utf-8")
    f.write("loss;elapsed_time;iteration\n")

    start_iter=0

    for i in range(1, 100):
        try:
            nets['f'].load_state_dict(torch.load(f"{models_dir_path}/Iter{i}_net_f_{suffix}_{model_name}_{model_version}_.pth", map_location=device))
            nets['b'].load_state_dict(torch.load(f"{models_dir_path}/Iter{i}_net_b_{suffix}_{model_name}_{model_version}_.pth", map_location=device))
            
            start_iter = i
        except:
            continue

    if start_iter == 0:
        iterate_ipf(nets=nets, opts=opts, device=device, dls=dls, gammas=gammas, npar=npar, batch_size=batch_size,
                    num_steps=num_steps, d=d, dy=dy, T=T, mean_final=mean_final, var_final=var_final, n_iter=100,
                    forward_or_backward='f', forward_or_backward_rev='b', first=True)
        for l, t in zip(nets['iter_loss'],nets['iter_et']):
            f.write(f"{l:.6f};{t:.2f};0\n")
        print('--------------- Done iter 0 ---------------')
        
    nets['f'].train()
    nets['b'].train()

    for i in range(start_iter+1, start_iter+n_iter):

        iterate_ipf(nets=nets, opts=opts, device=device, dls=dls, gammas=gammas, npar=npar, batch_size=batch_size,
                    num_steps=num_steps, d=d, dy=dy, T=T, mean_final=mean_final, var_final=var_final, n_iter=n_iter_glob,
                    forward_or_backward='b', forward_or_backward_rev='f', first=False)
        for l, t in zip(nets['iter_loss'],nets['iter_et']):
            f.write(f"{l:.6f};{t:.2f};{i}\n")
        print('--------------- Done iter B{:d} ---------------'.format(i))

        iterate_ipf(nets=nets, opts=opts, device=device, dls=dls, gammas=gammas, npar=npar, batch_size=batch_size,
                    num_steps=num_steps, d=d, dy=dy, T=T, mean_final=mean_final, var_final=var_final, n_iter=n_iter_glob,
                    forward_or_backward='f', forward_or_backward_rev='b', first=False)
        for l, t in zip(nets['iter_loss'],nets['iter_et']):
            f.write(f"{l:.6f};{t:.2f};{i}\n")
        print('--------------- Done iter F{:d} ---------------'.format(i))

        torch.save(net_f.state_dict(), f"{models_dir_path}/Iter{i}_net_f_{suffix}_{model_name}_{model_version}_.pth")
        torch.save(net_b.state_dict(), f"{models_dir_path}/Iter{i}_net_b_{suffix}_{model_name}_{model_version}_.pth")

    f.close()

    return 0


def train_conv_network(modelConv_type, enc_layers_dim, temb_dim, conv_dof, n_iter, abs_path = '/media/marcelomd/HDD2/UFRGS/TCC/Dados'):

    ## ----------------------------------------------------------------------------------------------------
    ## Define Models
    ## ----------------------------------------------------------------------------------------------------

    ## Conv
    if modelConv_type == "SQuIRELS":
        from score_models import SquirelsScoreNetworkConv as ScoreNetworkConv
    elif modelConv_type == "Bernstein":
        from score_models import BernScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Bottleneck":
        from score_models import BottleneckScoreKAGNConv as ScoreNetworkConv
    elif modelConv_type == "Chebyshev":
        from score_models import ChebyScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Fast":
        from score_models import FastScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Gram":
        from score_models import GramScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Jacobi":
        from score_models import JacobiScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Lagrange":
        from score_models import LagrangeScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "ReLU":
        from score_models import ReluScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Wav":
        from score_models import WavScoreKANConv as ScoreNetworkConv
    else:
        sys.exit("Selected energy model does not exist.")

    CUDA = True
    device = torch.device("cuda" if CUDA else "cpu")

    suffix = '_GFlash_Conv'

    num_steps = 20
    n = num_steps//2
    batch_size = 1024*8
    lr = 1e-5
    n_iter_glob = 50

    gamma_max = 0.001
    gamma_min = 0.001
    gamma_half = np.linspace(gamma_min, gamma_max, n)
    gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
    gammas = torch.tensor(gammas).to(device)
    T = torch.sum(gammas)

    data_dir_path = f"{abs_path}/datasets/SB_Refinement/"
    models_dir_path = f"{abs_path}/repos/sb_ref_kan/models/Conv/{modelConv_type}/{enc_layers_dim}_{temb_dim}_{conv_dof}"
    logs_path = f"{abs_path}/repos/sb_ref_kan/models/Conv/ConvLogs"

    Path(models_dir_path).mkdir(parents=True, exist_ok=True)


    file_path_gflash = data_dir_path + 'run_GFlash01_100k_10_100GeV_full.npy'
    file_path_g4 = data_dir_path + 'run_Geant_100k_10_100GeV_full.npy'

    data = load_data(file_path_gflash, file_path_g4, normalize_energy=False, shuffle=True, plotting=False)

    energy_gflash = data["energy_gflash"]
    energy_particle_gflash = data["energy_particle_gflash"]
    energy_voxel_gflash = data["energy_voxel_gflash"]
    energy_g4 = data["energy_g4"]
    energy_particle_g4 = data["energy_particle_g4"]
    energy_voxel_g4 = data["energy_voxel_g4"]

    npar = int(energy_voxel_g4.shape[0])
                
    X_init = energy_voxel_gflash
    Y_init = np.concatenate((energy_gflash, energy_g4, energy_particle_gflash), 1)
    init_sample = torch.tensor(X_init).view(X_init.shape[0], 1, 10, 10)
    init_lable = torch.tensor(Y_init)
    scaling_factor = 7
    #init_sample = (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor
    init_ds = TensorDataset(init_sample, init_lable)
    init_dl = DataLoader(init_ds, batch_size=batch_size, shuffle=False)
    #init_dl = repeater(init_dl)
    # print(init_sample.shape)

    X_final = energy_voxel_g4
    Y_final = np.concatenate((energy_g4, energy_gflash, energy_particle_g4), 1)
    scaling_factor = 7.
    final_sample = torch.tensor(X_final).view(X_final.shape[0], 1, 10, 10)
    final_label = torch.tensor(Y_final)
    #final_sample = (final_sample - final_sample.mean()) / final_sample.std() * scaling_factor
    final_ds = TensorDataset(final_sample, final_label)
    final_dl = DataLoader(final_ds, batch_size=batch_size, shuffle=False)
    #final_dl = repeater(final_dl)

    #mean_final = torch.tensor(0.)
    #var_final = torch.tensor(1.*10**3) #infty like

    mean_final = torch.zeros(1, 10, 10).to(device)
    var_final = 1.*torch.ones(1, 10, 10).to(device)

    # print(final_sample.shape)
    # print(mean_final.shape)
    # print(var_final.shape)


    dls = {'f': init_dl, 'b': final_dl}

    encoder_layers=[enc_layers_dim,enc_layers_dim]

    model_f = ScoreNetworkConv(encoder_layers=encoder_layers,
                            temb_dim=temb_dim,
                            conv_dof=conv_dof,
                            n_cond = init_lable.size(1)).to(device)

    model_version = f"{encoder_layers[0]}_{temb_dim}_{conv_dof}"

    print(f"{modelConv_type}{model_version}: {sum(p.numel() for p in model_f.parameters())} parameters")

    model_f = ScoreNetworkConv(encoder_layers=encoder_layers,
                           temb_dim=temb_dim,
                           conv_dof=conv_dof,
                           n_cond = init_lable.size(1)).to(device)

    model_b = ScoreNetworkConv(encoder_layers=encoder_layers,
                            temb_dim=temb_dim,
                            conv_dof=conv_dof,
                            n_cond = init_lable.size(1)).to(device)

    model_name = str(model_f.__class__)[21:-2]

    model_f = torch.nn.DataParallel(model_f)
    model_b = torch.nn.DataParallel(model_b)

    opt_f = torch.optim.Adam(model_f.parameters(), lr=lr)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr)

    net_f = EMA(model=model_f, decay=0.95).to(device)
    net_b = EMA(model=model_b, decay=0.95).to(device)

    nets  = {'f': net_f, 'b': net_b, 'iter_loss': [], 'iter_et': [] }
    opts  = {'f': opt_f, 'b': opt_b }

    nets['f'].train()
    nets['b'].train()


    d = init_sample[0].shape  # shape of object to diffuse
    dy = init_lable[0].shape  # shape of object to diffuse

    f = open(f"{logs_path}/{model_name}_{model_version}_.txt", 'w', encoding="utf-8")
    f.write("loss;elapsed_time;iteration\n")

    print(n_iter)
    start_iter=0

    for i in range(1, 400):
        try:
            nets['f'].load_state_dict(torch.load(f"{models_dir_path}/Iter{i}_net_f_{suffix}_{model_name}_{model_version}_.pth", map_location=device))
            nets['b'].load_state_dict(torch.load(f"{models_dir_path}/Iter{i}_net_b_{suffix}_{model_name}_{model_version}_.pth", map_location=device))
            
            start_iter = i
        except:
            continue

    if start_iter == 0:
        iterate_ipf(nets=nets, opts=opts, device=device, dls=dls, gammas=gammas, npar=npar, batch_size=batch_size,
                    num_steps=num_steps, d=d, dy=dy, T=T, mean_final=mean_final, var_final=var_final, n_iter=100,
                    forward_or_backward='f', forward_or_backward_rev='b', first=True)
        for l, t in zip(nets['iter_loss'],nets['iter_et']):
            f.write(f"{l:.6f};{t:.2f};0\n")
        print('--------------- Done iter 0 ---------------')
        
    nets['f'].train()
    nets['b'].train()

    for i in range(start_iter+1, start_iter+n_iter):

        iterate_ipf(nets=nets, opts=opts, device=device, dls=dls, gammas=gammas, npar=npar, batch_size=batch_size,
                    num_steps=num_steps, d=d, dy=dy, T=T, mean_final=mean_final, var_final=var_final, n_iter=n_iter_glob,
                    forward_or_backward='b', forward_or_backward_rev='f', first=False)
        for l, t in zip(nets['iter_loss'],nets['iter_et']):
            f.write(f"{l:.6f};{t:.2f};{i}\n")
        print('--------------- Done iter B{:d} ---------------'.format(i))

        iterate_ipf(nets=nets, opts=opts, device=device, dls=dls, gammas=gammas, npar=npar, batch_size=batch_size,
                    num_steps=num_steps, d=d, dy=dy, T=T, mean_final=mean_final, var_final=var_final, n_iter=n_iter_glob,
                    forward_or_backward='f', forward_or_backward_rev='b', first=False)
        for l, t in zip(nets['iter_loss'],nets['iter_et']):
            f.write(f"{l:.6f};{t:.2f};{i}\n")
        print('--------------- Done iter F{:d} ---------------'.format(i))

        torch.save(net_f.state_dict(), f"{models_dir_path}/Iter{i}_net_f_{suffix}_{model_name}_{model_version}_.pth")
        torch.save(net_b.state_dict(), f"{models_dir_path}/Iter{i}_net_b_{suffix}_{model_name}_{model_version}_.pth")

    f.close()

    return 0