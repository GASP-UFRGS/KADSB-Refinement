from collections import OrderedDict
from copy import deepcopy
from sys import stderr
import time

import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader


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
                forward_or_backward = 'f', forward_or_backward_rev = 'b'):
    
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

    netsEnergy_ts = {"old":[], "new":[]}
    netsConv_ts = {"old":[], "new":[]}

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
                t_old = energy__shower_start + netsEnergy[forward_or_backward_rev](energy__shower_start, 
                                                                             steps_energy[:, k, :],
                                                                             energy__particle, energy__shower_start)
                netsEnergy_ts["old"].append(t0-time.time())

                if k == num_iter_energy-1:
                    energy__shower_start = t_old
                else:
                    z = torch.randn(energy__shower_start.shape, device=x.device)
                    energy__shower_start = t_old + torch.sqrt(2 * gamma) * z
                
                t0 = time.time()
                t_new = energy__shower_start + netsEnergy[forward_or_backward_rev](energy__shower_start, 
                                                                             steps_energy[:, k, :],
                                                                             energy__particle, energy__shower_start)
                netsEnergy_ts["new"].append(t0-time.time())

                energy__shower_tot[:, k, :] = energy__shower_start
                energy__shower_out[:, k, :] = (t_old - t_new)
            
        energy__shower_tot = (energy__shower_tot * scaler_energy_g4_tensor) + shifter_energy_g4_tensor
        energy__shower_tot = energy__shower_tot * energy__particle.view(-1, 1, 1)
        energy__shower_tot = (energy__shower_tot - shifter_energy_fullrange_g4_tensor) / scaler_energy_fullrange_g4_tensor
            
        energy__shower_start = (energy__shower_start * scaler_energy_gflash_tensor) + shifter_energy_gflash_tensor
        energy__shower_start = energy__shower_start * energy__particle.view(-1, 1)
            
        y_current[:,1:2] = energy__shower_tot[:, iteration]
        
        with torch.no_grad():
            for k in range(num_iter_voxel):
                gamma = gammas_voxel[k]

                t0 = time.time()
                t_old = x + netsConv[forward_or_backward_rev](x, steps_voxel[:, k, :], y_current, x)
                netsConv_ts["old"].append(t0-time.time())

                if k == num_iter_voxel-1:
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z

                t0 = time.time()
                t_new = x + netsConv[forward_or_backward_rev](x, steps_voxel[:, k, :], y_current,x )
                netsConv_ts["new"].append(t0-time.time())

                x_tot[:, k, :] = x
                out[:, k, :] = (t_old - t_new)
        
        x_orig = x_orig * scaler_gflash_tensor + shifter_gflash_tensor
        x_orig = x_orig * energy__shower_orig

        energy__shower_tot = (energy__shower_tot * scaler_energy_fullrange_g4_tensor) + shifter_energy_fullrange_g4_tensor

        x_tot = (x_tot * scaler_g4_tensor) + shifter_g4_tensor

        sum_old = torch.sum(x_tot, (2,3,4)).view(-1, x_tot.size(1), 1, 1, 1)
        sum_new = energy__shower_tot[:,iteration].view(-1, 1, 1, 1, 1)
        
        x_tot = x_tot / sum_old * sum_new
        
        y_current[:,1:2] = energy__shower_tot[:, iteration]
        y_current[:,0:1] = energy__shower_start

        data_orig.append(x_orig.cpu().numpy())
        data_x.append(x_tot.cpu().numpy())
        data_y.append(y_current.cpu().numpy())
        
        data_energy_particle.append(energy__particle.cpu().numpy()*10.0)

    return({"energy_voxel_gflash_orig":np.concatenate(data_orig, 0),
            "energy_voxel_gflash_trafo":np.concatenate(data_x, 0),
            "energy_gflash_trafo":np.concatenate(data_y, 0),
            "energy_particle":np.concatenate(data_energy_particle, 0),
            "netsEnergy_ts":netsEnergy_ts,
            "netsConv_ts":netsConv_ts})