import sys
import time
import gc
from pathlib import Path

import numpy as np

from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import mplhep as hep
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.stats import wasserstein_distance

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from collections import OrderedDict

from functions import EMA, load_data, sample_data



def plot_image(data_l, data_name_l):
    
    fig = plt.figure(figsize=(6*len(data_l),6))
    fig.set_facecolor('white')
    outer = gridspec.GridSpec(1, len(data_l) , wspace=0.2, hspace=0.0)

    for (i, (d, d_n)) in enumerate( zip(data_l, data_name_l) ):

        subplot = fig.add_subplot(outer[i])

        im = subplot.imshow(np.mean(d, 0), norm=LogNorm(vmin=0.001, vmax=20), filternorm=False, interpolation='none', cmap = 'viridis',  origin='lower')
        subplot.patch.set_facecolor('white')
        subplot.title.set_text(d_n)
        subplot.set_xlabel('y [cells]')
        subplot.set_ylabel('x [cells]')
        fig.colorbar(im)
    
    return fig

def get_nhit(x):
    temp = np.array(x)
    temp = np.array(x)
    temp[temp>1e-3] = 1
    temp[temp<=1e-3] = 0
    return temp

def get_espec(x):
        temp = np.array(x)
        temp = np.array(x)
        temp[temp<1e-4] = 0
        return temp

def format_fig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)
    return ax0
    
def set_grid(ratio=True):
    if ratio:
        fig = plt.figure(figsize=(9, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        fig = plt.figure(figsize=(9, 7))
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

def emd(ref,array,weights_arr,nboot = 100):
    ds = []
    for _ in range(nboot):
        #ref_boot = np.random.choice(ref,ref.shape[0])
        arr_idx = np.random.choice(range(array.shape[0]),array.shape[0])
        array_boot = array[arr_idx]
        w_boot = weights_arr[arr_idx]
        ds.append(wasserstein_distance(ref,array_boot,v_weights=w_boot))
    
    return np.mean(ds), np.std(ds)
    # mse = np.square(ref-array)/ref
    # return np.sum(mse)

def triangle_distance(x,y,binning):
    dist = 0
    w = binning[1:] - binning[:-1]
    for ib in range(len(x)):
        dist+=0.5*w[ib]*(x[ib] - y[ib])**2/(x[ib] + y[ib]) if x[ib] + y[ib] >0 else 0.0
    return dist*1e3
        
def histogram(feed_dict,emds,errs,line_style,colors,xlabel='',ylabel='',reference_name='Truth',logy=False,binning=None,
              label_loc='best',plot_ratio=False,weights=None,uncertainty=None,triangle=True,
              y_lim_ratio=[0.5,1.5], title='', density=True, y_range=None, metric_name=''):
    
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    ref_plot = {'histtype':'stepfilled','alpha':0.2}
    other_plots = {'histtype':'step','linewidth':2}
    fig,gs = set_grid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])

    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),30)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=density,weights=weights[reference_name])

    d_list = []
    err_list = []
    
    maxy = 0    
    for ip,plot in enumerate(feed_dict.keys()):
        plot_style = ref_plot if reference_name == plot else other_plots
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=density,weights=weights[plot],**plot_style)
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=density,**plot_style)

            
        if triangle:
            # print(plot)
            emds[f"{metric_name}_{plot}"],errs[f"{metric_name}_{plot}"] = emd(feed_dict[reference_name][:100000],feed_dict[plot][:100000],weights[plot][:100000])
            # print("EMD distance is: {}+-{}".format(d,err))
            #d = get_triangle_distance(dist,reference_hist,binning)
            #print("Triangular distance is: {0:.2g}".format(d))
            if not reference_name == plot:
                #label_list.append(plot + ' EMD: {0:.2g} $\pm$ {1:.2g}'.format(d, err))
                #label_list.append(plot)
                d_list.append(emds[f"{metric_name}_{plot}"])
                #label_list.append(' $\pm$ ')
                err_list.append(errs[f"{metric_name}_{plot}"])
                
            if reference_name == plot:
                #label_list.append(plot + ' EMD: {0:.2g} $\pm$ {1:.2g}'.format(d, err))
                #label_list.append(plot)
                d_list.append(emds[f"{metric_name}_{plot}"])
                #label_list.append(' $\pm$ ')
                err_list.append(errs[f"{metric_name}_{plot}"])
                #label_list.append(plot)

            
        if np.max(dist) > maxy:
            maxy = np.max(dist)
            
        if plot_ratio:
            if reference_name!=plot:
                ratio = np.ma.divide(dist,reference_hist).filled(0)                
                ax1.plot(xaxis,ratio,color=colors[plot],marker='+',ms=8,lw=0,markerfacecolor='none',markeredgewidth=3)
                if uncertainty is not None:
                    for ibin in range(len(binning)-1):
                        xup = binning[ibin+1]
                        xlow = binning[ibin]
                        ax1.fill_between(np.array([xlow,xup]),
                                         uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
 
    len_list = []
    
    for err in (err_list):
        #print('{0:.2g}'.format(err))
        #print(len('{0:.2g}'.format(err)))
        temp = '{0:.2g}'.format(err)
        if 'e' in temp:
            len_list.append(int(temp[-1])+2)
        else:
            len_list.append(len(temp))
     
    precision = max(len_list)-2

        
    label_list = []
    
    for ip,plot in enumerate(feed_dict.keys()):
        label_list.append(plot)

    for d in (d_list):
        label_list.append('EMD: {0:.{1}f}'.format(d, precision))
        ax0.plot(binning[5], -100000, alpha=0.0)

    for ip,plot in enumerate(feed_dict.keys()):
        label_list.append(' $\pm$ ')
        ax0.plot(binning[5], -100000, alpha=0.0)

    for err in (err_list):
        label_list.append('{0:.{1}f}'.format(err, precision))
        ax0.plot(binning[5], -100000, alpha=0.0)

    out = ' ' 
    for ip,plot in enumerate(feed_dict.keys()):
        out = out + (' & {0:.{1}f}'.format(d_list[ip], precision)) + '$\pm$' + ('{0:.{1}f}'.format(err_list[ip], precision))
    
    # print('{}, {} , {}: '.format(xlabel, title, plot) + out)
    
    out = ' ' 
    for ip,plot in enumerate(feed_dict.keys()):
        temp = '{0:.1g}'.format(err_list[ip])
        if 'e' in temp:
            precision = temp[-1]
        else:
            precision = (len(temp))-2
        out = out + (' & {0:.{1}f}'.format(d_list[ip], precision)) + '(' + ('{0:.{1}f})'.format(err_list[ip], precision))[-2:]
    
    # print('{}, {} , {}: '.format(xlabel, title, plot) + out)
    

    if logy:
        ax0.set_yscale('log')

    if triangle:
        ax0.legend(loc=label_loc,fontsize=16,ncol=4, labels=label_list, columnspacing=-1.15)
    else:
        ax0.legend(loc=label_loc,fontsize=16,ncol=1)

    ax0.title.set_text(title)
    ax0.title.set_size(25)
    
    if y_range is None:
        ax0.set_ylim(0,1.3*maxy)
    else:
        ax0.set_ylim(y_range[0], y_range[1])
    
    if plot_ratio:
        ax0 = format_fig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to Truth')
        plt.axhline(y=1.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim(y_lim_ratio)
        plt.xlabel(xlabel)
    else:
        ax0 = format_fig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
       
    try:
        ax0.ticklabel_format(useOffset=False)
    except:
        pass
    return fig,ax0,emds,errs

def scatter(xdata, ydata, data_label ,xlabel='',ylabel='',label_loc='best', title=''):

    fig,gs = set_grid(ratio=False) 
    ax0 = plt.subplot(gs[0])

    ax0.scatter(xdata, ydata, label=data_label)

    ax0.legend(loc=label_loc,fontsize=16,ncol=2)
    ax0.title.set_text(title)
    ax0.title.set_size(25)
    ax0 = format_fig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
       
    ax0.ticklabel_format(useOffset=False)
    return fig,ax0

def sampling_and_plotting(modelEnergy_type, en_elayers_dim, pos_dim,
                          modelConv_type, conv_elayers_dim, temb_dim, conv_dof,
                          en_model_iter=-1, conv_model_iter=-1, 
                          abs_path='/mnt/f/UFRGS/TCC/Dados', record_metrics = True,
                          generate_plots = True, full_model_metrics = False, energy_intervals = False,
                          cuda = True, esum1d=False, esumfrac1d=False, esum=False, emax=False, espec=False,
                          especnorm=False, nhit=False, ex=False, ey=False):

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
        
    ## Conv
    if modelConv_type == "Bernstein":
        from score_models import BernScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Bottleneck":
        from score_models import BottleneckScoreKAGNConv as ScoreNetworkConv
    elif modelConv_type == "BottleneckAttention":
        from score_models import BottleneckScoreKAGNAttentionConv as ScoreNetworkConv
    elif modelConv_type == "BottleneckKAGNLinear":
        from score_models import BottleneckScoreKAGNLinear as ScoreNetworkConv
    elif modelConv_type == "Chebyshev":
        from score_models import ChebyScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Fast":
        from score_models import FastScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "FastLinear":
        from score_models import FastScoreKANLinear as ScoreNetworkConv
    elif modelConv_type == "FastWide":
        from score_models import FastScoreKANConvWide as ScoreNetworkConv
    elif modelConv_type == "Gram":
        from score_models import GramScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Jacobi":
        from score_models import JacobiScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "Lagrange":
        from score_models import LagrangeScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "ReLU":
        from score_models import ReluScoreKANConv as ScoreNetworkConv
    elif modelConv_type == "ReLULinear":
        from score_models import ReluScoreKANLinear as ScoreNetworkConv
    elif modelConv_type == "SQuIRELS":
        from score_models import SquirelsScoreNetworkConv as ScoreNetworkConv
    elif modelConv_type == "SQuIRELSLinear":
        from score_models import SquirelsScoreNetworkLinear as ScoreNetworkConv
    elif modelConv_type == "Wav":
        from score_models import WavScoreKANConv as ScoreNetworkConv
    else :
        sys.exit("Selected convolution model does not exist.")



    ##--------------------------------------------------------------------------------------------------------

    mpl.style.use('classic')

    if cuda:
        CUDA = True
    else:
        CUDA = False
    device = torch.device("cuda" if CUDA else "cpu")
    print(device)

    full_sample = full_model_metrics

    data_dir_path = abs_path + '/datasets/SB_Refinement'

    ## ----------------------------------------------------------------------------------------------------
    ## Energy
    ## ----------------------------------------------------------------------------------------------------
    en_encoder_layers = [en_elayers_dim,en_elayers_dim]
    en_decoder_layers = [en_elayers_dim,en_elayers_dim]
    modelEnergy_version = f"{en_encoder_layers[0]}_{pos_dim}_{en_encoder_layers[0]}"

    ## ----------------------------------------------------------------------------------------------------
    ## Conv
    ## ----------------------------------------------------------------------------------------------------
    conv_encoder_layers = [conv_elayers_dim,conv_elayers_dim]
    modelConv_version = f"{conv_encoder_layers[0]}_{temb_dim}_{conv_dof}"

    ## ----------------------------------------------------------------------------------------------------
    ## Paths
    ## ----------------------------------------------------------------------------------------------------
    models_energy_dir_path = f'{abs_path}/models/Energy/{modelEnergy_type}/{modelEnergy_version}/'
    models_conv_dir_path = f'{abs_path}/models/Conv/{modelConv_type}/{modelConv_version}/'

    full_modelEnergy_name = f"{modelEnergy_type}_{modelEnergy_version}"
    full_modelConv_name = f"{modelConv_type}_{modelConv_version}"

    plots_dir_path = f'{abs_path}/cpu/plots/Full_Models/{full_modelEnergy_name}_{full_modelConv_name}/'
    if full_model_metrics:
        metrics_dir_path = f'{abs_path}/cpu/metrics/full_model_metrics/' #{full_modelEnergy_name}_{full_modelConv_name}/'
    else:
        metrics_dir_path = f'{abs_path}/cpu/metrics/en_metrics/'

    if en_model_iter == -1:
        energy_iter_list = range(1,10)
    # elif len(en_model_iter) == 1:
    #     energy_iter_list = en_model_iter*19
    else:
        energy_iter_list = en_model_iter
    
    
    if conv_model_iter == -1:
        conv_iter_list = range(1,10)
    # elif len(conv_model_iter) == 1:
    #     conv_iter_list = conv_model_iter*19
    else:
        conv_iter_list = conv_model_iter


    ## ----------------------------------------------------------------------------------------------------
    ## Plot parameters
    ## ----------------------------------------------------------------------------------------------------
    
    line_style = {
        'Geant4':'dotted',
        'GFlash':'-',
        full_modelEnergy_name:'-',
        full_modelConv_name:'-',
    }

    colors = {
        'Geant4':'black',
        'GFlash':'red',
        full_modelEnergy_name:'#2ca25f',
        full_modelConv_name:'#7570b3',
    }

    rc('text', usetex=True)
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    mpl.rcParams.update({'font.size': 19})
    mpl.rcParams.update({'figure.titlesize': 11})
    mpl.rcParams.update({'xtick.labelsize': 18})
    mpl.rcParams.update({'ytick.labelsize': 18})
    mpl.rcParams.update({'axes.labelsize': 18})
    mpl.rcParams.update({'legend.frameon': False})
    mpl.rcParams.update({'lines.linewidth': 2})

    hep.style.use("CMS")

    mpl.rcParams['text.usetex'] = False
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

    # free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
    # mem_used_mb = (total - free) / 1024 ** 2
    # print(f"Before loading data: {mem_used_mb}")

    ## ----------------------------------------------------------------------------------------------------
    ## Load Data
    ## ----------------------------------------------------------------------------------------------------

    file_path_gflash = data_dir_path + '/run_GFlash01_100k_10_100GeV_eval_full.npy'
    file_path_g4 = data_dir_path + '/run_Geant_100k_10_100GeV_eval_full.npy'

    # data = load_data(file_path_gflash=file_path_gflash, file_path_g4=file_path_g4, normalize_energy=True)

    ## -------- Load Data --------- ##

    energy_voxel_g4 = np.load(file_path_g4)[:, 0:100].astype(np.float32)
    energy_voxel_gflash  = np.load(file_path_gflash)[:, 0:100].astype(np.float32)

    energy_particle_g4 = np.load(file_path_g4)[:, 200:201].astype(np.float32)/10000.0
    energy_particle_gflash  = np.load(file_path_gflash)[:, 200:201].astype(np.float32)/10000.0


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

    energy_g4 = np.sum(energy_voxel_g4, 1, keepdims=True)
    energy_gflash = np.sum(energy_voxel_gflash, 1, keepdims=True)

    energy_voxel_g4 = np.reshape(energy_voxel_g4, (-1, 1, 10, 10))
    energy_voxel_gflash = np.reshape(energy_voxel_gflash, (-1, 1, 10, 10))

    energy_voxel_g4 = energy_voxel_g4/np.tile(np.reshape(energy_g4, (-1, 1, 1, 1)), (1, 1, 10, 10))
    energy_voxel_gflash = energy_voxel_gflash/np.tile(np.reshape(energy_gflash, (-1, 1, 1, 1)), (1, 1, 10, 10))

    shifter_energy_fullrange_g4 = np.mean(energy_g4, 0)
    shifter_energy_fullrange_gflash = np.mean(energy_gflash, 0)
    scaler_energy_fullrange_g4 = np.std(energy_g4)
    scaler_energy_fullrange_gflash = np.std(energy_gflash, 0)

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
    scaler_energy_g4 = np.std(energy_g4)
    scaler_energy_gflash = np.std(energy_gflash, 0)

    energy_g4 = (energy_g4 - shifter_energy_g4)/scaler_energy_g4
    energy_gflash = (energy_gflash - shifter_energy_gflash)/scaler_energy_gflash
    
    data = {}

    data['energy_voxel_g4'] = energy_voxel_g4
    data['energy_voxel_gflash'] = energy_voxel_gflash
    data['energy_gflash'] = energy_gflash
    data['energy_g4'] = energy_g4
    data['energy_particle_gflash'] = energy_particle_gflash
    data['energy_particle_g4'] = energy_particle_g4
    data['shifter_energy_g4'] = shifter_energy_g4 
    data['shifter_energy_gflash'] = shifter_energy_gflash 
    data['scaler_energy_g4'] = scaler_energy_g4 
    data['scaler_energy_gflash'] = scaler_energy_gflash 
    data['shifter_energy_fullrange_g4'] = shifter_energy_fullrange_g4
    data['shifter_energy_fullrange_gflash'] = shifter_energy_fullrange_gflash
    data['scaler_energy_fullrange_g4'] = scaler_energy_fullrange_g4
    data['scaler_energy_fullrange_gflash'] = scaler_energy_fullrange_gflash
    data['shifter_g4'] = shifter_g4
    data['shifter_gflash'] = shifter_gflash
    data['scaler_g4'] = scaler_g4
    data['scaler_gflash'] = scaler_gflash

    # free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
    # mem_used_mb = (total - free) / 1024 ** 2
    # print(f"After loading data: {mem_used_mb}")
    
    ## -------- Load Data --------- ##

    # energy_voxel_g4 = data['energy_voxel_g4']
    # energy_voxel_gflash = data['energy_voxel_gflash']
    # energy_gflash = data['energy_gflash']
    # energy_g4 = data['energy_g4']
    # energy_particle_gflash = data['energy_particle_gflash']
    # energy_particle_g4 = data['energy_particle_g4']

    batch_size = 10000

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

    d = init_sample[0].shape  # shape of object to diffuse
    dy = init_lable[0].shape  # shape of object to diffuse
    de = [1]  # shape of object to diffuse

    lr = 1e-5

    num_steps_voxel = 20
    gamma_max_voxel = 0.001
    gamma_min_voxel = 0.001

    n = num_steps_voxel//2
    gamma_half_voxel = np.linspace(gamma_min_voxel, gamma_max_voxel, n)
    gammas_voxel = np.concatenate([gamma_half_voxel, np.flip(gamma_half_voxel)])
    gammas_voxel = torch.tensor(gammas_voxel).to(device)
    T_voxel = torch.sum(gammas_voxel)

    # print(gammas_voxel)

    num_steps_energy = 20
    gamma_max_energy = 0.001
    gamma_min_energy = 0.001

    n = num_steps_energy//2
    gamma_half_energy = np.linspace(gamma_min_energy, gamma_max_energy, n)
    gammas_energy = np.concatenate([gamma_half_energy, np.flip(gamma_half_energy)])
    gammas_energy = torch.tensor(gammas_energy).to(device)
    T_energy = torch.sum(gammas_energy)

    # print(gammas_energy)

    ## ----------------------------------------------------------------------------------------------------
    ## Create Models
    ## ----------------------------------------------------------------------------------------------------
    
    ## Energy
    modelEnergy_f = ScoreNetworkEnergy(encoder_layers=en_encoder_layers,
                                    pos_dim=pos_dim,
                                    decoder_layers=en_decoder_layers,
                                    n_cond = 1).to(device)

    modelEnergy_b = ScoreNetworkEnergy(encoder_layers=en_encoder_layers,
                                    pos_dim=pos_dim,
                                    decoder_layers=en_decoder_layers,
                                    n_cond = 1).to(device)

    modelEnergy_name = str(modelEnergy_f.__class__)[21:-2]
    print(modelEnergy_name)

    ## Conv
    modelConv_f = ScoreNetworkConv(encoder_layers=conv_encoder_layers,
                                temb_dim=temb_dim,
                                conv_dof=conv_dof,
                                n_cond = init_lable.size(1)).to(device)

    modelConv_b = ScoreNetworkConv(encoder_layers=conv_encoder_layers,
                                temb_dim=temb_dim,
                                conv_dof=conv_dof,
                                n_cond = init_lable.size(1)).to(device)

    modelConv_name = str(modelConv_f.__class__)[21:-2]
    print(modelConv_name)



    ## ----------------------------------------------------------------------------------------------------
    ## Create Folders
    ## ----------------------------------------------------------------------------------------------------

    if record_metrics:
        Path(f"{metrics_dir_path}/emds/").mkdir(parents=True, exist_ok=True)
        Path(f"{metrics_dir_path}/errors/").mkdir(parents=True, exist_ok=True)

    if generate_plots:
        Path(plots_dir_path).mkdir(parents=True, exist_ok=True)
    

    ## decay=1.0: No change on update
    ## decay=0.0: No memory of previous updates, memory is euqal to last update
    ## decay=0.9: New value 9 parts previous updates, 1 part current update
    ## decay=0.95: New value 49 parts previous updates, 1 part current update
    metrics = pd.DataFrame()
    errors = pd.DataFrame()

    with torch.no_grad():

        for i in range(len(energy_iter_list)):

            # free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
            # mem_used_mb = (total - free) / 1024 ** 2
            # print(f"Before loading data: {mem_used_mb}")

            ## ----------------------------------------------------------------------------------------------------
            ## Load Data
            ## ----------------------------------------------------------------------------------------------------

            file_path_gflash = data_dir_path + '/run_GFlash01_100k_10_100GeV_eval_full.npy'
            file_path_g4 = data_dir_path + '/run_Geant_100k_10_100GeV_eval_full.npy'

            # data = load_data(file_path_gflash=file_path_gflash, file_path_g4=file_path_g4, normalize_energy=True)

            ## -------- Load Data --------- ##

            energy_voxel_g4 = np.load(file_path_g4)[:, 0:100].astype(np.float32)
            energy_voxel_gflash  = np.load(file_path_gflash)[:, 0:100].astype(np.float32)

            energy_particle_g4 = np.load(file_path_g4)[:, 200:201].astype(np.float32)/10000.0
            energy_particle_gflash  = np.load(file_path_gflash)[:, 200:201].astype(np.float32)/10000.0


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

            energy_g4 = np.sum(energy_voxel_g4, 1, keepdims=True)
            energy_gflash = np.sum(energy_voxel_gflash, 1, keepdims=True)

            energy_voxel_g4 = np.reshape(energy_voxel_g4, (-1, 1, 10, 10))
            energy_voxel_gflash = np.reshape(energy_voxel_gflash, (-1, 1, 10, 10))

            energy_voxel_g4 = energy_voxel_g4/np.tile(np.reshape(energy_g4, (-1, 1, 1, 1)), (1, 1, 10, 10))
            energy_voxel_gflash = energy_voxel_gflash/np.tile(np.reshape(energy_gflash, (-1, 1, 1, 1)), (1, 1, 10, 10))

            shifter_energy_fullrange_g4 = np.mean(energy_g4, 0)
            shifter_energy_fullrange_gflash = np.mean(energy_gflash, 0)
            scaler_energy_fullrange_g4 = np.std(energy_g4)
            scaler_energy_fullrange_gflash = np.std(energy_gflash, 0)

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
            scaler_energy_g4 = np.std(energy_g4)
            scaler_energy_gflash = np.std(energy_gflash, 0)

            energy_g4 = (energy_g4 - shifter_energy_g4)/scaler_energy_g4
            energy_gflash = (energy_gflash - shifter_energy_gflash)/scaler_energy_gflash
            
            data = {}

            data['energy_voxel_g4'] = energy_voxel_g4
            data['energy_voxel_gflash'] = energy_voxel_gflash
            data['energy_gflash'] = energy_gflash
            data['energy_g4'] = energy_g4
            data['energy_particle_gflash'] = energy_particle_gflash
            data['energy_particle_g4'] = energy_particle_g4
            data['shifter_energy_g4'] = shifter_energy_g4 
            data['shifter_energy_gflash'] = shifter_energy_gflash 
            data['scaler_energy_g4'] = scaler_energy_g4 
            data['scaler_energy_gflash'] = scaler_energy_gflash 
            data['shifter_energy_fullrange_g4'] = shifter_energy_fullrange_g4
            data['shifter_energy_fullrange_gflash'] = shifter_energy_fullrange_gflash
            data['scaler_energy_fullrange_g4'] = scaler_energy_fullrange_g4
            data['scaler_energy_fullrange_gflash'] = scaler_energy_fullrange_gflash
            data['shifter_g4'] = shifter_g4
            data['shifter_gflash'] = shifter_gflash
            data['scaler_g4'] = scaler_g4
            data['scaler_gflash'] = scaler_gflash

            # free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
            # mem_used_mb = (total - free) / 1024 ** 2
            # print(f"After loading data: {mem_used_mb}")

            
            ## -------- Load Data --------- ##

            # energy_voxel_g4 = data['energy_voxel_g4']
            # energy_voxel_gflash = data['energy_voxel_gflash']
            # energy_gflash = data['energy_gflash']
            # energy_g4 = data['energy_g4']
            # energy_particle_gflash = data['energy_particle_gflash']
            # energy_particle_g4 = data['energy_particle_g4']

            batch_size = 10000

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

            d = init_sample[0].shape  # shape of object to diffuse
            dy = init_lable[0].shape  # shape of object to diffuse
            de = [1]  # shape of object to diffuse

            lr = 1e-5

            num_steps_voxel = 20
            gamma_max_voxel = 0.001
            gamma_min_voxel = 0.001

            n = num_steps_voxel//2
            gamma_half_voxel = np.linspace(gamma_min_voxel, gamma_max_voxel, n)
            gammas_voxel = np.concatenate([gamma_half_voxel, np.flip(gamma_half_voxel)])
            gammas_voxel = torch.tensor(gammas_voxel).to(device)
            T_voxel = torch.sum(gammas_voxel)

            # print(gammas_voxel)

            num_steps_energy = 20
            gamma_max_energy = 0.001
            gamma_min_energy = 0.001

            n = num_steps_energy//2
            gamma_half_energy = np.linspace(gamma_min_energy, gamma_max_energy, n)
            gammas_energy = np.concatenate([gamma_half_energy, np.flip(gamma_half_energy)])
            gammas_energy = torch.tensor(gammas_energy).to(device)
            T_energy = torch.sum(gammas_energy)

            cutOff = 0.0

            triangle=True

            y_lim_ratio_l = {'esum': [0.9, 1.1],
                            'esumfrac': [0.0, 2.0],
                            'emax': [0.0, 2.0],
                            'nhit': [0.0, 2.0],
                            'espec': [0.0, 2.0],
                            'ex': [0.0, 2.0],
                            }

            binning_l = {'esum': np.linspace(5,105,50),
                        'esumfrac': np.linspace(0.97,1.01,50),
                        'emax': np.linspace(0, 27,50),
                        'nhit': np.linspace(0, 100,101),
                        'espec': np.logspace(-4, 1.5, 100),
                        'ex': np.linspace(0, 10,11),
                        }

            y_range_l = {'esum': None,
                        'esumfrac': None,
                        'emax': None,
                        'nhit': None,
                        'espec': [1e-6, 1e3],
                        'ex': [8e-3, 5e2],
                        }

            print(f"Iteration: {energy_iter_list[i]} - {full_modelEnergy_name} - {full_modelConv_name}")

            # free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
            # mem_used_mb = (total - free) / 1024 ** 2
            # print(f"Before loading model: {mem_used_mb}")

            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Load Model
            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            modelEnergy_f = torch.nn.DataParallel(modelEnergy_f)
            modelEnergy_b = torch.nn.DataParallel(modelEnergy_b)

            netEnergy_f = EMA(model=modelEnergy_f, decay=0.95).to(device)
            netEnergy_b = EMA(model=modelEnergy_b, decay=0.95).to(device)

            netsEnergy  = {'f': netEnergy_f, 'b': netEnergy_b }

            netsEnergy['f'].load_state_dict(torch.load(f"{models_energy_dir_path}Iter{energy_iter_list[i]}_net_f_GFlash_Energy{modelEnergy_name}_{modelEnergy_version}_.pth", map_location=device), strict=False)
            netsEnergy['b'].load_state_dict(torch.load(f"{models_energy_dir_path}Iter{energy_iter_list[i]}_net_b_GFlash_Energy{modelEnergy_name}_{modelEnergy_version}_.pth", map_location=device), strict=False)

            netsEnergy['f'].eval()
            netsEnergy['b'].eval()

            modelConv_f = torch.nn.DataParallel(modelConv_f)
            modelConv_b = torch.nn.DataParallel(modelConv_b)


            netConv_f = EMA(model=modelConv_f, decay=0.95).to(device)
            netConv_b = EMA(model=modelConv_b, decay=0.95).to(device)

            netsConv  = {'f': netConv_f, 'b': netConv_b }

            netsConv['f'].load_state_dict(torch.load(f"{models_conv_dir_path}Iter{conv_iter_list[i]}_net_f_GFlash_Conv{modelConv_name}_{modelConv_version}_.pth", map_location=device), strict=False)
            netsConv['b'].load_state_dict(torch.load(f"{models_conv_dir_path}Iter{conv_iter_list[i]}_net_b_GFlash_Conv{modelConv_name}_{modelConv_version}_.pth", map_location=device), strict=False)

            netsConv['f'].eval()
            netsConv['b'].eval()
            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
            # mem_used_mb = (total - free) / 1024 ** 2
            # print(f"After loading model / before sampling: {mem_used_mb}")
            print(device)
            sample = sample_data(dls, data, netsEnergy, netsConv, de, d,
                                    num_steps_voxel, num_steps_energy,
                                    gammas_voxel, gammas_energy, device,
                                    forward_or_backward = 'f', forward_or_backward_rev = 'b',
                                    full_sample = full_sample)
            
            # free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
            # mem_used_mb = (total - free) / 1024 ** 2
            # print(f"After sampling: {mem_used_mb}")

            energy_voxel_gflash_orig = sample['energy_voxel_gflash_orig']
            energy_voxel_gflash_trafo = sample['energy_voxel_gflash_trafo']
            energy_gflash_trafo = sample['energy_gflash_trafo']
            energy_particle = sample['energy_particle']
            netsEnergy_ts = sample['netsEnergy_ts']
            netsConv_ts = sample['netsConv_ts']

            cutOff = 0.0
            
            data_trafo = energy_voxel_gflash_trafo[:,-1]#*100

            data_orig = energy_voxel_gflash_orig
            data_full = np.load(file_path_g4)

            data_full[data_full<0] = 0.0
            data_orig[data_orig<0] = 0.0
            data_trafo[data_trafo<0] = 0.0

            emds = {}
            errs = {}


            ## Calculate metrics and generate plots

            weight_dict = {
                'Geant4': np.ones(data_full.shape[0]),
                'GFlash': np.ones(data_full.shape[0]),
                full_modelConv_name: np.ones(data_full.shape[0]),
                full_modelEnergy_name: np.ones(data_full.shape[0]),
            }

            ### ESum 1D ###
            if esum1d:
                k = "Esum1D"
                feed_dict = {
                    'Geant4': np.sum(data_full[:,:100],(1)),
                    'GFlash': np.sum(data_orig,(1,2,3)),
                    full_modelEnergy_name: energy_gflash_trafo[:,1],
                }

                if generate_plots:
                    fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                        weights = weight_dict,
                                        label_loc= 'best',
                                        xlabel='Total Energy Sum [GeV]', ylabel= 'Normalized entries',
                                        binning = binning_l['esum'],triangle=triangle,
                                        logy=False, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['esum'],
                                        title='10-100 GeV', y_range= y_range_l['esum'])
                    fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'esum_1D_iter'+ str(i) + '.svg')
                    plt.close()
                if record_metrics and not(generate_plots):
                    for _,plot in enumerate(feed_dict.keys()):
                        emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])

            ### ESumFrac 1D ###
            if esumfrac1d:
                k = "EsumFrac1D"
                feed_dict = {
                    'Geant4': np.sum(data_full[:,:100],(1))/(data_full[:,200]/1000.0),
                    'GFlash': np.sum(data_orig,(1,2,3))/energy_particle[:,0],
                    full_modelEnergy_name: energy_gflash_trafo[:,1]/energy_particle[:,0],
                }
                if generate_plots:
                    fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                        weights = weight_dict,
                                        label_loc= 'best',
                                        xlabel='$E_{shower}/E_{particle}$', ylabel= 'Normalized entries',
                                        binning = binning_l['esumfrac'],triangle=triangle,
                                        logy=False, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['esumfrac'],
                                        title='10-100 GeV', y_range= y_range_l['esumfrac'])
                    fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'esumfrac_1D_iter'+ str(i) + '.svg')
                    plt.close()
                if record_metrics and not(generate_plots):
                    for _,plot in enumerate(feed_dict.keys()):
                        emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])

            if full_model_metrics:
                ### ESum ###
                if esum:
                    k = "ESum"
                    feed_dict = {
                        'Geant4': np.sum(data_full[:,:100],(1)),
                        'GFlash': np.sum(data_orig,(1,2,3)),
                        full_modelConv_name: np.sum(data_trafo,(1,2,3)),
                    }
                    if generate_plots:
                        fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                            weights = weight_dict,
                                            label_loc= 'best',
                                            xlabel='Total Energy Sum [GeV]', ylabel= 'Normalized entries',
                                            binning = binning_l['esum'],triangle=triangle,
                                            logy=False, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['esum'],
                                            title='10-100 GeV', y_range= y_range_l['esum'])
                        #ax0.set_xscale("log")
                        fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'esum_iter'+ str(i) + '.svg')
                        plt.close()
                    if record_metrics and not(generate_plots):
                        for _,plot in enumerate(feed_dict.keys()):
                            emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])



                ### EMax ###
                if emax:
                    k = "EMax"
                    feed_dict = {
                        'Geant4': np.max(data_full[:,:100],(1)),
                        'GFlash': np.max(data_orig,(1,2,3)),
                        full_modelConv_name: np.max(data_trafo,(1,2,3)),
                    }
                    if generate_plots:
                        fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                            weights = weight_dict,
                                            label_loc= 'best',
                                            xlabel='Brightest Cell Energy [GeV]', ylabel= 'Normalized entries',
                                            binning = binning_l['emax'],triangle=triangle,
                                            logy=False, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['emax'],
                                            title='10-100 GeV', y_range= y_range_l['emax'])
                        #ax0.set_xscale("log")
                        fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'emax_iter'+ str(i) + '.svg')
                        plt.close()
                    if record_metrics and not(generate_plots):
                        for _,plot in enumerate(feed_dict.keys()):
                            emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])



                ### NHit ###
                if nhit:
                    k = "NHit"
                    feed_dict = {
                        'Geant4': np.sum(get_nhit(data_full[:,:100]),(1)),
                        'GFlash': np.sum(get_nhit(data_orig),(1,2,3)),
                        full_modelConv_name: np.sum(get_nhit(data_trafo),(1,2,3)),
                    }
                    if generate_plots:
                        fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                            weights = weight_dict,
                                            label_loc= 'best',
                                            xlabel='# of hits above 1MeV', ylabel= 'Normalized entries',
                                            binning = binning_l['nhit'],triangle=triangle,
                                            logy=False, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['nhit'],
                                            title='10-100 GeV', y_range= y_range_l['nhit'])
                        #ax0.set_xscale("log")
                        fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'nhit_iter'+ str(i) + '.svg')
                        plt.close()
                    if record_metrics and not(generate_plots):
                        for _,plot in enumerate(feed_dict.keys()):
                            emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])


                ### Espec ###
                if espec:
                    k = 'ESpec'
                    feed_dict = {
                        'Geant4': np.reshape(data_full[:,:100], -1),
                        'GFlash': np.reshape(data_orig, -1),
                        full_modelConv_name: np.reshape(data_trafo, -1),
                    }

                    weight_dict = {
                        'Geant4': np.ones(np.reshape(data_full[:,:100], -1).shape[0]),
                        'GFlash': np.ones(np.reshape(data_full[:,:100], -1).shape[0]),
                        full_modelConv_name: np.ones(np.reshape(data_full[:,:100], -1).shape[0]),
                    }
                    if generate_plots:
                        fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                            weights = weight_dict,
                                            label_loc= 'best',
                                            xlabel='Cell Energies [GeV]', ylabel= 'Normalized entries',
                                            binning = binning_l['espec'],
                                            logy=True, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['espec'],
                                            title='10-100 GeV', triangle=triangle, y_range= y_range_l['espec'])
                        ax0.set_xscale("log")
                        fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'espec_iter'+ str(i) + '.svg')
                        plt.close()
                    if record_metrics and not(generate_plots):
                        for _,plot in enumerate(feed_dict.keys()):
                            emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])

                ### EspecNorm ###
                if especnorm:
                    k = 'ESpecNorm'
                    feed_dict = {
                        'Geant4': np.reshape(data_full[:,:100], -1),
                        'GFlash': np.reshape(data_orig, -1),
                        full_modelConv_name: np.reshape(data_trafo, -1),
                    }

                    weight_dict = {
                        'Geant4': np.ones(np.reshape(data_full[:,:100], -1).shape[0])/np.reshape(data_full[:,:100], -1).shape[0],
                        'GFlash': np.ones(np.reshape(data_full[:,:100], -1).shape[0])/np.reshape(data_full[:,:100], -1).shape[0],
                        full_modelConv_name: np.ones(np.reshape(data_full[:,:100], -1).shape[0])/np.reshape(data_full[:,:100], -1).shape[0],
                    }
                    if generate_plots:
                        fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                            weights = weight_dict,
                                            label_loc= 'best', density=False,
                                            xlabel='Cell Energies', ylabel= 'Normalized entries',
                                            binning = binning_l['espec'],
                                            logy=True, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['espec'],
                                            title='10-100 GeV', triangle=triangle, y_range= y_range_l['espec'])
                        ax0.set_xscale("log")
                        fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'espec_norm_iter'+ str(i) + '.svg')
                        plt.close()
                    if record_metrics and not(generate_plots):
                        for _,plot in enumerate(feed_dict.keys()):
                            emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])


                ### Ex ###
                if ex:
                    k = 'Ex'
                    weight_dict = {
                        'Geant4': np.reshape(np.sum(np.reshape(data_full[:,:100], (-1, 10, 10)), 2), -1)/data_full.shape[0],
                        'GFlash': np.reshape(np.sum(np.reshape(data_orig, (-1, 10, 10)), 2), -1)/data_full.shape[0],
                        full_modelConv_name: np.reshape(np.sum(np.reshape(data_trafo, (-1, 10, 10)), 2), -1)/data_full.shape[0],
                    }

                    feed_dict = {
                        'Geant4': np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                        'GFlash': np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                        full_modelConv_name: np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                    }
                    if generate_plots:
                        fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                            weights = weight_dict,
                                            label_loc= 'best', density=False,
                                            xlabel='x Profile', ylabel= 'Mean Energy [GeV]',
                                            binning = binning_l['ex'],triangle=triangle,
                                            logy=True, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['ex'],
                                            title='10-100 GeV', y_range= y_range_l['ex'])
                        #ax0.set_xscale("log")
                        fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'ex_iter'+ str(i) + '.svg')
                        plt.close()
                    if record_metrics and not(generate_plots):
                        for _,plot in enumerate(feed_dict.keys()):
                            emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])

                ### Ey ###
                if ey:
                    k = 'Ey'
                    weight_dict = {
                        'Geant4': np.reshape(np.sum(np.reshape(data_full[:,:100], (-1, 10, 10)), 1), -1)/data_full.shape[0],
                        'GFlash': np.reshape(np.sum(np.reshape(data_orig, (-1, 10, 10)), 1), -1)/data_full.shape[0],
                        full_modelConv_name: np.reshape(np.sum(np.reshape(data_trafo, (-1, 10, 10)), 1), -1)/data_full.shape[0],
                    }

                    feed_dict = {
                        'Geant4': np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                        'GFlash': np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                        full_modelConv_name: np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                    }
                    if generate_plots:
                        fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                            weights = weight_dict,
                                            label_loc= 'best', density=False,
                                            xlabel='y Profile', ylabel= 'Mean Energy [GeV]',
                                            binning = binning_l['ex'],triangle=triangle,
                                            logy=True, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['ex'],
                                            title='10-100 GeV', y_range= y_range_l['ex'])
                        #ax0.set_xscale("log")
                        fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'ey_iter'+ str(i) + '.svg')
                        plt.close()
                    if record_metrics and not(generate_plots):
                        for _,plot in enumerate(feed_dict.keys()):
                            emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])

                # if generate_plots:
                #     fig,ax0 = scatter(xdata=energy_gflash_trafo[:10000,0], ydata=energy_gflash_trafo[:10000,1],
                #                         data_label = full_modelEnergy_name,
                #                         xlabel='$e_{GF}$',ylabel='$e_{refined}$',
                #                         label_loc='best', title='10-100 GeV')
                #     fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'scatter_iter'+ str(i) + '.svg')
                    plt.close()
            
            print(" - Done 0-100 GeV")

        
            if energy_intervals:

                shower_dict = {}
                for energy in [20, 50, 80]:
                    
                    #------------------------------------------------------------
                    # Discard last sampling
                    #------------------------------------------------------------
                    # free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
                    # mem_used_mb = (total - free) / 1024 ** 2
                    # print(f"Before discard sampling: {mem_used_mb}")
                    
                    sample = None
                    gc.collect()

                    # free, total = torch.cuda.mem_get_info(torch.device('cuda:0'))
                    # mem_used_mb = (total - free) / 1024 ** 2
                    # print(f"After discard sampling: {mem_used_mb}")
                    #------------------------------------------------------------
                    #------------------------------------------------------------


                    file_path_gflash = data_dir_path + '/run_GFlash01_100k_{:d}GeV_full.npy'.format(energy)
                    file_path_g4 = data_dir_path + '/run_Geant_100k_{:d}GeV_full.npy'.format(energy)

                    # data = load_data(file_path_gflash, file_path_g4, normalize_energy=True, shuffle=False, plotting=True)
                    
                    ## -------- Load Data --------- ##

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

                    energy_voxel_g4 = (energy_voxel_g4 - shifter_g4)/scaler_g4
                    energy_voxel_gflash = (energy_voxel_gflash - shifter_gflash)/scaler_gflash

                    energy_g4 = (energy_g4 - shifter_energy_g4)/scaler_energy_g4
                    energy_gflash = (energy_gflash - shifter_energy_gflash)/scaler_energy_gflash

                    data = {}

                    data['energy_voxel_g4'] = energy_voxel_g4
                    data['energy_voxel_gflash'] = energy_voxel_gflash
                    data['energy_gflash'] = energy_gflash
                    data['energy_g4'] = energy_g4
                    data['energy_particle_gflash'] = energy_particle_gflash
                    data['energy_particle_g4'] = energy_particle_g4
                    data['shifter_energy_g4'] = shifter_energy_g4 
                    data['shifter_energy_gflash'] = shifter_energy_gflash 
                    data['scaler_energy_g4'] = scaler_energy_g4 
                    data['scaler_energy_gflash'] = scaler_energy_gflash 
                    data['shifter_energy_fullrange_g4'] = shifter_energy_fullrange_g4
                    data['shifter_energy_fullrange_gflash'] = shifter_energy_fullrange_gflash
                    data['scaler_energy_fullrange_g4'] = scaler_energy_fullrange_g4
                    data['scaler_energy_fullrange_gflash'] = scaler_energy_fullrange_gflash
                    data['shifter_g4'] = shifter_g4
                    data['shifter_gflash'] = shifter_gflash
                    data['scaler_g4'] = scaler_g4
                    data['scaler_gflash'] = scaler_gflash
        

                    ## -------- Load Data --------- ##
                    if modelConv_type in ["Bernstein", "Gram", "Jacobi"]:
                        batch_size = 5000
                    else:
                        batch_size = 25000

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

                    dls = {'f': init_dl, 'b': final_dl}

                    #energy_voxel_gflash_orig, energy_voxel_gflash_trafo, energy_gflash_trafo, _ = sample(forward_or_backward = 'f', forward_or_backward_rev = 'b')
                    # energy_voxel_gflash_orig, energy_voxel_gflash_trafo, energy_gflash_trafo, energy_particle = sample(forward_or_backward = 'f', forward_or_backward_rev = 'b')
                    print(device)
                    sample = sample_data(dls, data, netsEnergy, netsConv, de, d,
                                    num_steps_voxel, num_steps_energy,
                                    gammas_voxel, gammas_energy, device,
                                    forward_or_backward = 'f', forward_or_backward_rev = 'b',
                                    full_sample = full_sample)

                    energy_voxel_gflash_orig = sample['energy_voxel_gflash_orig']
                    energy_voxel_gflash_trafo = sample['energy_voxel_gflash_trafo']
                    energy_gflash_trafo = sample['energy_gflash_trafo']
                    energy_particle = sample['energy_particle']

                    data_trafo = energy_voxel_gflash_trafo[:,-1]#*100

                    data_orig = energy_voxel_gflash_orig
                    data_full = np.load(file_path_g4)

                    data_full[data_full<0] = 0.0
                    data_orig[data_orig<0] = 0.0
                    data_trafo[data_trafo<0] = 0.0

                    shower_dict[energy] = [data_full, data_orig, data_trafo, energy_gflash_trafo]

                print(" - Done loading shower dicts")
                
                for energy in [20, 50, 80]:

                    [data_full, data_orig, data_trafo, energy_gflash_trafo] = shower_dict[energy]


                    weight_dict = {
                        'Geant4': np.ones(data_full.shape[0]),
                        'GFlash': np.ones(data_full.shape[0]),
                        full_modelConv_name: np.ones(data_full.shape[0]),
                        full_modelEnergy_name: np.ones(data_full.shape[0]),
                    }
                    triangle=True

                    y_lim_ratio_l = {'esum': [0.0, 2.0],
                                    'emax': [0.0, 2.0],
                                    'nhit': [0.0, 2.0],
                                    'espec': [0.0, 2.0],
                                    'ex': [0.0, 2.0],
                                    }

                    y_range_l = {'esum_20': None,
                                'esum_50': None,
                                'esum_80': None,
                                'emax_20': None,
                                'emax_50': None,
                                'emax_80': None,
                                'nhit_20': None,
                                'nhit_50': None,
                                'nhit_80': None,
                                'espec': [1e-6, 1e5],
                                'ex': [8e-3, 5e2],
                            }



                    binning_l = {'esum_20': np.linspace(19.5,20.1,50),
                                'esum_50': np.linspace(48.5,50.1,50),
                                'esum_80': np.linspace(77.5,80.1,50),
                                'emax_20': np.linspace(4.5, 6.5,50),
                                'emax_50': np.linspace(11,15,50),
                                'emax_80': np.linspace(18.5,22,50),
                                'nhit_20': np.linspace(0, 100,101),
                                'nhit_50': np.linspace(0, 100,101),
                                'nhit_80': np.linspace(0, 100,101),
                                'espec': np.logspace(-4, 1.5, 100),
                                'ex': np.linspace(0, 10,11),
                                }

                    # print(data_full.shape)

                    ### Esum 1D ###
                    if esum1d:
                        k = f"Esum1D_{energy}GeV"
                        feed_dict = {
                            'Geant4': np.sum(data_full[:,:100],(1)),
                            'GFlash': np.sum(data_orig,(1,2,3)),
                            full_modelEnergy_name: energy_gflash_trafo[:,1],
                        }
                        if generate_plots:
                            fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                                weights = weight_dict,
                                                label_loc= 'best',
                                                xlabel='Total Energy Sum [GeV]', ylabel= 'Normalized entries',
                                                binning = binning_l['esum_{:d}'.format(energy)] ,triangle=triangle,
                                                logy=False, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['esum'],
                                                title='{:d} GeV'.format(energy), y_range= y_range_l['esum_{:d}'.format(energy)])
                            #ax0.set_xscale("log")
                            fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'esum_1D_{:d}GeV_iter'.format(energy) + str(i) + '.svg')
                            plt.close()
                        if record_metrics and not(generate_plots):
                            for _,plot in enumerate(feed_dict.keys()):
                                emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])

                    #break
                    if full_model_metrics:
                        ### Esum ###
                        if esum:
                            k = f"ESum_{energy}GeV"
                            feed_dict = {
                                'Geant4': np.sum(data_full[:,:100],(1)),
                                'GFlash': np.sum(data_orig,(1,2,3)),
                                full_modelConv_name: np.sum(data_trafo,(1,2,3)),
                            }
                            if generate_plots:
                                fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                                    weights = weight_dict,
                                                    label_loc= 'best',
                                                    xlabel='Total Energy Sum [GeV]', ylabel= 'Normalized entries',
                                                    binning = binning_l['esum_{:d}'.format(energy)] ,triangle=triangle,
                                                    logy=False, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['esum'],
                                                    title='{:d} GeV'.format(energy), y_range= y_range_l['esum_{:d}'.format(energy)])
                                #ax0.set_xscale("log")
                                fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'esum_{:d}GeV_iter'.format(energy) + str(i) + '.svg')
                                plt.close()
                            if record_metrics and not(generate_plots):
                                for _,plot in enumerate(feed_dict.keys()):
                                    emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])

                        ### EMax ###
                        if emax:
                            k = f"EMax_{energy}GeV"
                            feed_dict = {
                                'Geant4': np.max(data_full[:,:100],(1)),
                                'GFlash': np.max(data_orig,(1,2,3)),
                                full_modelConv_name: np.max(data_trafo,(1,2,3)),
                            }
                            if generate_plots:
                                fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                                    weights = weight_dict,
                                                    label_loc= 'best',
                                                    xlabel='Brightest Cell Energy [GeV]', ylabel= 'Normalized entries',
                                                    binning = binning_l['emax_{:d}'.format(energy)],triangle=triangle,
                                                    logy=False, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['emax'],
                                                    title='{:d} GeV'.format(energy), y_range= y_range_l['emax_{:d}'.format(energy)])
                                #ax0.set_xscale("log")
                                fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'emax_{:d}GeV_iter'.format(energy) + str(i) + '.svg')
                                plt.close()
                            if record_metrics and not(generate_plots):
                                for _,plot in enumerate(feed_dict.keys()):
                                    emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])


                        ### NHit ###
                        if nhit:
                            k = f"NHit_{energy}GeV"
                            feed_dict = {
                                'Geant4': np.sum(get_nhit(data_full[:,:100]),(1)),
                                'GFlash': np.sum(get_nhit(data_orig),(1,2,3)),
                                full_modelConv_name: np.sum(get_nhit(data_trafo),(1,2,3)),
                            }
                            if generate_plots:
                                fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                                    weights = weight_dict,
                                                    label_loc= 'best',
                                                    xlabel='# of hits above 1MeV', ylabel= 'Normalized entries',
                                                    binning = binning_l['nhit_{:d}'.format(energy)],triangle=triangle,
                                                    logy=False, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['nhit'],
                                                    title='{:d} GeV'.format(energy), y_range= y_range_l['nhit_{:d}'.format(energy)])
                                #ax0.set_xscale("log")
                                fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'nhit_{:d}GeV_iter'.format(energy) + str(i) + '.svg')
                                plt.close()
                            if record_metrics and not(generate_plots):
                                for _,plot in enumerate(feed_dict.keys()):
                                    emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])


                        ### ESpec ###
                        if espec:
                            k = f"ESpec_{energy}GeV"
                            feed_dict = {
                                'Geant4': get_espec(np.reshape(data_full[:,:100], -1)),
                                'GFlash': get_espec(np.reshape(data_orig, -1)),
                                full_modelConv_name: get_espec(np.reshape(data_trafo, -1)),
                            }

                            weight_dict = {
                                'Geant4': np.ones(np.reshape(data_full[:,:100], -1).shape[0]),
                                'GFlash': np.ones(np.reshape(data_full[:,:100], -1).shape[0]),
                                full_modelConv_name: np.ones(np.reshape(data_full[:,:100], -1).shape[0]),
                            }
                            if generate_plots:
                                fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                                    weights = weight_dict,
                                                    label_loc= 'best',
                                                    xlabel='Cell Energies [GeV]', ylabel= 'Normalized entries',
                                                    binning = binning_l['espec'],
                                                    logy=True, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['espec'],
                                                    title='{:d} GeV'.format(energy), triangle=triangle, y_range= y_range_l['espec'])
                                ax0.set_xscale("log")
                                fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'espec_{:d}GeV_iter'.format(energy) + str(i) + '.svg')
                                plt.close()
                            if record_metrics and not(generate_plots):
                                for _,plot in enumerate(feed_dict.keys()):
                                    emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])


                        ### ex ###
                        if ex:
                            k = f"Ex_{energy}GeV"
                            weight_dict = {
                                'Geant4': np.reshape(np.sum(np.reshape(data_full[:,:100], (-1, 10, 10)), 2), -1)/data_full.shape[0],
                                'GFlash': np.reshape(np.sum(np.reshape(data_orig, (-1, 10, 10)), 2), -1)/data_full.shape[0],
                                full_modelConv_name: np.reshape(np.sum(np.reshape(data_trafo, (-1, 10, 10)), 2), -1)/data_full.shape[0],
                            }

                            feed_dict = {
                                'Geant4': np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                                'GFlash': np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                                full_modelConv_name: np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                            }
                            if generate_plots:
                                fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                                    weights = weight_dict,
                                                    label_loc= 'best', density=False,
                                                    xlabel='x Profile', ylabel= 'Mean Energy [GeV]',
                                                    binning = binning_l['ex'],triangle=triangle,
                                                    logy=True, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['ex'],
                                                    title='{:d} GeV'.format(energy), y_range= y_range_l['ex'])
                                #ax0.set_xscale("log")
                                fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'ex_{:d}GeV_iter'.format(energy) + str(i) + '.svg')
                                plt.close()
                            if record_metrics and not(generate_plots):
                                for _,plot in enumerate(feed_dict.keys()):
                                    emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])


                        ### ey ###
                        if ey:
                            k = f"Ey_{energy}GeV"
                            weight_dict = {
                                'Geant4': np.reshape(np.sum(np.reshape(data_full[:,:100], (-1, 10, 10)), 1), -1)/data_full.shape[0],
                                'GFlash': np.reshape(np.sum(np.reshape(data_orig, (-1, 10, 10)), 1), -1)/data_full.shape[0],
                                full_modelConv_name: np.reshape(np.sum(np.reshape(data_trafo, (-1, 10, 10)), 1), -1)/data_full.shape[0],
                            }

                            feed_dict = {
                                'Geant4': np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                                'GFlash': np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                                full_modelConv_name: np.tile(np.arange(0.5, 9.6, 1), data_full.shape[0]),
                            }
                            if generate_plots:
                                fig,ax0,emds,errs = histogram(feed_dict, emds, errs, line_style, colors,
                                                    weights = weight_dict,
                                                    label_loc= 'best', density=False,
                                                    xlabel='y Profile', ylabel= 'Mean Energy [GeV]',
                                                    binning = binning_l['ex'],triangle=triangle,
                                                    logy=True, reference_name='Geant4', y_lim_ratio = y_lim_ratio_l['ex'],
                                                    title='{:d} GeV'.format(energy), y_range= y_range_l['ex'])
                                #ax0.set_xscale("log")
                                fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'ey_{:d}GeV_iter'.format(energy) + str(i) + '.svg')
                                plt.close()
                            if record_metrics and not(generate_plots):
                                for _,plot in enumerate(feed_dict.keys()):
                                    emds[f"{k}_{plot}"],errs[f"{k}_{plot}"] = emd(feed_dict['Geant4'][:100000],feed_dict[plot][:100000],weight_dict[plot][:100000])

                        # if generate_plots:
                        #     fig,ax0 = scatter(xdata=energy_gflash_trafo[:10000,0], ydata=energy_gflash_trafo[:10000,1],
                        #                             data_label = full_modelEnergy_name,
                        #                             xlabel='$e_{GF}$',ylabel='$e_{refined}$',
                        #                             label_loc='best',title='{:d} GeV'.format(energy))
                        #     fig.savefig(plots_dir_path + f"{full_modelEnergy_name}_{full_modelConv_name}_" + 'scatter_{:d}GeV_iter'.format(energy) + str(i) + '.svg')
                            plt.close()
                    
                    print(f" - Done {energy} GeV")

            if record_metrics:
                # print(f"EMD: {emds} - Error: {err}")
                metrics = pd.concat([metrics, pd.DataFrame(emds, index=[i])])
                errors = pd.concat([errors, pd.DataFrame(errs, index=[i])])


        if record_metrics:
            if full_model_metrics:
                # metrics["Iteration"] = metrics.index + 1
                # metrics["ESum1D"] = metrics[f"Esum1D_{full_modelEnergy_name}"] - metrics["Esum1D_Geant4"]
                # metrics["ESumFrac1D"] = metrics[f"EsumFrac1D_{full_modelEnergy_name}"] - metrics["EsumFrac1D_Geant4"]
                # metrics["ESum"] = metrics[f"ESum_{full_modelConv_name}"] - metrics["ESum_Geant4"]
                # metrics["EMax"] = metrics[f"EMax_{full_modelConv_name}"] - metrics["EMax_Geant4"]
                # metrics["NHit"] = metrics[f"NHit_{full_modelConv_name}"] - metrics["NHit_Geant4"]
                # metrics["ESpec"] = metrics[f"ESpec_{full_modelConv_name}"] - metrics["ESpec_Geant4"]
                # metrics["ESpecNorm"] = metrics[f"ESpecNorm_{full_modelConv_name}"] - metrics["ESpecNorm_Geant4"]
                # metrics["Ex"] = metrics[f"Ex_{full_modelConv_name}"] - metrics["Ex_Geant4"]
                # metrics["Ey"] = metrics[f"Ey_{full_modelConv_name}"] - metrics["Ey_Geant4"]
                metrics["En_Inference_Time"] = np.mean(netsEnergy_ts)
                metrics["Conv_Inference_Time"] = np.mean(netsConv_ts)

                metrics.to_csv(f"{metrics_dir_path}/emds/{modelEnergy_type}_{modelEnergy_version}_{modelConv_type}_{modelConv_version}_metrics.csv")
                errors.to_csv(f"{metrics_dir_path}/errors/{modelEnergy_type}_{modelEnergy_version}_{modelConv_type}_{modelConv_version}_errors.csv")

            else:
                metrics["Iteration"] = metrics.index + 1
                # metrics["ESum1D"] = metrics[f"Esum1D_{full_modelEnergy_name}"] - metrics["Esum1D_Geant4"]
                # metrics["ESumFrac1D"] = metrics[f"EsumFrac1D_{full_modelEnergy_name}"] - metrics["EsumFrac1D_Geant4"]
                metrics["En_Inference_Time"] = np.mean(netsEnergy_ts)
                metrics["Conv_Inference_Time"] = np.mean(netsConv_ts)

                metrics.to_csv(f"{metrics_dir_path}/emds/{modelEnergy_type}_{modelEnergy_version}_metrics.csv")
                errors.to_csv(f"{metrics_dir_path}/errors/{modelEnergy_type}_{modelEnergy_version}_errors.csv")
    
    # torch.cuda.empty_cache()
    return "Done!"
