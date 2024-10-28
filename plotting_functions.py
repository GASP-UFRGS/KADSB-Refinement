import numpy as np

from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm



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
        
def histogram(feed_dict,line_style,colors,xlabel='',ylabel='',reference_name='Truth',logy=False,binning=None,
              label_loc='best',plot_ratio=False,weights=None,uncertainty=None,triangle=True,
              y_lim_ratio=[0.5,1.5], title='', density=True, y_range=None):
    
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
            print(plot)
            d,err = emd(feed_dict[reference_name][:100000],feed_dict[plot][:100000],weights[plot][:100000])
            print("EMD distance is: {}+-{}".format(d,err))
            #d = get_triangle_distance(dist,reference_hist,binning)
            #print("Triangular distance is: {0:.2g}".format(d))
            if not reference_name == plot:
                #label_list.append(plot + ' EMD: {0:.2g} $\pm$ {1:.2g}'.format(d, err))
                #label_list.append(plot)
                d_list.append(d)
                #label_list.append(' $\pm$ ')
                err_list.append(err)
                
            if reference_name == plot:
                #label_list.append(plot + ' EMD: {0:.2g} $\pm$ {1:.2g}'.format(d, err))
                #label_list.append(plot)
                d_list.append(d)
                #label_list.append(' $\pm$ ')
                err_list.append(err)
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
    
    print('{}, {} , {}: '.format(xlabel, title, plot) + out)
    
    out = ' ' 
    for ip,plot in enumerate(feed_dict.keys()):
        temp = '{0:.1g}'.format(err_list[ip])
        if 'e' in temp:
            precision = temp[-1]
        else:
            precision = (len(temp))-2
        out = out + (' & {0:.{1}f}'.format(d_list[ip], precision)) + '(' + ('{0:.{1}f})'.format(err_list[ip], precision))[-2:]
    
    print('{}, {} , {}: '.format(xlabel, title, plot) + out)
    

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
    return fig,ax0

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
