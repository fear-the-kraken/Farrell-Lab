#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for data handling and plotting tasks

@author: amandaschott
"""
import sys
import scipy
import numpy as np
import matplotlib.colors as mcolors
import colorsys
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
import pdb

##################################################
########         GENERAL FUNCTIONS        ########
##################################################

def Downsample(arr, n):
    """ Downsample 2D input $arr by factor of $n """
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def Normalize(collection):
    """ Normalize data between 0 and 1 """
    Max=max(collection)
    Min=min(collection)
    return np.array([(i-Min)/(Max-Min) for i in collection])

def Closest(num, collection):
    """ Return the $collection value closest to $num """
    return min(collection,key=lambda x:abs(x-num))

def IdxClosest(num, collection):
    """ Return index of the $collection value closest to $num """
    return list(collection).index(Closest(num,collection))

def Edges(collection):
    """ Return (first,last) values in collection """
    return (collection[0], collection[-1])

def MinMax(collection):
    """ Return (min,max) values in collection, excluding NaNs """
    return (np.nanmin(collection), np.nanmax(collection))

def Limit(collection, mode=2, pad=0.01):
    """ Return lower and/or upper data limits of collection plus padding """
    if np.array(collection).size > 0:
        min_max = MinMax(collection)
        vpad = np.ptp(collection[~np.isnan(collection)]) * pad
        vmin, vmax = np.add(min_max, (-vpad, vpad))
    else:
        vmin, vmax = None,None
    if   mode==0 : return vmin
    elif mode==1 : return vmax
    elif mode==2 : return (vmin, vmax)


def SymLimit(collection, pad=0.0):
    """ Return (negative, positive) maximum absolute value in collection """
    abs_max = np.nanmax(np.abs(MinMax(collection)))
    return (-abs_max, abs_max)
    
    
def CenterWin(collection, n, total=True):
    """ Return window of $n values surrounding central point in collection """
    ctr = int(round(len(collection)/2))
    N = int(round(n)) if total==True else int(round(n*2))
    nwin = int(n/2)
    ii = np.arange(ctr-nwin, ctr+nwin)
    if N % 2 > 0:
        ii = np.append(ii, ctr+nwin)
    return ii, collection[ii]


##################################################
########         SIGNAL PROCESSING        ########
##################################################


def butter_bandpass(lowcut, highcut, lfp_fs, order=3):
    """ Return filter coefficients for given freq. cutoffs/sampling rate """
    nyq = 0.5 * lfp_fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, lfp_fs, order=3, axis=-1):
    """ Return bandpass-filtered data arrays """
    b, a = butter_bandpass(lowcut, highcut, lfp_fs, order=order)
    y = scipy.signal.filtfilt(b, a, data, axis=axis)
    return y


##################################################
########       MATPLOTLIB FUNCTIONS       ########
##################################################


def Cmap(data, cmap=plt.cm.coolwarm, alpha=1.0):
    """ Return RGBA array (N x 4) of colors mapped from $data values """
    try:
        normal = plt.cm.colors.Normalize(np.nanmin(data), np.nanmax(data))
        arr = cmap(normal(data))
        arr[:, 3] = alpha
        return arr
    except:
        return np.ones((len(data), 4))
    
def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_rgb(c, cscale=255):
    """ Convert input color to RGB values, scaled from 0-255 or 0-1 """
    if isinstance(c, str): 
        c = mcolors.to_rgb(c)  # string --> rgb
    rgb = np.array(c, dtype='float')
    if any(rgb > 1):
        rgb /= 255  # scale 0-1
    if cscale == 255:
        rgb = np.round(rgb * 255).astype('int')  # scale 0-255
    return rgb
    

def hue(c, percent, mode=1, cscale=255, alpha=1, res='tuple'):
    """ Adjust input color tint (mode=1), shade (0), or saturation (0.5) """
    rgb = get_rgb(c, cscale=1)[0:3]
    if mode == 1     : tg = np.array([1., 1., 1.])  # lighten color
    elif mode == 0   : tg = np.array([0., 0., 0.])  # darken color
    elif mode == 0.5 : tg = rgb.mean()              # de-intensify color
    distance = tg - rgb
    step = distance * percent
    adj_c = np.array(list(rgb + step) + [alpha])
        
    if cscale == 255:
        adj_c = np.round(adj_c * 255).astype('int')
    if res == 'tuple' : return tuple(adj_c)
    elif res == 'hex' : return mcolors.to_hex(adj_c, keep_alpha=True)
    else              : return adj_c
    
    
def rand_hex(n=1, bright=True):
    """ Return $n random colors in hex code format """
    hue = np.random.uniform(size=n)
    if bright:
        l_lo, l_hi, s_lo, s_hi = [0.4, 0.6, 0.7, 1.0]
    else:
        l_lo, l_hi, s_lo, s_hi = [0.0, 1.0, 0.0, 1.0]
    lightness = np.random.uniform(low=l_lo, high=l_hi, size=n)
    saturation = np.random.uniform(low=s_lo, high=s_hi, size=n)
    
    hex_list = []
    for h,l,s in zip(hue,lightness,saturation):
        hexc = mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))
        hex_list.append(hexc)
    if n==1:
        hex_list = hex_list[0]
    return hex_list
    
    
def match_ylimits(*axs, set_ylim=True):
    """ Standardize y-axis limits across plots """
    ylim = MinMax(np.concatenate([ax.get_ylim() for ax in axs]))
    if set_ylim:
        for ax in axs:
            ax.set_ylim(ylim)
    return ylim


def add_legend_items(leg, new_items):
    """ Add new Matplotlib item(s) to existing legend """
    try    : new_items.__iter__()
    except : new_items = [new_items]
    handles = leg.legend_handles + list(new_items)
    labels = [handle.get_label() for handle in handles]
    title = leg.get_title().get_text()
    leg._init_legend_box(handles, labels)
    leg.set_title(title)
    return leg


##################################################
########           PYQT WIDGETS           ########
##################################################


class DividerLine(QtWidgets.QFrame):
    """ Basic horizontal (or vertical) separator line """
    def __init__(self, orientation='h', lw=3, mlw=3, parent=None):
        super().__init__(parent)
        if orientation == 'h':
            self.setFrameShape(QtWidgets.QFrame.HLine)
        else:
            self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setLineWidth(lw)
        self.setMidLineWidth(mlw)
        
        
def InterWidgets(parent, orientation='v'):
    """ Return layout with an intermediate widget between $parent and children """
    # parent -> interlayout > interwidget > layout
    interlayout = QtWidgets.QVBoxLayout(parent)
    interlayout.setContentsMargins(0,0,0,0)
    interwidget = QtWidgets.QWidget()
    interwidget.setContentsMargins(0,0,0,0)
    if orientation   == 'v' : layout = QtWidgets.QVBoxLayout(interwidget)
    elif orientation == 'h' : layout = QtWidgets.QHBoxLayout(interwidget)
    else                    : layout = QtWidgets.QGridLayout(interwidget)
    interlayout.addWidget(interwidget)
    return interlayout, interwidget, layout
        

def ScreenRect(perc_width=1, perc_height=1, keep_aspect=True):
    """ Return QRect box centered and scaled relative to screen geometry """
    screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
    if keep_aspect:
        perc_width, perc_height = [min([perc_width, perc_height])] * 2
    app_width = int(screen_rect.width() * perc_width)
    app_height = int(screen_rect.height() * perc_height)
    app_x = int((screen_rect.width() - app_width) / 2)
    app_y = int((screen_rect.height() - app_height) / 2)
    qrect = QtCore.QRect(app_x, app_y, app_width, app_height)
    return qrect


def get_ddir():
    app = QtWidgets.QApplication(sys.argv)
    res = QtWidgets.QFileDialog().getExistingDirectory(None, '', '', 
                                                       QtWidgets.QFileDialog.DontUseNativeDialog)
    return res
    


# if __name__ == '__main__':
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     app.setStyle('Fusion')
    
#     #w = hippos()
#     w = QtWidgets.QFileDialog().getExistingDirectory(None, '', '', QtWidgets.QFileDialog.DontUseNativeDialog)
    
#     w.show()
#     w.raise_()
    
#     res = w.exec()
#     #sys.exit(app.exec())
