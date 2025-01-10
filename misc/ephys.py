#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 04:00:25 2024

@author: amandaschott
"""
import os
from pathlib import Path
import scipy.io as so
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pickle
from PyQt5 import QtWidgets
from open_ephys.analysis import Session
import probeinterface as prif
import pdb
# custom modules
import pyfx


##################################################
########          FILE MANAGEMENT         ########
##################################################


def get_recording_path(ppath='', rec='', txt='Select recording folder'):
    """ Return base directory and recording folder name """
    if os.path.exists(os.path.join(ppath, rec)):
        return ppath, rec
    _ = QtWidgets.QApplication([])
    init_ddir = ppath if os.path.exists(ppath) else os.getcwd()
    ddir = QtWidgets.QFileDialog.getExistingDirectory(None, txt, init_ddir,
                                                      QtWidgets.QFileDialog.ShowDirsOnly)
    dir_list = ddir.split(os.sep)
    rec = dir_list.pop(-1)
    ppath = os.sep.join(dir_list)
    return ppath, rec


def base_dirs(return_keys=False):
    """ Return default data directories saved in base_folders.txt """
    # Mode 0 for paths only, 1 for keys only
    with open('base_folders.txt', 'r') as fid:
        keys,vals = zip(*[map(str.strip, l.split('=')) for l in fid.readlines()])
    if not return_keys:
        return vals
    return list(zip(keys,vals))


def write_base_dirs(ddir_list):
    """ Save input directories to base_folders.txt """
    assert len(ddir_list) == 3
    with open('base_folders.txt', 'w') as fid:
        for k,path in zip(['RAW_DATA','PROCESSED_DATA','PROBE_FILES'], ddir_list):
            fid.write(k + ' = ' + str(path) + '\n')

def read_param_file(filepath='default_params.txt'):
    """ Return dictionary of parameters loaded from .txt file """
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    lines = [l for l in lines if not l.startswith('#') and len(l) > 0]
    
    PARAMS = {}
    for line in lines:
        d = line.split(';')[0]
        k,v = [x.strip() for x in d.split('=')]
        if v.startswith('[') and v.endswith(']'):
            val = [float(x.strip()) for x in v[1:-1].split(',')]
        else:
            try    : val = float(v)
            except : val = str(v)
        if   val == 'True'  : val = True
        elif val == 'False' : val = False
        PARAMS[k] = val
    return PARAMS


def write_param_file(PARAMS, filepath='default_params.txt'):
    """ Save parameter dictionary to .txt file """
    fid = open(filepath, 'w')
    fid.write('###  PARAMETERS  ###' + os.linesep)
    fid.write(os.linesep)
    for k,v in PARAMS.items():
        fid.write(f'{k} = {v};' + os.linesep)
    fid.close()
    
def load_recording_info(ddir):
    """ Load info dictionary from processed recording """
    INFO = pd.Series(pickle.load(open(Path(ddir, 'info.pkl'), 'rb')))
    return INFO

def load_recording_params(ddir):
    """ Load param dictionary from processed recording """
    PARAMS = pd.Series(pickle.load(open(Path(ddir, 'params.pkl'), 'rb')))
    return PARAMS

def save_recording_info(ddir, INFO):
    """ Save recording metadata to processed data folder """
    with open(Path(ddir, 'info.pkl'), 'wb') as f:
        pickle.dump(INFO, f)

def save_recording_params(ddir, PARAMS):
    """ Save param values used to analyze processed data """
    PARAMS = dict(PARAMS)
    if 'RAW_DATA_FOLDER' in PARAMS.keys()       : del PARAMS['RAW_DATA_FOLDER']
    if 'PROCESSED_DATA_FOLDER' in PARAMS.keys() : del PARAMS['PROCESSED_DATA_FOLDER']
    with open(Path(ddir, 'params.pkl'), 'wb') as f:
        pickle.dump(PARAMS, f)


def get_openephys_session(ddir):
    """ Return top-level Session object of recording directory $ddir """
    session = None
    child_dir = str(ddir)
    while True:
        parent_dir = os.path.dirname(child_dir)
        if os.path.samefile(parent_dir, child_dir):
            break
        if 'settings.xml' in os.listdir(parent_dir):
            session_ddir = os.path.dirname(parent_dir)
            try:
                session = Session(session_ddir) # top-level folder 
            except OSError:
                session = Session(parent_dir)   # recording node folder
            break
        else:
            child_dir = str(parent_dir)
    return session
    

def oeNodes(session, ddir):
    """ Return Open Ephys nodes from parent $session to child recording """
    # session is first node in path
    objs = {'session' : session}
    
    def isPar(par, ddir):
        return os.path.commonpath([par]) == os.path.commonpath([par, ddir])
    # find recording node in path
    if hasattr(session, 'recordnodes'):
        for node in session.recordnodes:
            if isPar(node.directory, ddir):
                objs['node'] = node
                break
        recs = node.recordings
    else:
        recs = session.recordings
    # find recording folder with raw data files
    for recording in recs:
        if os.path.samefile(recording.directory, ddir):
            objs['recording'] = recording
            break
    objs['continuous'] = recording.continuous
    return objs


def load_lfp(ddir, key='', iprb=-1):
    """ Load LFP signals, timestamps, and sampling rate """
    DATA = load_bp(ddir, key=key, iprb=iprb)
    lfp_time = np.load(Path(ddir, 'lfp_time.npy'))
    lfp_fs = int(np.load(Path(ddir, 'lfp_fs.npy')))
    return DATA, lfp_time, lfp_fs


def load_ds_csd(ddir, iprb):
    """ Load or create dictionary of CSD data for probe $iprb """
    # load DS CSDs
    if os.path.exists(Path(ddir, f'ds_csd_{iprb}.npz')):
        with np.load(Path(ddir,f'ds_csd_{iprb}.npz'), allow_pickle=True) as npz:
            csd_dict = dict(npz)
    else:
        csd_dict = dict.fromkeys(['raw_csd','filt_csd','norm_filt_csd','csd_chs'])
    return csd_dict
    

def load_bp(ddir, key='', iprb=-1):
    """ Load bandpass-filtered LFP data for 1 or more probes """
    nprobes = len(prif.io.read_probeinterface(str(Path(ddir, 'probe_group'))).probes)
    #nprobes = load_recording_info(ddir).nprobes
    data_list = [{} for _ in range(nprobes)]
    # load dictionary from npz file
    with np.load(Path(ddir,'lfp_bp.npz'), allow_pickle=True) as npz:
        keys = list(npz.keys())
        for k,v in npz.items():
            if key in keys and k != key:    # skip non-matching keys
                continue
            for i in range(nprobes):
                if k==key:
                    try:
                        data_list[i] = v[i]  # arrays at key $k
                    except:
                        pdb.set_trace()
                else:
                    data_list[i][k] = v[i]  # dict with all keys
    if 0 <= iprb < len(data_list):
        return data_list[iprb]
    else:
        return data_list


def csv2list(ddir, f=''):
    """ Return list of dataframes from keyed .csv file """
    ddf = pd.read_csv(Path(ddir, f))
    llist = [x.droplevel(0) for _,x in ddf.groupby(level=0)]
    return llist
    

def load_event_dfs(ddir, event, iprb=-1):
    """ Load event dataframes (ripples or DS) for 1 or more probes """
    DFS = list(zip(csv2list(ddir, f'ALL_{event.upper()}'), # event dfs
                   csv2list(ddir, 'channel_bp_std')))      # ch bandpass power
    #pdb.set_trace()
    LLIST = []
    for i,(DF_ALL, STD) in enumerate(DFS):
        #nch = load_recording_info(ddir).probe_nch[i]
        channels = np.arange(len(STD))
        # create 'ch' column from index, merge with bandpass power, add event counts
        DF_ALL.insert(0, 'ch', np.array(DF_ALL.index.values))#, dtype='float64'))
        DF_ALL[STD.columns] = STD
        DF_ALL['n'] = DF_ALL.index.value_counts(sort=False)
        # average values by channel (NaNs for channels with no detected events)
        DF_MEAN = DF_ALL.groupby('ch').agg('mean')#
        DF_MEAN = replace_missing_channels(DF_MEAN, channels).astype({'n':int})
        DF_MEAN[STD.columns] = np.array(STD)
        
        DF_MEAN.insert(0, 'ch', DF_MEAN.index.values)
        LLIST.append([DF_ALL, DF_MEAN])
        
    if 0 <= iprb < len(LLIST):
        return LLIST[i]
    else:
        return LLIST

##################################################
########        PROBE CONFIGURATION       ########
##################################################

def read_probe_file(fpath):
    """ Load probe configuration from .json, .mat, or .prb, file """
    if not os.path.exists(fpath):
        return
    # load data according to file extension
    ext = os.path.splitext(fpath)[-1]
    try:
        if ext == '.json':
            probe = prif.io.read_probeinterface(fpath).probes[0]
        elif ext == '.prb':
            probe = prif.io.read_prb(fpath).probes[0]
        elif ext == '.mat':
            probe = mat2probe(fpath)
        # keep probe name consistent with the file name
        probe.name = os.path.splitext(os.path.basename(fpath))[0].replace('_config','')
    except:
        probe = None
    return probe

        
def mat2probe(fpath):
    """ Load probe config from .mat file """
    file = scipy.io.loadmat(fpath, squeeze_me=True)
    xy_arr = np.array([file['xcoords'], 
                       file['ycoords']]).T
    probe = prif.Probe(ndim=int(file['ndim']), 
                            name=str(file['name']))
    probe.set_contacts(xy_arr, 
                       shank_ids   = np.array(file['shankInd']), 
                       contact_ids = np.array(file['contact_ids']))
    probe.set_device_channel_indices(np.array(file['chanMap0ind']))
    return probe
        
        
def write_probe_file(probe, fpath):
    """ Write probe configuration to .json, .mat, .prb, or .csv file """
    ext = os.path.splitext(fpath)[-1]
    
    if ext == '.json':   # best for probeinterface
        prif.io.write_probeinterface(fpath, probe)
        
    elif ext == '.prb':  # loses a bunch of data, but required by some systems
        probegroup = prif.ProbeGroup()
        probegroup.add_probe(probe)
        prif.io.write_prb(fpath, probegroup)
        
    elif ext == '.mat':  # preserves data, not automatically handled by probeinterface
        _ = probe2mat(probe, fpath)
        
    elif ext == '.csv':  # straightforward, easy to view (TBD)
        probe.to_dataframe(complete=True)
        return False
    return True


def probe2mat(probe, fpath):
    """ Save probe config as .mat file"""
    chanMap = probe.device_channel_indices
    probe_dict = {'chanMap'     : np.array(chanMap + 1), 
                  'chanMap0ind' : np.array(chanMap),
                  'connected'   : np.ones_like(chanMap, dtype='int'),
                  'name'        : str(probe.name),
                  'shankInd'    : np.array(probe.shank_ids, dtype='int'),
                  'xcoords'     : np.array(probe.contact_positions[:,0]),
                  'ycoords'     : np.array(probe.contact_positions[:,1]),
                  # specific to probeinterface module
                  'ndim' : int(probe.ndim),
                  'contact_ids' : np.array(probe.contact_ids, dtype='int')}
    probe_dict['connected'][np.where(chanMap==-1)[0]] = 0
    # save file
    so.savemat(fpath, probe_dict)
    return True


def make_probe_group(probe, n=1):
    """ For multi-probe recordings, create group of $n probes to map channels """
    nch = probe.get_contact_count()
    PG = prif.ProbeGroup()
    for i in range(n):
        prb = probe.copy()
        cids = np.array(probe.contact_ids, dtype='int') + i*nch
        dids = np.array(probe.device_channel_indices) + i*nch
        prb.set_contact_ids(cids)
        prb.set_device_channel_indices(dids)
        PG.add_probe(prb)
    return PG


##################################################
########         DATA MANIPULATION        ########
##################################################


def getwaves(LFP, iev, iwin):
    """ Get LFP waveforms surrounding the given event indices $iev """
    arr = np.full((len(iev), iwin*2), np.nan)
    for i,idx in enumerate(iev):
        try:
            arr[i,:] = pad_lfp(LFP, idx, iwin)
        except:
            pdb.set_trace()
    return arr


def getavg(LFP, iev, iwin):
    """ Get event-averaged LFP waveform """
    return np.nanmean(getwaves(LFP, iev, iwin), axis=0)


def pad_lfp(LFP, idx, iwin, pad_val=np.nan):
    """ Add padding to data windows that extend past the recording boundaries """
    if idx >= iwin and idx < len(LFP)-iwin:
        return LFP[idx-iwin : idx+iwin]
    elif idx < iwin:
        pad = np.full(iwin*2 - (idx+iwin), pad_val)
        return np.concatenate([pad, LFP[0 : idx+iwin]])
    else:
        #pad = np.full(len(LFP)-idx, pad_val)
        pad = np.full(iwin*2 - (iwin+len(LFP)-idx), pad_val)
        return np.concatenate([LFP[idx-iwin :], pad])


def replace_missing_channels(DF, channels):
    """ Replace any missing channels in event dataframe with rows of NaNs"""
    if len(DF) == len(channels):
        return DF
    # fill in rows for any channels with no detected events
    missing_ch = np.setdiff1d(channels, DF.index.values)
    missing_df = pd.DataFrame(0.0, index=missing_ch, columns=DF.columns)
    DF = pd.concat([DF, missing_df], axis=0, ignore_index=False).sort_index()
    # set missing values to NaN (except for event counts, which are zero)
    DF.loc[missing_ch, [c for c in DF.columns if c!='n']] = np.nan
    return DF


def encoder2pos(ddir='', A_signal=None, B_signal=None, circumference=44.8, ppr=256):
    """ Get wheel position from rotary encoder A and B signals """
    if A_signal is None or B_signal is None:
        # load A and B signals
        A_path, B_path = Path(ddir, 'A_signal.npy'), Path(ddir, 'B_signal.npy')
        if not (A_path.exists() and B_path.exists()):
            raise Exception('ERROR: Encoder A/B signals not found')
        A_signal, B_signal = np.load(A_path), np.load(B_path)
    # binarize signals
    chA = (A_signal > 0.5) * 1
    chB = (B_signal > 0.5) * 1
    chA_plus_minus = (chA * 2) - 1
    chB_diff = np.concatenate(([0], np.diff(chB)))
    chA_prod = chA_plus_minus * chB_diff
    chB_plus_minus = (chB * 2) - 1
    chA_diff = np.concatenate(([0], np.diff(chA)))
    chB_prod = -chB_plus_minus * chA_diff
    position = np.cumsum(chA_prod + chB_prod)
    # circumfence in cm divided by no. pulses per revolution
    position_cm = position * (circumference/ppr)
    return position_cm


def pos2speed(pos, sf=500):
    """ Translate wheel position to speed with Gaussian smoothing factor $sf """
    pos_dif = np.concatenate((np.array([0]), np.diff(pos)))
    speed_smth = scipy.ndimage.gaussian_filter1d(pos_dif, sf)
    return speed_smth


##################################################
########          EVENT DETECTION         ########
##################################################


def get_inst_freq(x, lfp_fs, swr_freq=[120,180]):
    """ Calculate LFP instantaneous frequency """
    angle  = np.angle(x)       # radian phase (-π to π) of each LFP timepoint
    iphase = np.unwrap(angle)  # phase + 2π*k, where k=cycle number (0-K total cycles)
    difs   = np.diff(iphase)/(2.0*np.pi)  # distance (% of 2π cycle) between consecutive points
    ifreq  = np.clip(difs*lfp_fs, *swr_freq)  # inst. freq (Hz) at each point (bounds=SWR cutoff freqs)
    return ifreq


def get_swr_peaks(LFP, lfp_time, lfp_fs, pprint=True, **kwargs):
    """ Detect peaks in the envelope of sharp-wave ripple activity """
    # load optional keyword args
    swr_freq     = kwargs.get('swr_freq', [120,180])
    swr_min_dur  = kwargs.get('swr_min_dur',  0) / 1000  # ms -> s
    swr_freq_thr = kwargs.get('swr_freq_thr', 0)
    swr_min_dist = kwargs.get('swr_dist_thr', 0)
    swr_fwin     = int(round(kwargs.get('swr_freq_win', 8)/1000 * lfp_fs))
    swr_ampwin   = int(round(kwargs.get('swr_maxamp_win', 40)/1000 * lfp_fs))
    height, distance, swr_min = None,None,None
    
    # get SWR envelope, calculate detection thresholds
    hilb = scipy.signal.hilbert(LFP)     # Hilbert transform of SWR LFP signal
    env = np.abs(hilb).astype('float32') # Hilbert absolute value (amplitude of pos/neg peaks)
    std = np.std(env)                        # standard deviation of SWR envelope
    if 'swr_height_thr' in kwargs:
        height = std * kwargs['swr_height_thr']
    if 'swr_min_thr' in kwargs:
        swr_min = std * kwargs['swr_min_thr']
    if swr_min_dist > 0:
        distance = int(round(lfp_fs * swr_min_dist))
    thresholds = dict(dur=swr_min_dur,         # min. SWR duration (s)
                      inst_freq=swr_freq_thr,  # min. SWR instantaneous freq (Hz)
                      peak_height=height,      # min. SWR peak amplitude
                      edge_height=swr_min,     # min. SWR edge amplitude
                      isi=swr_min_dist)        # min. distance (s) between SWRs
    thresholds = pd.Series(thresholds)
        
    # get instantaneous frequency for each timepoint
    ifreq = get_inst_freq(hilb, lfp_fs, swr_freq)
    env_clip = np.clip(env, swr_min, max(env))
    
    # get indexes of putative SWR envelope peaks
    ippks = scipy.signal.find_peaks(env, height=height, distance=distance)[0]
    ippks = ippks[np.where((ippks > lfp_fs) & (ippks < len(LFP)-lfp_fs))[0]]
    ppk_freqs = np.array([np.mean(ifreq[i-swr_fwin:i+swr_fwin]) for i in ippks])
    
    # get width of each SWR (first point above SWR min to next point below SWR min) 
    durs, _, starts, stops = scipy.signal.peak_widths(env_clip, peaks=ippks, rel_height=1)
    
    # filter for peaks above duration/frequency thresholds
    idur = np.where(durs/lfp_fs > swr_min_dur)[0] # SWRs > min. duration
    ifreq = np.where(ppk_freqs > swr_freq_thr)[0] # SWRs > min. inst. freq
    idx = np.intersect1d(idur, ifreq)
    swr_rate = len(idx) / (lfp_time[-1]-lfp_time[0])
    
    if pprint:
        print((f'{len(idx)} sharp-wave ripples detected; '
               f'SWR rate = {swr_rate:0.3f} Hz ({swr_rate*60:0.1f} events/min)'))
    
    ipks = ippks[idx]
    istarts, istops = [x[idx].astype('int') for x in [starts, stops]]
    
    # get timepoint of largest positive cycle for each SWR
    offsets = [np.argmax(LFP[i-swr_ampwin:i+swr_ampwin]) - swr_ampwin for i in ipks]
    imax = np.array(ipks + np.array(offsets), dtype='int')
    ddict = dict(time      = lfp_time[imax],  # times (s) of largest ripple oscillations
                 amp       = env[ipks],       # SWR envelope peak amplitudes
                 dur       = durs[idx] / (lfp_fs/1000), # SWR durations (ms)
                 freq      = ppk_freqs[idx],     # SWR instantaneous freqs
                 start     = lfp_time[istarts],  # SWR start times
                 stop      = lfp_time[istops],   # SWR end times
                 idx       = imax,    # idx of largest ripple oscillations
                 idx_peak  = ipks,    # idx of max envelope amplitudes
                 idx_start = istarts, # idx of SWR starts
                 idx_stop  = istops)  # idx of SWR stops
    df = pd.DataFrame(ddict)
    return df, thresholds


def get_ds_peaks(LFP, lfp_time, lfp_fs, pprint=True, **kwargs):
    """ Detect peaks of dentate spike waveforms """
    # load optional keyword args
    ds_min_dist = kwargs.get('ds_dist_thr', 0)
    height, distance, wlen, LFPraw = None,None,None,None
    if 'ds_height_thr' in kwargs:
        height = np.std(LFP) * kwargs['ds_height_thr']
    if 'ds_wlen' in kwargs:
        wlen = int(round(lfp_fs * kwargs['ds_wlen']))
    if 'LFPraw' in kwargs:
        LFPraw = kwargs['LFPraw']
    if ds_min_dist > 0:
        distance = int(round(lfp_fs * ds_min_dist))
    min_prominence = kwargs.get('ds_prom_thr', 0)
    thresholds = dict(peak_height=height,  # min. DS peak amplitude
                      isi=ds_min_dist)     # min. distance (s) between DS events
    thresholds = pd.Series(thresholds)
    
    # detect qualifying peaks
    ipks,props = scipy.signal.find_peaks(LFP, height=height, distance=distance, 
                                         prominence=min_prominence)
    ds_prom = props['prominences']
    
    # get peak size/shape
    pws = scipy.signal.peak_widths(LFP, peaks=ipks, rel_height=0.5, wlen=wlen)
    ds_half_width, ds_width_height, starts, stops = pws
    
    # calculate peak half-widths and asymmetry (peak pos. relative to bases)
    istarts, istops = [x.astype('int') for x in [starts, stops]]
    ds_half_width = (ds_half_width/lfp_fs) * 1000  # convert nsamples to ms
    ipre, ipost = ipks-istarts, istops-ipks
    ds_asym = list(map(lambda i0,i1: (i1-i0)/min(i0,i1)*100, ipre, ipost))
    
    # for each peak, get index of max raw LFP value in surrounding 20 samples
    if type(LFPraw) in [list,tuple,np.ndarray] and len(LFPraw) == len(LFP):
        max_ds_loc = [np.argmax(LFPraw[ipk-10:ipk+10]) for ipk in ipks]
        imax   = np.array([ipk-10+max_ds_loc[i] for i,ipk in enumerate(ipks)])
    else:
        imax = np.array(ipks)
    ds_rate = len(ipks) / (lfp_time[-1]-lfp_time[0])
    if pprint:
        print((f'{len(ipks)} dentate spikes detected; '
               f'DS rate = {ds_rate:0.3f} Hz ({ds_rate*60:0.1f} spks/min)'))
    ddict = dict(time         = lfp_time[imax],     # times (s) of DS peak
                 amp          = LFP[ipks],          # DS peak amplitudes
                 half_width   = ds_half_width,      # half-widths (ms) of DS waveforms
                 width_height = ds_width_height,    # DS height at 0.5 peak prominence
                 asym         = ds_asym,  # DS asymmetry (peak pos. relative to bases)
                 prom         = ds_prom,  # DS peak prominence (relative to surround)
                 start        = lfp_time[istarts],  # DS start times
                 stop         = lfp_time[istops],   # DS end times
                 idx          = imax,    # idx of max DS amplitudes
                 idx_peak     = ipks,    # idx of DS scipy peaks
                 idx_start    = istarts, # idx of DS starts
                 idx_stop     = istops)  # idx of DS stops
    df = pd.DataFrame(ddict)
    return df, thresholds


##################################################
########           STATIC PLOTS           ########
##################################################


def plot_channel_events(DF_ALL, DF_MEAN, ax0, ax1, ax2, pal='default'):
    """
    Plot summary statistics for ripples or dentate spikes on each LFP channel
    """
    # plot ripple or DS events
    if 'width_height' in DF_ALL.columns : EVENT = 'DS'
    elif 'freq' in DF_ALL.columns       : EVENT = 'Ripple'
    channels = np.array(DF_MEAN.ch)
    
    # set default palette
    if pal == 'default':
        pal = sns.cubehelix_palette(dark=0.2, light=0.9, rot=0.4, as_cmap=True)
    
    # plot number of events
    _ = ax0.bar(DF_MEAN.ch, DF_MEAN.n, lw=1, color=pyfx.Cmap(DF_MEAN.n, pal))
    ax0.set(xlabel='Channel', ylabel='# events', xmargin=0.05)
    ax0.set_title(f'{EVENT} count', fontdict=dict(fontweight='bold'))
    
    # plot event amplitude
    _ = sns.stripplot(DF_ALL, x='ch', y='amp', hue='amp', palette=pal, legend=False, ax=ax1)
    ax1.set(xlabel='Channel', ylabel='Amplitude', xmargin=0.05)
    ax1.set_title(f'{EVENT} amplitude', fontdict=dict(fontweight='bold'))
    
    if EVENT == 'DS':
        # get standard error for channel half-width heights
        sem = DF_ALL[['ch','width_height']].groupby('ch').agg('sem')
        sem = replace_missing_channels(sem, channels)
        d,yerr = np.array([DF_MEAN.width_height.values, sem.width_height.values])
        clrs = pyfx.Cmap(d, pal)
        # plot summary data
        _ = ax2.vlines(DF_MEAN.ch, d-yerr, d+yerr, lw=3.5, zorder=-1, colors=clrs)
        _ = ax2.scatter(DF_MEAN.ch, d, ec=clrs, fc='white', s=75, lw=3, zorder=0)
        _ = ax2.scatter(DF_MEAN.ch, d, ec=clrs, fc=clrs*[1,1,1,0.2], s=75, lw=3, zorder=1)
        ax2.set(xlabel='Channel', ylabel='prominence / 2', xmargin=0.05)
        ax2.set_title('DS height above surround', fontdict=dict(fontweight='bold'))
        
    elif EVENT == 'Ripple':
        # plot theta and ripple power for all channels
        tmp = d0,d1 = DF_MEAN[['norm_swr','norm_theta']].values.T
        _ = ax2.scatter(DF_MEAN.ch, d0, fc='w', ec='g', s=50, lw=2, label='ripple power')
        _ = ax2.scatter(DF_MEAN.ch, d1, fc='w', ec='b', s=50, lw=2, label='theta power')
        _ = ax2.vlines(DF_MEAN.ch, *np.sort(tmp.T).T, lw=3, zorder=0, colors=pyfx.Cmap(d0-d1, pal))
        _ = ax2.legend(frameon=False)
        ax2.set(xlabel='Channel', ylabel='Norm. power', xmargin=0.05)
        ax2.set_title('Ripple/theta power', fontdict=dict(fontweight='bold'))
        
    ax0.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    
    sns.despine()
    
    return (ax0,ax1,ax2)


##################################################
########          LIVE DATA VIEW          ########
##################################################


def plot_signals(t, ddict, fs, twin=4, step_perc=0.25, **kwargs):
    """
    Show timecourse data on interactive Matplotlib plot with scrollable x-axis
    @Params
    t - time vector (x-axis)
    ddict - dictionary of labeled data vectors
    fs - sampling rate (Hz) of data signals
    twin - time window (s) to show in plot
    step_perc - size of each slider step, as a percentage of $twin
    **kwargs - t_init       : initialize slider at given timepoint (default = minimum t-value)
               hide         : list of data signal(s) to exclude from plot
               plot_nonzero : list of data signal(s) for which to plot nonzero values only
               
               color      : default color for all data signals (if $colordict not given)
               colordict  : dictionary matching data signal(s) with specific colors
               OTHER STYLE PROPERTIES: * lw, lwdict (linewidths)
                                       * ls, lsdict (linestyles)
                                       * mkr, mkrdict (marker shapes)
    @Returns
    fig, ax, slider - Matplotlib figure, data axes, and slider widget
    """
    if isinstance(ddict, np.ndarray):
        ddict = dict(data=np.array(ddict))
    # clean keyword arguments
    t_init     = kwargs.get('t_init', None)   # initial plot timepoint
    hide       = kwargs.get('hide', [])       # hidden data items
    title      = kwargs.get('title', '')
    
    # get dictionary of visible data
    data_dict = {k:v for k,v in ddict.items() if k not in hide}
    
    # set up Matplotlib style properties, set y-axis limits
    props = pd.Series()
    for k,v in zip(['color','lw','ls','mkr'], [None,None,'-',None]):
        dflt_dict = dict.fromkeys(data_dict.keys(), kwargs.get(k,v))
        props[k] = {**dflt_dict, **kwargs.get(k + 'dict', {})}
    ylim = pyfx.Limit(np.concatenate(list(data_dict.values())), pad=0.05)
    
    # get number of samples in plot window / per slider step
    iwin = int(round(twin/2*fs))
    istep = int(round(iwin/4))
    tpad = twin*0.05/2
    # get initial slider value
    if t_init is None : val_init = iwin
    else              : val_init = pyfx.IdxClosest(t_init, t)
    
    # create Matplotlib figure and axes, create slider widget
    fig, (sax0,ax) = plt.subplots(nrows=2, height_ratios=[1,9])
    slider = matplotlib.widgets.Slider(ax=sax0, label='', valmin=iwin, valmax=len(t)-iwin-1, 
                                       valstep=istep, initcolor='none')
    slider.valtext.set_visible(False)
    
    # create data items
    line_dict = {}
    for lbl,data in ddict.items():
        if lbl not in hide:
            #line = ax.plot([0,0], [0,0], color=cdict[lbl], marker=mdict[lbl], label=lbl)[0]
            line = ax.plot([0,0], [0,0], color=props.color[lbl], marker=props.mkr[lbl], 
                           linewidth=props.lw[lbl], linestyle=props.ls[lbl], label=lbl)[0]
            line_dict[lbl] = line
    # set axis limits and legend
    ax.set_ylim(ylim)
    ax.set_title(title)
    leg = ax.legend()
    leg.set_draggable(True)
    sns.despine()
    
    def plot(i):
        # update each data item for current time window
        x = t[i-iwin : i+iwin]
        for lbl,data in data_dict.items():
            line_dict[lbl].set_data(x, data[i-iwin : i+iwin])
        ax.set_xlim([x[0]-tpad, x[-1]+tpad])
        fig.canvas.draw_idle()
        
    # connect slider to plot function, plot initial value
    slider.on_changed(plot)
    slider.set_val(val_init)
    
    return fig, ax, slider
