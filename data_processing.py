#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:00:24 2024

@author: amandaschott
"""
import os
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import probeinterface as prif
import quantities as pq
import pdb
# custom modules
import pyfx
import ephys


def validate_raw_ddir(ddir):
    """ Check whether directory contains raw Open Ephys or NeuroNexus data """
    if not os.path.exists(ddir):
        return False
    files = os.listdir(ddir)
    a = bool('structure.oebin' in files)
    b = bool(len([f for f in files if f.endswith('.xdat.json')]) > 0)
    return bool(a or b)


def bp_filter_lfps(lfp, lfp_fs, **kwargs):
    """ Bandpass filter LFP signals within fixed frequency bands """
    # set filter cutoffs
    theta      = kwargs.get('theta',      [6,10])
    slow_gamma = kwargs.get('slow_gamma', [25,55])
    fast_gamma = kwargs.get('fast_gamma', [60,100])
    swr_freq   = kwargs.get('swr_freq',   [120,180])
    ds_freq    = kwargs.get('ds_freq',    [5,100])
    
    # collect filtered LFPs in data dictionary
    bp_dict = {'raw' : lfp}
    bp_dict['theta']      = pyfx.butter_bandpass_filter(lfp, *theta,      lfp_fs=lfp_fs, axis=1)
    bp_dict['slow_gamma'] = pyfx.butter_bandpass_filter(lfp, *slow_gamma, lfp_fs=lfp_fs, axis=1)
    bp_dict['fast_gamma'] = pyfx.butter_bandpass_filter(lfp, *fast_gamma, lfp_fs=lfp_fs, axis=1)
    bp_dict['swr']        = pyfx.butter_bandpass_filter(lfp, *swr_freq,   lfp_fs=lfp_fs, axis=1)
    bp_dict['ds']         = pyfx.butter_bandpass_filter(lfp, *ds_freq,    lfp_fs=lfp_fs, axis=1)
    
    return bp_dict


def load_openephys_data(ddir):
    """ Load raw data files from Open Ephys recording software """
    # initialize Open Ephys data objects
    session = ephys.get_openephys_session(ddir)
    OE = ephys.oeNodes(session, ddir)
    node = OE['node']            # node containing the selected recording
    recording = OE['recording']  # experimental recording object
    continuous_list = OE['continuous']  # continuous data from 1 or more processors
    #settings_file = str(Path(node.directory, 'settings.xml'))
    metadata_list = recording.info['continuous']
    nch_list = []
    for i,(continuous,meta) in enumerate(zip(continuous_list, metadata_list)):
        # load sampling rate, timestamps, and number of channels
        fs = meta['sample_rate']
        num_channels = meta['num_channels']
        tstart, tend = pyfx.Edges(continuous.timestamps)
        print('under construction')
        pdb.set_trace()
        # load channel wiring, units, and bit-volt conversion factors
        ch_info = [[(d['channel_name'], d['units'], d['bit_volts']), 
                    d['description']] for d in meta['channels']]
        hstg = [(int(a[1:]),b,c) for [(a,b,c),d] in ch_info if d.startswith('Headstage')]
        dev_idx, units, bit_volts = map(list, zip(*hstg))
        aux = [(int(a[1:]),b,c) for [(a,b,c),d] in ch_info if d.startswith('ADC')]
        aux_idx, aux_units, aux_bit_volts = map(list, zip(*aux))
        
        # load raw signals (uV)
        first,last = pyfx.Edges(continuous.sample_numbers)
        raw_signal_array = np.array([x*bv for x,bv in zip(continuous.get_samples(0, last-first+1).T, 
                                                          bit_volts)])
        nch = raw_signal_array.shape[0]
    
    node_name = str(meta['recorded_processor']) + str(meta['recorded_processor_id'])
    # collect recording info in dictionary
    # info = pd.Series(dict(raw_data_path = ddir,
    #                       recording_system = 'Open Ephys',
    #                       units = units[0],
    #                       ports = np.arange(len(continuous_list)),
    #                       nprobes = len(continuous_list),
    #                       probe_nch = [nch],
                          
    #                       fs = fs,
    #                       nchannels = num_channels,
    #                       nsamples = raw_signal_array.shape[1],
    #                       tstart = tstart,
    #                       tend = tend,
    #                       dur = tend - tstart))
    return raw_signal_array, fs#, info


def load_neuronexus_data(ddir):
    """ Load raw data files from Allego NeuroNexus recording software """
    # get raw file names
    meta_file = [f for f in os.listdir(ddir) if f.endswith('.xdat.json')][0]
    stem = meta_file.replace('.xdat.json', '')
    data_file = os.path.join(ddir, stem + '_data.xdat')
    
    # load metadata
    with open(os.path.join(ddir, meta_file), 'rb') as f:
        metadata = json.load(f)
    fs             = metadata['status']['samp_freq']
    num_channels   = metadata['status']['signals']['pri']
    total_channels = metadata['status']['signals']['total']
    tstart, tend   = metadata['status']['t_range']
    num_samples    = int(round(tend * fs)) - int(round(tstart * fs))
    # get SI units
    udict = {'micro-volts':'uV', 'milli-volts':'mV', 'volts':'V'}
    units = metadata['sapiens_base']['sigUnits']['sig_units_pri']
    units = udict.get(units, units)
    
    # organize electrode channels by port
    ports,ddicts = map(list, zip(*metadata['sapiens_base']['sensors_by_port'].items()))
    nprobes = len(ports)
    probe_nch = [d['num_channels'] for d in ddicts]
    
    # separate primary vs aux channels
    ch_names = metadata['sapiens_base']['biointerface_map']['chan_name']
    ipri = np.array([i for i,n in enumerate(ch_names) if n.startswith('pri')])
    iaux = np.setdiff1d(np.arange(total_channels), ipri)
    #iaux = np.array([i for i,n in enumerate(ch_names) if n.startswith('aux')])
    
    # load raw probe data
    with open(data_file, 'rb') as fid:
        fid.seek(0, os.SEEK_SET)
        raw_signals = np.fromfile(fid, dtype=np.float32, count=num_samples*total_channels)
    raw_signal_array = np.reshape(raw_signals, (num_samples, total_channels)).T
    pri_mx = raw_signal_array[ipri]
    aux_mx = raw_signal_array[iaux]
    
    # collect recording info in dictionary
    # info = pd.Series(dict(raw_data_path = ddir,
    #                       recording_system = 'Allego NeuroNexus',
    #                       units = units,
    #                       ports = ports,
    #                       nprobes = nprobes,
    #                       probe_nch = probe_nch,
                          
    #                       fs = fs,
    #                       nchannels = num_channels,
    #                       nsamples = raw_signal_array.shape[1],
    #                       tstart = tstart,
    #                       tend = tend,
    #                       dur = tend - tstart))
    return (pri_mx, aux_mx), fs#, info


def load_raw_data(ddir, pprint=True):
    """ Load raw data files from Open Ephys or NeuroNexus software """
    try:
        files = os.listdir(ddir)
    except:
        raise Exception(f'Directory "{ddir}" does not exist')
    xdat_files = [f for f in files if f.endswith('.xdat.json')]
    # load Open Ephys data
    if 'structure.oebin' in files:
        if pprint: print('Loading Open Ephys raw data ...')
        (pri_array, aux_array), fs = load_openephys_data(ddir) # removed info, added fs
    # load NeuroNexus data
    elif len(xdat_files) > 0:
        if pprint: print('Loading NeuroNexus raw data ...')
        (pri_array, aux_array), fs = load_neuronexus_data(ddir)
    # no valid raw data found
    else:
        raise Exception(f'No raw Open Ephys (.oebin) or NeuroNexus (.xdat.json) files found in directory "{ddir}"')
    return (pri_array, aux_array), fs
    #return signal_array, fs#, info


# def yargle(ddir, probe):
#     raw_signal_array, info = load_raw_data(ddir)
    
#     nch = probe.get_contact_count()
#     num_channels = raw_signal_array.shape[0]
#     assert (num_channels % nch > 0)
    
#     probe_group = ephys.make_probe_group(probe, int(num_channels / nch))
    

def get_idx_by_probe(probe):
    """ Clean $probe input, return list of channel maps """
    if probe.__class__ == prif.Probe:
        idx_by_probe = [probe.device_channel_indices]
    elif probe.__class__ == prif.ProbeGroup:
        idx_by_probe = [prb.device_channel_indices for prb in probe.probes]
    elif type(probe) in [list, np.ndarray]:
        if type(probe) == list:
            probe = np.array(probe)
        if type(probe) == np.ndarray:
            if probe.ndim == 1:
                idx_by_probe = [probe]
            elif probe.ndim == 2:
                idx_by_probe = [x for x in probe]
    return idx_by_probe
        

def extract_data(raw_signal_array, idx, fs=30000, lfp_fs=1000, units='uV', lfp_units='mV'):
    """ Extract, scale, and downsample each raw signal in depth order down the probe """
    ds_factor = int(fs / lfp_fs)  # calculate downsampling factor
    cf = pq.Quantity(1, units).rescale(lfp_units).magnitude  # mV conversion factor
    lfp = np.array([pyfx.Downsample(raw_signal_array[i], ds_factor)*cf for i in idx])
    return lfp
    
def extract_data_by_probe(raw_signal_array, chMap, fs=30000, lfp_fs=1000, units='uV', lfp_units='mV'):
    """ Get LFP array for each probe represented in $chMap """
    idx_by_probe = get_idx_by_probe(chMap)
    lfp_list = [extract_data(raw_signal_array, idx, fs=fs, lfp_fs=lfp_fs, 
                             units=units, lfp_units=lfp_units) for idx in idx_by_probe]
    return lfp_list

def process_probe_data(_lfp, lfp_time, lfp_fs, PARAMS, pprint=True):
    """ Filter LFPs, run ripple and DS detection on each channel """
    
    # bandpass filter LFPs within different frequency bands
    if pprint: print('Bandpass filtering signals ...')    
    bp_dict = bp_filter_lfps(_lfp, lfp_fs, **PARAMS)
    # get standard deviation (raw and normalized) for each filtered signal
    std_dict = {k : np.std(v, axis=1) for k,v in bp_dict.items()}
    std_dict.update({f'norm_{k}' : pyfx.Normalize(v) for k,v in std_dict.items()})
    STD = pd.DataFrame(std_dict)
    
    # run ripple detection on all channels
    SWR_DF = pd.DataFrame()
    SWR_THRES = {}
    if pprint: print('Detecting ripples on each channel ...')
    for ch in range(_lfp.shape[0]):
        # sharp-wave ripples
        swr_df, swr_thres = ephys.get_swr_peaks(bp_dict['swr'][ch], lfp_time, lfp_fs, 
                                                pprint=False, **PARAMS)
        swr_df.set_index(np.repeat(ch, len(swr_df)), inplace=True)
        SWR_DF = pd.concat([SWR_DF, swr_df], ignore_index=False)
        SWR_THRES[ch] = swr_thres
    
    # run DS detection on all channels
    DS_DF = pd.DataFrame()
    DS_THRES = {}
    if pprint: print('Detecting dentate spikes on each channel ...')
    for ch in range(_lfp.shape[0]):
        # dentate spikes
        ds_df, ds_thres = ephys.get_ds_peaks(bp_dict['ds'][ch], lfp_time, lfp_fs, 
                                             pprint=False, **PARAMS)
        ds_df.set_index(np.repeat(ch, len(ds_df)), inplace=True)
        DS_DF = pd.concat([DS_DF, ds_df], ignore_index=False)
        DS_THRES[ch] = ds_thres
    THRESHOLDS = dict(SWR=SWR_THRES, DS=DS_THRES)
    
    return bp_dict, STD, SWR_DF, DS_DF, THRESHOLDS


def process_all_probes(lfp_list, lfp_time, lfp_fs, PARAMS, save_ddir, pprint=True):
    """
    Process LFPs for each probe in dataset, save to new data folder
    """
    if type(lfp_list) == np.ndarray:
        lfp_list = [lfp_list]
    bp_dicts = {'raw':[], 'theta':[], 'slow_gamma':[], 'fast_gamma':[], 'swr':[], 'ds':[]}
    std_dfs, swr_dfs, ds_dfs, thresholds = [], [], [], []
    
    for i,_lfp in enumerate(lfp_list):
        if pprint: print(f'\n#####   PROBE {i+1} / {len(lfp_list)}   #####\n')
        bp_dict, STD, SWR_DF, DS_DF, THRESHOLDS = process_probe_data(_lfp, lfp_time, lfp_fs, 
                                                                     PARAMS, pprint=pprint)
        for k,l in bp_dicts.items(): l.append(bp_dict[k])
        std_dfs.append(STD)
        swr_dfs.append(SWR_DF)
        ds_dfs.append(DS_DF)
        thresholds.append(THRESHOLDS)
    ALL_STD = pd.concat(std_dfs, keys=range(len(std_dfs)), ignore_index=False)
    ALL_SWR = pd.concat(swr_dfs, keys=range(len(swr_dfs)), ignore_index=False)
    ALL_DS = pd.concat(ds_dfs, keys=range(len(ds_dfs)), ignore_index=False)
    
    # save downsampled data
    if pprint: print('Saving files ...')
    if not os.path.isdir(save_ddir):
        os.mkdir(save_ddir)
    np.save(Path(save_ddir, 'lfp_time.npy'), lfp_time)
    np.save(Path(save_ddir, 'lfp_fs.npy'), lfp_fs)
    np.savez(Path(save_ddir, 'lfp_bp.npz'), **bp_dicts)
    
    # save bandpass-filtered power in each channel (index)
    ALL_STD.to_csv(Path(save_ddir, 'channel_bp_std'), index_label=False)
    
    # save event quantifications and thresholds
    ALL_SWR.to_csv(Path(save_ddir, 'ALL_SWR'), index_label=False)
    ALL_DS.to_csv(Path(save_ddir, 'ALL_DS'), index_label=False)
    np.save(Path(save_ddir, 'THRESHOLDS.npy'), thresholds)
    
    # save params and info file
    with open(Path(save_ddir, 'params.pkl'), 'wb') as f:
        pickle.dump(PARAMS, f)
        
    # with open(Path(save_ddir, 'info.pkl'), 'wb') as f:
    #     pickle.dump(INFO, f)
    
    if pprint: print('Done!' + os.linesep)
    
    
#%%
ddir = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/raw_data/'
        'JG007_2_2024-07-09_15-40-43_openephys/Record Node 103/experiment1/recording1')

    
# def tmpl_info(lfp_list, lfp_fs, **kwargs):
#     ddict = dict(raw_data_path='',
#                  recording_system='unknown',
#                  units='uV',
#                  ports=[str(i) for i in range(len(lfp_list))],
#                  nprobes=len(lfp_list),
#                  probe_nch=[arr.shape[0] for arr in lfp_list],
#                  fs=lfp_fs)
#     ddict['nchannels']=np.sum(ddict['probe_nch'])
#     ddict['nsamples']=lfp_list[0].shape[1]
#     ddict['tstart']=0
#     ddict['tend']=ddict['nsamples']/lfp_fs
#     ddict['dur']=ddict['tend'] - ddict['tstart']
#     ddict.update(**kwargs)
#     return ddict