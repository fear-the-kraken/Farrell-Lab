#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:25:32 2024

@author: amandaschott
"""
import sys
import os
from pathlib import Path
import numpy as np
from PyQt5 import QtWidgets
import probeinterface as prif
import pdb
# set app folder as working directory
app_ddir = Path(__file__).parent
os.chdir(app_ddir)
# import custom modules
import pyfx
import ephys
import data_processing as dp
import gui_items as gi
from channel_selection_gui import ChannelSelectionWindow
from ds_classification_gui import DS_CSDWindow
#%%

def startup():
    if not Path('default_params.txt').exists():
        # create settings text file from hidden params
        _ddict = np.load('.default_params.npy', allow_pickle=True).item()
        ephys.write_param_file(_ddict)
        
    if not Path('base_folders.txt').exists():
        # set default base directories to the app folder
        with open('base_folders.txt', 'w') as fid:
            for k in ['RAW_DATA','PROCESSED_DATA','PROBE_FILES']:
                fid.write(k + ' = ' + str(app_ddir) + '\n')


    
class hippos(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.BASE_FOLDERS = ephys.base_dirs()  # base data directories
        self.PARAMS = ephys.read_param_file()  # default param values
        
        self.gen_layout()
        
        self.show()
        self.center_window()
        
    
    def gen_layout(self):
        """ Set up layout """
        self.setWindowTitle('Hippos')
        self.setContentsMargins(25,25,25,25)
        
        self.centralWidget = QtWidgets.QWidget()
        self.centralLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.centralLayout.setSpacing(20)
        
        mode_btn_ss = ('QPushButton {'
                       'background-color : gainsboro;'
                       'border : 3px outset gray;'
                       'border-radius : 2px;'
                       'color : black;'
                       'padding : 4px;'
                       'font-weight : bold;'
                       '}'
                       
                       'QPushButton:pressed {'
                       'background-color : dimgray;'
                       'border : 3px inset gray;'
                       'color : white;'
                       '}'
                       
                       'QPushButton:checked {'
                       'background-color : darkgray;'
                       'border : 3px inset gray;'
                       'color : black;'
                       '}'
                       
                       'QPushButton:disabled {'
                       'background-color : gainsboro;'
                       'border : 3px outset darkgray;'
                       'color : gray;'
                       '}'
                       
                       'QPushButton:disabled:checked {'
                       'background-color : darkgray;'
                       'border : 3px inset darkgray;'
                       'color : dimgray;'
                       '}'
                       )
        
        # create popup window for processed data
        self.analysis_popup = gi.ProcessedDirectorySelectionPopup(go_to_last=False, parent=self)
        self.analysis_popup.ab.option1_btn.clicked.connect(self.ch_selection_popup)
        self.analysis_popup.ab.option2_btn.clicked.connect(self.classify_ds_popup)
        
        # create main buttons
        self.process_btn = QtWidgets.QPushButton('Process raw data')
        self.process_btn.setStyleSheet(mode_btn_ss)
        self.analyze_btn = QtWidgets.QPushButton('Analyze data')
        self.analyze_btn.setStyleSheet(mode_btn_ss)
        self.view_params_btn = QtWidgets.QPushButton('View parameters')
        self.view_params_btn.setStyleSheet(mode_btn_ss)
        self.base_folder_btn = QtWidgets.QPushButton('Base folders')
        self.base_folder_btn.setStyleSheet(mode_btn_ss)
        # connect to functions
        self.process_btn.clicked.connect(self.raw_data_popup)
        self.analyze_btn.clicked.connect(self.processed_data_popup)
        self.view_params_btn.clicked.connect(self.view_param_popup)
        self.base_folder_btn.clicked.connect(self.base_folder_popup)
        
        self.centralLayout.addWidget(self.process_btn)
        self.centralLayout.addWidget(self.analyze_btn)
        self.centralLayout.addWidget(self.view_params_btn)
        self.centralLayout.addWidget(self.base_folder_btn)
        self.setCentralWidget(self.centralWidget)
        
        
    def raw_data_popup(self, init_raw_ddir='base', init_save_ddir='base'):
        """ Select raw data for processing """
        popup = gi.RawDirectorySelectionPopup(init_raw_ddir, init_save_ddir, parent=self)
        res = popup.exec()
        if not res:
            return
        # get paths to raw data directory, save directory, and probe file
        raw_ddir = popup.raw_ddir         # raw data directory
        save_ddir = popup.processed_ddir  # processed data location
        probe = popup.probe               # channel map for raw signal array
        nch = probe.get_contact_count()
        PARAMS = dict(self.PARAMS)
        
        # load raw files and recording info, make sure probe maps onto raw signals
        (pri_mx, aux_mx), fs = dp.load_raw_data(raw_ddir) # removed info
        num_channels = pri_mx.shape[0]
        dur = pri_mx.shape[1] / fs
        if num_channels % nch > 0:
            msg = f'ERROR: Cannot map {nch} probe contacts to {num_channels} raw data signals'
            res = gi.MsgboxError(msg, parent=self)
            return
        lfp_fs = PARAMS.pop('lfp_fs')
        #info['lfp_fs'] = lfp_fs = PARAMS.pop('lfp_fs')
        #info['lfp_units'] = 'mV'
        
        # create probe group for recording
        probe_group = ephys.make_probe_group(probe, int(num_channels / nch))
        
        # get LFP array for each probe
        lfp_list = dp.extract_data_by_probe(pri_mx, probe_group, fs=fs, lfp_fs=lfp_fs)#, fs=info.fs, lfp_fs=lfp_fs,
                                            #units=info.units, lfp_units=info.lfp_units)
        lfp_time = np.linspace(0, dur, int(lfp_list[0].shape[1]))
        
        # process data by probe, save files in target directory 
        dp.process_all_probes(lfp_list, lfp_time, lfp_fs, PARAMS, save_ddir)
        prif.write_probeinterface(Path(save_ddir, 'probe_group'), probe_group)
        
        # process auxilary channels
        if aux_mx.size > 0:
            aux_dn = dp.extract_data(aux_mx, np.arange(aux_mx.shape[0]), fs=fs, 
                                     lfp_fs=lfp_fs, units='V', lfp_units='V')
            #np.save(Path(save_ddir, 'AUX.npy'), aux_dn)
            aux_dlg = gi.AuxDialog(aux_dn.shape[0], parent=self)
            res = aux_dlg.exec()
            if res:
                for i,fname in enumerate(aux_dlg.aux_files):
                    if fname != '':
                        np.save(Path(save_ddir, fname), aux_dn[i])
            
        # pop-up messagebox appears when processing is complete
        msgbox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, '',
                                       'Data processing complete!', QtWidgets.QMessageBox.Ok, self)
        # set check icon
        chk_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        px_size = msgbox.findChild(QtWidgets.QLabel, 'qt_msgboxex_icon_label').pixmap().size()
        msgbox.setIconPixmap(chk_icon.pixmap(px_size))
        msgbox.exec()
        
        
    def processed_data_popup(self, _, init_ddir=None, go_to_last=True):
        """ Show processed data options """
        self.analysis_popup.show()
        self.analysis_popup.raise_()
        
    
    def ch_selection_popup(self):
        """ Launch event channel selection window """
        ddir = self.analysis_popup.ddir
        iprb = self.analysis_popup.current_probe
        self.ch_selection_dlg = ChannelSelectionWindow(ddir, iprb=iprb, parent=self.analysis_popup)
        self.ch_selection_dlg.show()
        self.ch_selection_dlg.raise_()
        _ = self.ch_selection_dlg.exec()
        # check for updated files, enable/disable analysis options
        self.analysis_popup.ab.ddir_toggled(ddir)
        
    
    def classify_ds_popup(self):
        """ Launch DS analysis window """
        ddir = self.analysis_popup.ddir
        iprb = self.analysis_popup.current_probe
        PARAMS = ephys.load_recording_params(ddir)
        self.classify_ds_dlg = DS_CSDWindow(ddir, iprb, self.PARAMS, parent=self.analysis_popup)
        self.classify_ds_dlg.show()
        self.classify_ds_dlg.raise_()
        _ = self.classify_ds_dlg.exec()
        # check for updated files, enable/disable analysis options
        self.analysis_popup.ab.ddir_toggled(ddir)
        
        
    def view_param_popup(self):
        """ View default parameters """
        params = ephys.read_param_file()
        keys, vals = zip(*params.items())
        vstr = [*map(str,vals)]
        klens = [*map(len, keys)]; kmax=max(klens)
        padk = [*map(lambda k: k + '_'*(kmax-len(k))+':', keys)]
        #rows = [(pdk+vst) for pdk,vst in zip(padk,vstr)]
        html = ['<pre>'+k+v+'</pre>' for k,v in zip(padk,vstr)]
        text = '<h3><tt>DEFAULT PARAMETERS</tt></h3>' + '<hr>' + ''.join(html)
        textwidget = QtWidgets.QTextEdit(text)
        # create popup window for text widget
        dlg = QtWidgets.QDialog(self)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(textwidget)
        qrect = pyfx.ScreenRect(perc_height=0.5, perc_width=0.3, keep_aspect=False)
        dlg.setGeometry(qrect)
        dlg.show()
        dlg.raise_()
    
    
    def base_folder_popup(self):
        """ View or change base data directories """
        dlg = gi.thing()
        dlg.show()
        dlg.raise_()
        
    
    def center_window(self):
        """ Move GUI to center of screen """
        qrect = self.frameGeometry()  # proxy rectangle for window with frame
        screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
        qrect.moveCenter(screen_rect.center())  # move center of qr to center of screen
        self.move(qrect.topLeft())
        
#%%
if __name__ == '__main__':
    args = list(sys.argv)
    if len(args) == 1:
        args.append('')
        
    # TO-DO
    # add notes section
    # add behavior data
    
    #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/NN_JG023_2probes2/'
    
    app = QtWidgets.QApplication(args)
    app.setStyle('Fusion')
    app.setQuitOnLastWindowClosed(True)
    
    w = hippos()
    #w = ChannelSelectionWindow(ddir, 0)
    
    w.show()
    w.raise_()
    sys.exit(app.exec())