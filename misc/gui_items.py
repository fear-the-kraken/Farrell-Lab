#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:45:34 2024

@author: amandaschott
"""
import sys
import os
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
import probeinterface as prif
from probeinterface.plotting import plot_probe
import pdb
# custom modules
import pyfx
import ephys

btn_ss = ('QPushButton {'
          'background-color : rgba%s;'  # light
          'border : 4px outset rgb(128,128,128);'
          'border-radius : 11px;'
          'min-width : 15px;'
          'max-width : 15px;'
          'min-height : 15px;'
          'max-height : 15px;'
          '}'
          
          'QPushButton:disabled {'
          'background-color : rgb(220,220,220);'#'rgba%s;'
          #'border : 4px outset rgb(128,128,128);'
          '}'
          
          'QPushButton:pressed {'
          'background-color : rgba%s;'  # dark
          '}'
          
          # 'QPushButton:checked {'
          # 'background-color : rgba%s;'  # dark
          # #'outline : 2px solid red;'
          # 'border : 3px solid red;'
          # 'border-radius : 2px;'
          # '}'
          )

def iter2str(v):
    if not hasattr(v, '__iter__'): 
        return str(v)
    return ', '.join(np.array(v,dtype='str').tolist())

def val2str(v):
    if np.ndim(v) < 2:
        txt = iter2str(v)
    elif np.ndim(v) == 2:
        txt = ', '.join([f'({x})' for x in map(iter2str, v)])
    return txt

def info2text2(info):
    keys, vals = zip(*info.items())
    vstr = [*map(str,vals)]
    klens = [*map(len, keys)]
    kmax=max(klens)
    padk = [*map(lambda k: k + '_'*(kmax-len(k)), keys)]
    rows = ['<tr><td align="center"><pre>'+pdk+' : '+'</pre></td><td><pre>'+v+'</pre></td></tr>' for pdk,v in zip(padk,vstr)]
    text = '<body><h2><pre>Recording Info</pre></h2><hr><table>' + ''.join(rows) + '</table></body>'
    return text
    #text = ''.join(rows)
    
def info2text(info, rich=True):
    sep = '<br>' if rich else os.linesep
    fmt = ('<tr>'
               '<td align="center"; style="background-color:#f2f2f2; border:2px solid #e6e6e6; white-space:nowrap;"><tt>%s</tt></td>'
               '<td align="center"><font size="4"></font></td>'
               '<td style="background-color:#f2f2f2; border:2px solid #e6e6e6;"><tt>%s</tt></td>'
           '</tr>')
    
    div_row = '<tr><td colspan="3"><hr></td></tr>'
    
    gen_rows = [fmt % (k,info[k]) for k in ['raw_data_path','recording_system','units']]
    gen_rows[1] = gen_rows[1].replace('recording_system','system')
    # probe rows
    keys = ['ports', 'nprobes', 'probe_nch']
    vals = [val2str(info[k]) for k in keys]
    probe_rows = [fmt % (a,b) for a,b in zip(keys,vals)]
    # recording rows
    rec_rows = [fmt % ('fs',        '%s Hz'        % info['fs']),  #    f"{info['nchannels']} Hz"  f'%s Hz'
                fmt % ('nchannels', '%s primary el.' % info['nchannels']),
                fmt % ('nsamples',  '%.1E bins' % info['nsamples']),
                fmt % ('tstart',    '%.2f s' % info['tstart']),
                fmt % ('tend',      '%.2f s' % info['tend']),
                fmt % ('duration',  '%.2f s' % info['dur'])]
        
    # info popup title '<hr width="70%">'
    #'<p style="line-height:30%; vertical-align:middle;">'
    #'<hr align="center"; width="70%"></p>'
    #<div> <p align="center"; style="background-color:green;">' + ; width="60%"
    header = (#'<hr height="5px"; style="background-color:yellow; color:blue; border:5px solid orange;">'
              #'<hr width="60%">'
              # '<p style="background-color:red; border:5px solid black;">'
              # #'nekkid'
              # #'***'
              # '<td style="border:2px solid green;"><hr></td>'
              # '</p>'
              '<h2 align="center"; style="background-color:#f2f2f2; border:2px solid red; padding:100px;"><tt>Recording Info</tt></h2>'
              #'<p style="background-color:none;">'
              #'<hr style="border:5px solid black">')
              #'---')
              )
    
    info_text = ('<body style="background-color:#e6e6e6;">' + header + 
                 '<br style="line-height:30%">' + 
                 '<table border-collapse="collapse"; cellspacing="0"; cellpadding="3">' +
                ''.join(gen_rows)   + str(div_row)     + #'</table>')#str(div_row) +
                ''.join(probe_rows) + str(div_row)     + #'</table>')
                ''.join(rec_rows)   + '</table>' + '</body')
    
    return info_text


def unique_fname(ddir, base_name):
    existing_files = os.listdir(ddir)
    fname = str(base_name)
    i = 1
    while fname in existing_files:
        new_name = fname + f' ({i})'
        if new_name not in existing_files:
            fname = str(new_name)
            break
        i += 1
    return fname


def validate_raw_ddir(ddir):
    if not os.path.isdir(ddir):
        return False
    files = os.listdir(ddir)
    a = bool('structure.oebin' in files)
    b = bool(len([f for f in files if f.endswith('.xdat.json')]) > 0)
    return bool(a or b)


def validate_processed_ddir(ddir):
    if not os.path.isdir(ddir):
        return False
    files = os.listdir(ddir)
    flist = ['lfp_bp.npz', 'lfp_time.npy', 'lfp_fs.npy']
    x = all([bool(f in files) for f in flist])
    return x


class CSlider(matplotlib.widgets.Slider):
    """ Slider with enable/disable function """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handle._markersize = 20
        self._handle._markeredgewidth = 2
        self.nsteps = 500
    
    def key_step(self, event):
        if event.key == 'right':
            self.set_val(self.val + self.nsteps)
        elif event.key == 'left':
            self.set_val(self.val - self.nsteps)
        
        
    def enable(self, x):
        if x:
            self.track.set_facecolor('lightgray')
            self.track.set_alpha(1)
            self.poly.set_facecolor('indigo')
            self.poly.set_alpha(1)
            self._handle.set_markeredgecolor('darkgray')
            self._handle.set_markerfacecolor('white')
            self._handle.set_alpha(1)
            self.valtext.set_alpha(1)
            self.label.set_alpha(1)
            
        else:
            self.track.set_facecolor('lightgray')
            self.track.set_alpha(0.3)
            self.poly.set_facecolor('lightgray')
            self.poly.set_alpha(0.5)
            self._handle.set_markeredgecolor('darkgray')
            self._handle.set_markerfacecolor('gainsboro')
            self._handle.set_alpha(0.5)
            self.valtext.set_alpha(0.2)
            self.label.set_alpha(0.2)


class EventArrows(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QPushButton {font-weight:bold; padding:2px;}')
        self.left = QtWidgets.QPushButton('\u2190') # unicode ◄ and ►
        self.right = QtWidgets.QPushButton('\u2192')
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.setSpacing(1)
        hbox.setContentsMargins(0,0,0,0)
        hbox.addWidget(self.left)
        hbox.addWidget(self.right)
        self.bgrp = QtWidgets.QButtonGroup(self)
        self.bgrp.addButton(self.left, 0)
        self.bgrp.addButton(self.right, 1)
    
    
class ShowHideBtn(QtWidgets.QPushButton):
    def __init__(self, text_shown='\u00BB', text_hidden='\u00AB', init_show=False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.TEXTS = [text_hidden, text_shown]
        #self.SHOWN_TEXT = text_shown
        #self.HIDDEN_TEXT = text_hidden
        # set checked/visible or unchecked/hidden
        self.setChecked(init_show)
        self.setText(self.TEXTS[int(init_show)])
        
        
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                        QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(policy)
        self.toggled.connect(self.update_state)

        self.setStyleSheet('QPushButton {'
                            'background-color : gainsboro;'
                            'border : 3px outset gray;'
                            'border-radius : 2px;'
                            'color : rgb(50,50,50);'
                            'font-size : 30pt;'
                            'font-weight : normal;'
                            'max-width : 30px;'
                            'padding : 4px;'
                            '}'
                            
                            'QPushButton:pressed {'
                            'background-color : dimgray;'
                            'border : 3px inset gray;'
                            'color : whitesmoke;'
                            '}')
        
    def update_state(self, show):
        self.setText(self.TEXTS[int(show)])


class ReverseSpinBox(QtWidgets.QSpinBox):
    """ Spin box with reversed increments (down=+1) to match LFP channels """
    def stepEnabled(self):
        if self.wrapping() or self.isReadOnly():
            return super().stepEnabled()
        ret = QtWidgets.QAbstractSpinBox.StepNone
        if self.value() > self.minimum():
            ret |= QtWidgets.QAbstractSpinBox.StepUpEnabled
        if self.value() < self.maximum():
            ret |= QtWidgets.QAbstractSpinBox.StepDownEnabled
        return ret

    def stepBy(self, steps):
        return super().stepBy(-steps)
        
    
class MsgboxSave(QtWidgets.QMessageBox):
    def __init__(self, msg='Save successful! Exit window?', parent=None):
        super().__init__(parent)
        # pop-up messagebox appears when save is complete
        self.setIcon(QtWidgets.QMessageBox.Information)
        self.setText(msg)
        self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        # set check icon
        chk_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        px_size = self.findChild(QtWidgets.QLabel, 'qt_msgboxex_icon_label').pixmap().size()
        self.setIconPixmap(chk_icon.pixmap(px_size))


class MsgboxError(QtWidgets.QMessageBox):
    def __init__(self, msg='ERROR', parent=None):
        super().__init__(parent)
        self.setIcon(QtWidgets.QMessageBox.Critical)
        self.setText(msg)
        self.setStandardButtons(QtWidgets.QMessageBox.Close)
        
        
class AuxDialog(QtWidgets.QDialog):
    def __init__(self, n, parent=None):
        super().__init__(parent)
        
        #flags = self.windowFlags() | QtCore.Qt.FramelessWindowHint
        #self.setWindowFlags(flags)
        self.setWindowTitle('AUX channels')
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(10)
        qlabel = QtWidgets.QLabel('Set AUX file names (leave blank to ignore)')
        # load list of previously saved aux files
        self.auxf = list(np.load('.aux_files.npy'))
        completer = QtWidgets.QCompleter(self.auxf, self)
        grid = QtWidgets.QGridLayout()
        self.qedits = []
        # create QLineEdit for each AUX channel
        for i in range(n):
            lbl = QtWidgets.QLabel(f'AUX {i}')
            qedit = QtWidgets.QLineEdit()
            qedit.setCompleter(completer)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(qedit, i, 1)
            self.qedits.append(qedit)
        # action button
        bbox = QtWidgets.QHBoxLayout()
        self.continue_btn = QtWidgets.QPushButton('Continue')
        self.continue_btn.clicked.connect(self.accept)
        self.clear_btn = QtWidgets.QPushButton('Clear all')
        self.clear_btn.clicked.connect(self.clear_files)
        bbox.addWidget(self.continue_btn)
        bbox.addWidget(self.clear_btn)
        # set up layout
        self.layout.addWidget(qlabel)
        line = pyfx.DividerLine()
        self.layout.addWidget(line)
        self.layout.addLayout(grid)
        self.layout.addLayout(bbox)
    
    def update_files(self):
        for i,qedit in enumerate(self.qedits):
            txt = qedit.text()
            if txt != '':
                if not txt.endswith('.npy'):
                    txt += '.npy'
            self.aux_files[i] = txt
    
    def clear_files(self):
        for qedit in self.qedits:
            qedit.setText('')
    
    def accept(self):
        self.aux_files = []
        for qedit in self.qedits:
            txt = qedit.text()
            if txt.endswith('.npy'):
                txt = txt[0:-4]
            if txt not in self.auxf:
                self.auxf.append(txt)
            fname = txt + ('' if txt == '' else '.npy')
            self.aux_files.append(fname)
        np.save('.aux_files.npy', self.auxf)
        super().accept()
        
            

class FileDialog(QtWidgets.QFileDialog):
    
    def __init__(self, init_ddir='', load_or_save='load', overwrite_mode=0, parent=None):
        super().__init__(parent)
        self.load_or_save = load_or_save
        self.overwrite_mode = overwrite_mode  # 0=add to folder, 1=overwrite files
        options = self.Options()
        options |= self.DontUseNativeDialog
        
        self.setViewMode(self.List)
        self.setAcceptMode(self.AcceptOpen)  # open file
        self.setFileMode(self.Directory)     # allow directories only
        
        try: self.setDirectory(init_ddir)
        except: print('init_ddir argument in FileDialog is invalid')
        self.setOptions(options)
        self.connect_signals()
    
    
    def connect_signals(self):
        self.lineEdit = self.findChild(QtWidgets.QLineEdit)
        self.stackedWidget = self.findChild(QtWidgets.QStackedWidget)
        self.view = self.stackedWidget.findChild(QtWidgets.QListView)
        self.view.selectionModel().selectionChanged.connect(self.updateText)
    
    
    def updateText(self, selected, deselected):
        if selected.indexes() == []:
            return
        # update contents of the line edit widget with the selected files
        txt = self.view.selectionModel().currentIndex().data()
        self.lineEdit.setText(txt)
    
    
    def overwrite_msgbox(self, ddir):
        txt = (f'The directory <code>{ddir.split(os.sep)[-1]}</code> contains '
               f'<code>{len(os.listdir(ddir))}</code> items.<br><br>Overwrite existing files?')
        msg = '<center>{}</center>'.format(txt)
        res = QtWidgets.QMessageBox.warning(self, 'Overwrite Warning', msg, 
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if res == QtWidgets.QMessageBox.Yes:
            return True
        return False
        
    
    def accept(self):
        ddir = self.directory().path()
        if self.load_or_save == 'save' and len(os.listdir(ddir)) > 0:
            # show overwrite warning for existing directory files
            res = self.overwrite_msgbox(ddir)
            if not res: return
        QtWidgets.QDialog.accept(self)
    
    
    
class QEdit_HBox(QtWidgets.QHBoxLayout):
    def __init__(self, simple=False, colors=['gray','darkgreen'], parent=None):
        super().__init__(parent)
        self.simple_mode = simple
        self.c0, self.c1 = colors
        
        self.setContentsMargins(0,0,0,0)
        self.setSpacing(0)
        
        # ellipsis (...)
        self.ellipsis = QtWidgets.QLineEdit()
        self.ellipsis.setAlignment(QtCore.Qt.AlignCenter)
        self.ellipsis.setTextMargins(0,4,0,4)
        self.ellipsis.setReadOnly(True)
        self.ellipsis.setText('...')
        self.ellipsis.setMaximumWidth(20)
        # base path to directory (resizable)
        self.path = QtWidgets.QLineEdit()
        self.path.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.path.setTextMargins(0,4,0,4)
        self.path.setReadOnly(True)
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                        QtWidgets.QSizePolicy.Fixed)
        self.path.setSizePolicy(policy)
        # directory name (gets size priority according to text length)
        self.folder = QtWidgets.QLineEdit()
        self.folder.setTextMargins(0,4,0,4)
        self.folder.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.folder.setReadOnly(True)
        policy2 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)
        self.folder.setSizePolicy(policy2)
        # set attributes for all QLineEdit items
        self.qedits = [self.ellipsis, self.path, self.folder]
        ss = ('QLineEdit {'
              f'border : 2px groove {self.c0};'
              'border-left : %s;'
              'border-right : %s;'
              'font-weight : %s;'
              'padding : 0px;}')
        self.ellipsis.setStyleSheet(ss % (f'2px groove {self.c0}', 'none', 'normal'))
        self.folder.setStyleSheet(ss % ('none', f'2px groove {self.c0}', 'bold'))
        if self.simple_mode:
            self.path.setStyleSheet(ss % (f'2px groove {self.c0}', f'2px groove {self.c0}', 'normal'))
        else:
            self.path.setStyleSheet(ss % ('none', 'none', 'normal'))
        
        self.addWidget(self.path)
        if not self.simple_mode:
            self.insertWidget(0, self.ellipsis)
            self.addWidget(self.folder)
    
    
    def update_qedit(self, ddir, x=False):
        # update QLineEdit text
        if self.simple_mode:
            self.path.setText(ddir)
            return
        
        dirs = ddir.split(os.sep)
        folder_txt = dirs.pop(-1)
        path_txt = os.sep.join(dirs) + os.sep
        self.qedits[1].setText(path_txt)
        self.qedits[2].setText(folder_txt)
        fm = self.qedits[2].fontMetrics()
        width = fm.horizontalAdvance(folder_txt) + int(fm.maxWidth()/2)
        self.qedits[2].setFixedWidth(width)
        
        c0, c1 = [self.c0, self.c1] if x else [self.c1, self.c0]
        for qedit in self.qedits:
            qedit.setStyleSheet(qedit.styleSheet().replace(c0, c1))
            

class StatusIcon(QtWidgets.QPushButton):
    def __init__(self, init_state=0):
        super().__init__()
        self.icons = [QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogNoButton),
                      QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogYesButton)]
        self.new_status(init_state)
        self.setStyleSheet('QPushButton,'
                            'QPushButton:default,'
                            'QPushButton:hover,'
                            'QPushButton:selected,'
                            'QPushButton:disabled,'
                            'QPushButton:pressed {'
                            'background-color: none;'
                               'border: none;'
                               'color: none;}')
    def new_status(self, x):
        self.setIcon(self.icons[int(x)])  # status icon
        
class DirectorySelectionWidget(QtWidgets.QWidget):
    def __init__(self, title='', simple=False, qedit_simple=False, parent=None):
        super().__init__(parent)
        self.simple_mode = simple
        
        # row 1 - QLabel and status icon (formerly info button)
        self.ddir_lbl_hbox = QtWidgets.QHBoxLayout()
        self.ddir_lbl_hbox.setContentsMargins(0,0,0,0)
        self.ddir_lbl_hbox.setSpacing(3)
        self.ddir_lbl = QtWidgets.QLabel()
        self.ddir_lbl.setText(title)
        
        # yes/no icons
        self.ddir_icon_btn = StatusIcon(init_state=0)#QtWidgets.QPushButton()  # folder status icon
        self.ddir_lbl_hbox.addWidget(self.ddir_icon_btn)
        self.ddir_lbl_hbox.addWidget(self.ddir_lbl)
        
        # row 2 - QLineEdit, and folder button
        self.qedit_hbox = QEdit_HBox(simple=qedit_simple)  # 3 x QLineEdit items
        self.ddir_btn = QtWidgets.QPushButton()  # file dlg launch button
        self.ddir_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.ddir_btn.setMinimumSize(30,30)
        self.ddir_btn.setIconSize(QtCore.QSize(20,20))
        
        # assemble layout
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setHorizontalSpacing(5)
        self.grid.setVerticalSpacing(8)
        self.grid.addLayout(self.ddir_lbl_hbox,      0, 1)
        #self.grid.addWidget(self.ddir_icon_btn, 1, 0)
        self.grid.addLayout(self.qedit_hbox,    1, 1, 1, 4)
        self.grid.addWidget(self.ddir_btn,      1, 5)
        self.grid.setColumnStretch(0, 0)
        self.grid.setColumnStretch(1, 2)
        self.grid.setColumnStretch(2, 2)
        self.grid.setColumnStretch(3, 2)
        self.grid.setColumnStretch(4, 2)
        self.grid.setColumnStretch(5, 0)
        
        if self.simple_mode:
            #self.ddir_icon_btn.hide()
            self.grid.setColumnStretch(0,0)
    
    def update_status(self, ddir, x=False):
        try:
            self.qedit_hbox.update_qedit(ddir, x)  # line edit
            self.ddir_icon_btn.new_status(x)  # status icon
        except:
            print('fu')
            #pdb.set_trace()
        #self.ddir_icon_btn.setIcon(self.ddir_icon_btn.icons[int(x)]) 
        
def make_probe_plot(probes):
    if probes.__class__ == prif.ProbeGroup:
        PG = probes
    elif probes.__class__ == prif.Probe:
        PG = prif.ProbeGroup()
        PG.add_probe(probes)
    else:
        raise Exception('Error: Probe plot input must be a Probe or ProbeGroup')
    
    fig, axs = plt.subplots(nrows=1, ncols=len(PG.probes), layout='tight')
    if type(axs) not in (list, np.ndarray):
        axs = [axs]
    for i,ax in enumerate(axs):
        P = PG.probes[i]
        plot_probe(P, with_contact_id=False, 
                   with_device_index=True, title=False, 
                   probe_shape_kwargs=dict(ec='black',lw=3), ax=ax)
        rads = list(map(lambda x: x['radius'], P.contact_shape_params))
        xshift = P.contact_positions[:,0] + np.array(rads) + 30
        kw = dict(ha='left', clip_on=False, 
                  bbox=dict(ec='none', fc='white', alpha=0.3))
        _ = [txt.set(x=xs, **kw) for txt,xs in zip(ax.texts,xshift)]
        ax.spines[['right','top','bottom']].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.label.set_fontsize('large')
    return fig, axs


class ProbeFileSimple(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gen_layout()
        
        xc = np.array([63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 
                       63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 
                       63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 
                       63.5, 63.5, 63.5, 63.5, 63.5])
        yc = np.array([1290, 1250, 1210, 1170, 1130, 1090, 1050, 1010,  970,  
                        930,  890,  850,  810,  770,  730,  690,  650,  610,  
                        570,  530,  490,  450,  410,  370,  330,  290,  250,  
                        210,  170,  130,   90,   50])
        shank = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        chMap = np.array([10, 11,  2,  3, 24, 25, 16, 17,  8,  9,  0,  1, 19, 
                          18, 26, 27,  5, 4, 12, 20, 13, 21, 29,  7,  6, 15, 
                          23, 31, 14, 22, 30, 28])
        
        self.name_input.setText('A1x32-Edge-10mm-40-177')
        self.xcoor_input.setPlainText(str(list(xc)))
        self.ycoor_input.setPlainText(str(list(yc)))
        self.shk_input.setPlainText(str(list(shank)))
        self.chMap_input.setPlainText(str(list(chMap)))
        
    def gen_layout(self):
        """ Layout for popup window """
        self.setWindowTitle('Create probe file')
        self.setMinimumWidth(250)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(20)
        
        # probe name
        row0 = QtWidgets.QHBoxLayout()
        row0.setSpacing(10)
        self.name_lbl = QtWidgets.QLabel('<big><u>Probe name</u></big>')
        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setObjectName('probe_text')
        self.name_input.setText('Probe_0')
        row0.addWidget(self.name_lbl)
        row0.addWidget(self.name_input)
        
        # probe geometry
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(15)
        row1_vb0 = QtWidgets.QVBoxLayout()
        row1_vb0.setSpacing(1)
        # electrode configuration
        self.config_lbl = QtWidgets.QLabel('Configuration')
        self.config_cbox = QtWidgets.QComboBox()
        self.config_cbox.addItems(['Linear','Polytrode','Tetrode'])
        row1_vb0.addWidget(self.config_lbl)
        row1_vb0.addWidget(self.config_cbox)
        row1.addLayout(row1_vb0)
        row1_vb1 = QtWidgets.QVBoxLayout()
        row1_vb1.setSpacing(1)
        # number of electrode columns
        self.ncol_lbl = QtWidgets.QLabel('# electrode columns')
        self.ncol_sbox = QtWidgets.QSpinBox()
        self.ncol_sbox.setMinimum(1)
        self.ncol_sbox.setEnabled(False)
        row1_vb1.addWidget(self.ncol_lbl)
        row1_vb1.addWidget(self.ncol_sbox)
        row1.addLayout(row1_vb1)
        
        # x and y-coordinates
        row2 = QtWidgets.QVBoxLayout()
        row2.setSpacing(2)
        self.x_lbl = QtWidgets.QLabel('<b>Enter X-coordinates for each electrode</b>')
        self.xcoor_input = QtWidgets.QTextEdit()
        self.xcoor_input.setObjectName('probe_text')
        row2.addWidget(self.x_lbl)
        row2.addWidget(self.xcoor_input)
        row3 = QtWidgets.QVBoxLayout()
        row3.setSpacing(2)
        self.y_lbl = QtWidgets.QLabel('<b>Enter Y-coordinates for each electrode</b>')
        self.ycoor_input = QtWidgets.QTextEdit()
        self.ycoor_input.setObjectName('probe_text')
        row3.addWidget(self.y_lbl)
        row3.addWidget(self.ycoor_input)
        
        # shank IDs
        row4 = QtWidgets.QVBoxLayout()
        row4.setSpacing(2)
        self.shk_lbl = QtWidgets.QLabel('<b>Electrode shank IDs (multi-shank probes only)</b>')
        self.shk_input = QtWidgets.QTextEdit()
        self.shk_input.setObjectName('probe_text')
        row4.addWidget(self.shk_lbl)
        row4.addWidget(self.shk_input)
        
        # headstage channel map
        row5 = QtWidgets.QVBoxLayout()
        row5.setSpacing(2)
        self.chMap_lbl = QtWidgets.QLabel('<b>Channel map (probe to headstage)</b>')
        self.chMap_input = QtWidgets.QTextEdit()
        self.chMap_input.setObjectName('probe_text')
        row5.addWidget(self.chMap_lbl)
        row5.addWidget(self.chMap_input)
        
        # action buttons
        bbox = QtWidgets.QHBoxLayout()
        self.makeprobe_btn = QtWidgets.QPushButton('Configure probe')
        self.makeprobe_btn.setEnabled(False)
        self.save_prbf_btn = QtWidgets.QPushButton('Save probe file')
        self.save_prbf_btn.setEnabled(False)
        bbox.addWidget(self.makeprobe_btn)
        bbox.addWidget(self.save_prbf_btn)
        
        self.layout.addLayout(row0)
        self.line0 = pyfx.DividerLine(lw=3, mlw=3)
        self.layout.addWidget(self.line0)
        #self.layout.addLayout(row1)
        self.layout.addLayout(row2)
        self.layout.addLayout(row3)
        self.layout.addLayout(row4)
        self.layout.addLayout(row5)
        self.layout.addLayout(bbox)
        
        self.name_input.textChanged.connect(self.check_probe)
        self.xcoor_input.textChanged.connect(self.check_probe)
        self.ycoor_input.textChanged.connect(self.check_probe)
        self.shk_input.textChanged.connect(self.check_probe)
        self.chMap_input.textChanged.connect(self.check_probe)
        self.makeprobe_btn.clicked.connect(self.construct_probe)
        self.save_prbf_btn.clicked.connect(self.save_probe_file)
    
        self.setStyleSheet('QTextEdit { border : 2px solid gray; }')
        
        
    def check_probe(self):
        print('check_probe called')
        self.makeprobe_btn.setEnabled(False)
        self.save_prbf_btn.setEnabled(False)
        probe_name = self.name_input.text()
        xdata      = ''.join(self.xcoor_input.toPlainText().split())
        ydata      = ''.join(self.ycoor_input.toPlainText().split())
        shk_data   = ''.join(self.shk_input.toPlainText().split())
        cmap_data  = ''.join(self.chMap_input.toPlainText().split())
        
        try: 
            xc = np.array(eval(xdata), dtype='float')   # x-coordinates
            yc = np.array(eval(ydata), dtype='float')   # y-coordinates
            if shk_data  == '' : shk = np.ones_like(xc, dtype='int')    # shank IDs
            else               : shk = np.array(eval(shk_data), dtype='int')
            if cmap_data == '' : cmap = np.arange(xc.size, dtype='int') # channel map
            else               : cmap = np.array(eval(cmap_data), dtype='int')
        except:
            #pdb.set_trace()
            return  # failed to convert text to array
        print('got arrays')
        n = [xc.size, yc.size, shk.size, cmap.size]
        if len(np.unique(n)) > 1: return              # mismatched array lengths
        if cmap.size != np.unique(cmap).size: return  # duplicates in channel map 
        if len(probe_name.split()) > 1: return        # spaces in probe name
        if probe_name == '': return                   # probe name left blank
        
        self.probe_name = probe_name
        self.xc   = xc
        self.yc   = yc
        self.shk  = shk
        self.cmap = cmap
        self.makeprobe_btn.setEnabled(True)
        self.save_prbf_btn.setEnabled(True)
        
    
    def construct_probe(self, arg=None, pplot=True):
        print('construct_probe called')
        # create dataframe, sort by shank > x-coor > y-coor
        pdf = pd.DataFrame(dict(chanMap=self.cmap, xc=self.xc, yc=self.yc, shank=self.shk))
        df = pdf.sort_values(['shank','xc','yc'], ascending=[True,True,False]).reset_index(drop=True)
        #ncoors = np.array(df[['xc','yc']].nunique())  # (ncols, nrows)
        #ndim = max((ncoors > 1).astype('int').sum(), 1)
        
        # initialize probe data object
        self.probe = prif.Probe(ndim=2, name=self.probe_name)
        self.probe.set_contacts(np.array(df[['xc','yc']]), shank_ids=df.shank, 
                                contact_ids=df.index.values)
        self.probe.set_device_channel_indices(df.chanMap)
        self.probe.create_auto_shape('tip', margin=30)
        #self.probe_df = self.probe.to_dataframe()
        #self.probe_df['chanMap'] = np.array(self.probe.device_channel_indices)
        if pplot:
            fig, axs = make_probe_plot(self.probe)
            fig_popup = MatplotlibPopup(fig, parent=self)
            qrect = pyfx.ScreenRect(perc_height=0.9, perc_width=0.2, keep_aspect=False)
            fig_popup.setGeometry(qrect)
            fig_popup.setWindowTitle(self.probe_name)
            fig_popup.show()
            fig_popup.raise_()
            
            #self.show_probe_plot()
            
    
    # def show_probe_plot(self):
    #     fig, ax = plt.subplots(layout='tight')
    #     plot_probe(self.probe, with_contact_id=False, with_device_index=True, 
    #                title=False, probe_shape_kwargs=dict(ec='black',lw=3), ax=ax)
    #     rads = list(map(lambda x: x['radius'], self.probe.contact_shape_params))
    #     xshift = self.probe.contact_positions[:,0] + np.array(rads) + 30
    #     kw = dict(ha='left', clip_on=False, 
    #               bbox=dict(ec='none', fc='white', alpha=0.3))
    #     _ = [txt.set(x=xs, **kw) for txt,xs in zip(ax.texts,xshift)]
    #     ax.spines[['right','top','bottom']].set_visible(False)
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.label.set_fontsize('large')
    #     #ax.set_xlim(right=ax.get_xlim()[1] * 1.5)
        
    #     fig_popup = MatplotlibPopup(fig, parent=self)
    #     qrect = pyfx.ScreenRect(perc_height=0.9, perc_width=0.2, keep_aspect=False)
    #     fig_popup.setGeometry(qrect)
    #     fig_popup.setWindowTitle(self.probe_name)
    #     fig_popup.show()
    #     fig_popup.raise_()
    #     print('show_probe_plot called')
    
    def save_probe_file(self, arg=None, extension='.json'):
        # create probe object, select file name/location
        self.construct_probe(pplot=False)
        filename = f'{self.probe_name}_config{extension}'
        fpath,_ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save probe file',
                                                        str(Path(os.getcwd(), filename)))
        if not fpath: return
        
        res = ephys.write_probe_file(self.probe, fpath)
        self.probe_filepath = fpath
        if res:
            # pop-up messagebox appears when save is complete
            msg = 'Probe configuration saved!\nExit window?'
            msgbox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, '', msg, 
                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, self)
            # set check icon
            chk_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
            px_size = msgbox.findChild(QtWidgets.QLabel, 'qt_msgboxex_icon_label').pixmap().size()
            msgbox.setIconPixmap(chk_icon.pixmap(px_size))
            res = msgbox.exec()
            if res == QtWidgets.QMessageBox.Yes:
                self.accept()
        
    
def clean(mode, base, last, init_ddir=''):
    if os.path.exists(init_ddir) and os.path.isdir(init_ddir):
        return init_ddir
    if mode==0: return base
    if mode==1: return last
    if mode==2:
        res = last if (Path(base) in Path(last).parents) else base
        return res


class RawDirectorySelectionPopup(QtWidgets.QDialog):
    RAW_DDIR_VALID = False
    PROCESSED_DDIR_VALID = False
    PROBE_CONFIG_VALID = False
    
    
    def __init__(self, mode=2, raw_ddir='', processed_ddir='', probe_ddir='', parent=None):
        # 0==base, 1=last visited, 2=last visited IF it's within base
        super().__init__(parent)
        
        bases = raw_base, processed_base, probe_base = ephys.base_dirs()
        # get most recently entered directory
        qfd = QtWidgets.QFileDialog()
        last_ddir = str(qfd.directory().path())
        
        self.raw_ddir = clean(mode, raw_base, last_ddir, str(raw_ddir))
        self.processed_ddir = clean(mode, processed_base, last_ddir, str(processed_ddir))
        self.probe_ddir = clean(mode, probe_base, last_ddir, str(probe_ddir))
        
        self.gen_layout()
        self.ddir_gbox2.hide()
        self.probe_gbox.hide()
        
        self.update_raw_ddir()
        if len(os.listdir(self.processed_ddir)) == 0:
            self.update_processed_ddir()
        else:
            self.ddw2.update_status(self.processed_ddir, False)
        try:
            self.probe = ephys.read_probe_file(self.probe_ddir)  # get Probe or None
        except:
            pdb.set_trace()
        self.update_probe_obj()
            
    
    def gen_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(20)
        gbox_ss = 'QGroupBox {background-color : rgba(230,230,230,255);}'# border : 2px ridge gray; border-radius : 4px;}'
        
        ###   SELECT RAW DATA FOLDER
        
        self.ddir_gbox = QtWidgets.QGroupBox()
        self.ddir_gbox.setStyleSheet(gbox_ss)
        ddir_vbox = pyfx.InterWidgets(self.ddir_gbox, 'v')[2]
        # basic directory selection widget
        self.ddw = DirectorySelectionWidget(title='<b><u>Raw data directory</u></b>')
        self.qedit_hbox = self.ddw.qedit_hbox
        # types of recording files
        self.oe_radio = QtWidgets.QRadioButton('Open Ephys (.oebin)')
        self.nn_radio = QtWidgets.QRadioButton('NeuroNexus (.xdat.json)')
        self.manual_radio = QtWidgets.QRadioButton('Upload custom file')
        for btn in [self.oe_radio, self.nn_radio, self.manual_radio]:
            btn.setAutoExclusive(False)
            btn.setEnabled(False)
            btn.setStyleSheet('QRadioButton {color : black;}'
                              'QRadioButton:disabled {color : black;}')
        self.ddw.grid.addWidget(self.oe_radio, 2, 1, 1, 2)
        self.ddw.grid.addWidget(self.nn_radio, 2, 3, 1, 2)
        ddir_vbox.addWidget(self.ddw)
        # manual file upload with custom parameters 
        custom_bar = QtWidgets.QFrame()
        custom_bar.setFrameShape(QtWidgets.QFrame.Panel)
        custom_bar.setFrameShadow(QtWidgets.QFrame.Sunken)
        frame_hlay = QtWidgets.QHBoxLayout(custom_bar)
        self.manual_upload_btn = QtWidgets.QPushButton()
        self.manual_upload_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward))
        frame_hlay.addWidget(self.manual_radio)
        frame_hlay.addWidget(self.manual_upload_btn)
        ddir_vbox.addWidget(custom_bar)
        
        ###   CREATE PROCESSED DATA FOLDER
        
        self.ddir_gbox2 = QtWidgets.QGroupBox()
        self.ddir_gbox2.setStyleSheet('QGroupBox {border-width : 0px; font-weight : bold; text-decoration : underline;}')
        #ddir_gbox2.setStyleSheet(gbox_ss)
        ddir_vbox2 = pyfx.InterWidgets(self.ddir_gbox2, 'v')[2]
        #ddir_vbox2.setContentsMargins(6,10,6,10)
        #ddir_vbox2.setContentsMargins(11,15,11,15)
        # create/overwrite processed data directory
        self.ddw2 = DirectorySelectionWidget(title='<b><u>Save data</u></b>', simple=True)
        #self.ddw2.ddir_lbl.hide()
        self.qedit_hbox2 = self.ddw2.qedit_hbox
        ddir_vbox2.addWidget(self.ddw2)
        
        ###   LOAD/CREATE PROBE FILE
        
        self.probe_gbox = QtWidgets.QGroupBox()
        self.probe_gbox.setStyleSheet('QGroupBox {border-width : 0px; font-weight : bold; text-decoration : underline;}')
        #prbw = QtWidgets.QWidget()
        probe_vbox = pyfx.InterWidgets(self.probe_gbox, 'v')[2] #findme
        #probe_hbox.setContentsMargins(11,15,11,15)
        #probe_hbox = QtWidgets.QHBoxLayout()
        # title and status button
        row0 = QtWidgets.QHBoxLayout()
        row0.setContentsMargins(0,0,0,0)
        row0.setSpacing(3)
        self.prb_icon_btn = StatusIcon(init_state=0)
        probe_lbl = QtWidgets.QLabel('<b><u>Probe(s)</u></b>')
        row0.addWidget(self.prb_icon_btn)
        row0.addWidget(probe_lbl)
        row0.addStretch()
        # displayed probe name
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(5)
        self.probe_qlabel = QtWidgets.QLabel('---')
        self.probe_qlabel.setStyleSheet('QLabel {background-color:white;'
                                        'border:1px solid gray;'
                                        #'border-right:none;'
                                        'padding:10px;}')
        self.probe_qlabel.setAlignment(QtCore.Qt.AlignCenter)
        probe_x = QtWidgets.QLabel('x')
        probe_x.setStyleSheet('QLabel {background-color:transparent;'
                                      #'border:1px solid gray;'
                                      #'border-right:none;'
                                      #'border-left:none;'
                                      'font-size:14pt; '#'font-weight:bold;'
                                      #'padding: 10px 5px;'
                                      '}')
        probe_x.setAlignment(QtCore.Qt.AlignCenter)
        self.probe_n = QtWidgets.QSpinBox()
        self.probe_n.setAlignment(QtCore.Qt.AlignCenter)
        self.probe_n.setMinimum(1)
        #self.probe_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        #self.probe_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.probe_n.setStyleSheet('QSpinBox {'
                                   #'background-color:transparent;'
                                   #'border:3px solid black;'
                                  # 'border-color:white;'
                                   'font-size:14pt; font-weight:bold;'
                                   'padding:10px 0px;}')
        self.probe_n.valueChanged.connect(self.update_nchannels)
        
        probe_arrow = QtWidgets.QLabel('\u27A4')  # unicode ➤
        probe_arrow.setAlignment(QtCore.Qt.AlignCenter)
        probe_arrow.setStyleSheet('QLabel {padding: 0px 5px;}')
        self.total_channel_fmt = '<code>{}<br>channels</code>'
        self.total_channel_lbl = QtWidgets.QLabel(self.total_channel_fmt.format('-'))
        self.total_channel_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.total_channel_lbl.setStyleSheet('QLabel {background-color:rgba(255,255,255,150); padding:2px;}')
        row1.addWidget(self.probe_qlabel, stretch=2)
        row1.addWidget(probe_x, stretch=0)
        row1.addWidget(self.probe_n, stretch=0)
        row1.addWidget(probe_arrow, stretch=0)
        row1.addWidget(self.total_channel_lbl, stretch=1)
        # probe buttons
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(10)
        self.prbf_load = QtWidgets.QPushButton('Load')
        self.prbf_load.setStyleSheet('QPushButton {padding:5px;}')
        #self.prbf_load.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.prbf_view = QtWidgets.QPushButton('View')
        self.prbf_view.setStyleSheet('QPushButton {padding:5px;}')
        self.prbf_view.setEnabled(False)
        self.prbf_make = QtWidgets.QPushButton('New')
        self.prbf_make.setStyleSheet('QPushButton {padding:5px;}')
        
        #row1.addWidget(self.probe_qlabel, stretch=4)
        row2.addWidget(self.prbf_load)
        row2.addWidget(self.prbf_view)
        row2.addWidget(self.prbf_make)
        probe_vbox.addLayout(row0)
        probe_vbox.addLayout(row1)
        #probe_vbox.addWidget(self.probe_qlabel)
        probe_vbox.addLayout(row2)
        
        splitbox = QtWidgets.QHBoxLayout()
        splitbox.setSpacing(10)
        splitbox.setContentsMargins(0,0,0,0)
        ggbox = QtWidgets.QGroupBox()
        ggv = QtWidgets.QVBoxLayout(ggbox)
        gg_lbl = QtWidgets.QLabel('<b>View<br><u>Settings</u></b>')
        gg_lbl.setAlignment(QtCore.Qt.AlignCenter)
        ggv.addWidget(gg_lbl)
        
        
        self.big_btn = ShowHideBtn(text_shown='', init_show=True)
        #self.big_btn.
        self.big_btn.setFixedWidth(60)
        #splitbox.addWidget(self.big_btn)
        #splitbox.addWidget(probe_gbox)
        #splitbox.addWidget(ggbox)
        
        
        ###   ACTION BUTTONS
        
        # continue button
        bbox = QtWidgets.QHBoxLayout()
        self.continue_btn = QtWidgets.QPushButton('Next')
        self.continue_btn.setEnabled(False)
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        bbox.addWidget(self.cancel_btn)
        bbox.addWidget(self.continue_btn)
        
        # assemble layout
        self.layout.addWidget(self.ddir_gbox)
        self.layout.addWidget(self.ddir_gbox2)
        #self.layout.addLayout(splitbox)
        self.layout.addWidget(self.probe_gbox)
        line0 = pyfx.DividerLine()
        self.layout.addWidget(line0)
        self.layout.addLayout(bbox)
        
        # connect buttons
        self.ddw.ddir_btn.clicked.connect(self.select_ddir)
        self.ddw2.ddir_btn.clicked.connect(self.make_ddir)
        self.prbf_load.clicked.connect(self.load_probe_from_file)
        self.prbf_view.clicked.connect(self.view_probe)
        self.prbf_make.clicked.connect(self.probe_config_popup)
        self.continue_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    
    
    def load_probe_from_file(self, arg=None, fpath=None):
        """ Load probe data from filepath (if None, user can select a probe file """
        if fpath is None:
            ffilter = 'Probe files (*.json *.mat *.prb *.csv)'
            fpath,_ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select probe file', self.probe_ddir, ffilter)
            if not fpath: return
        # try to load data - could be probe, could be None
        self.probe = ephys.read_probe_file(fpath)
        self.update_probe_obj()
        
    def view_probe(self):
        print('view_probe called')
        PG = ephys.make_probe_group(self.probe, self.probe_n.value())
        fig, axs = make_probe_plot(PG)
        fig_popup = MatplotlibPopup(fig, parent=self)
        qrect = pyfx.ScreenRect(perc_height=1.0, perc_width=min(0.2*len(PG.probes), 1.0), 
                                keep_aspect=False)
        fig_popup.setGeometry(qrect)
        #fig_popup.setWindowTitle(self.probe_name)
        fig_popup.show()
        fig_popup.raise_()
        
    def probe_config_popup(self):
        print('probe_config_popup called')
        popup = ProbeFileSimple(parent=self)
        popup.show()
        popup.raise_()
        res = popup.exec()
        if not res:
            return
        # popup saved new probe configuration to file; load it!
        self.probe_ddir = popup.probe_filepath
        self.probe = ephys.read_probe_file(self.probe_ddir)
        self.update_probe_obj()
        
    def update_probe_obj(self):
        x = bool(self.probe is not None)
        self.prb_icon_btn.new_status(x)
        self.update_nchannels()
        if x:
            self.probe_qlabel.setText(self.probe.name)
            self.probe.create_auto_shape('tip', margin=30)
            #self.probe_df = self.probe.to_dataframe()
            #self.probe_df['chanMap'] = np.array(self.probe.device_channel_indices)
        else:
            self.probe_qlabel.setText('---')
        self.prbf_view.setEnabled(bool(x))
        self.PROBE_CONFIG_VALID = bool(x)
        self.continue_btn.setEnabled(bool(self.RAW_DDIR_VALID and self.PROCESSED_DDIR_VALID and self.PROBE_CONFIG_VALID))
    
    def update_nchannels(self):
        if self.probe is None:
            nch = '-'
        else:
            nch = self.probe.get_contact_count() * self.probe_n.value()
        self.total_channel_lbl.setText(self.total_channel_fmt.format(nch))
        
    def select_ddir(self):
        # open file popup to select raw data folder
        dlg = FileDialog(init_ddir=self.raw_ddir, parent=self)
        res = dlg.exec()
        if res:
            self.raw_ddir = str(dlg.directory().path())
            self.update_raw_ddir()
            
            
    def update_raw_ddir(self):
        # check if raw data files are present
        files = os.listdir(self.raw_ddir)
        xdat_files = [f for f in files if f.endswith('.xdat.json')]
        a = bool('structure.oebin' in files)
        b = bool(len(xdat_files) > 0)
        x = bool(a or b)
        
        # update widgets
        try:
            self.ddw.update_status(self.raw_ddir, x)
        except:
            pdb.set_trace()
        self.oe_radio.setChecked(a)  # recording system buttons
        self.nn_radio.setChecked(b)
        
        self.RAW_DDIR_VALID = bool(x)
        self.ddir_gbox2.setVisible(x)
        self.probe_gbox.setVisible(bool(x and self.PROCESSED_DDIR_VALID))
        self.adjustSize()
        self.continue_btn.setEnabled(bool(self.RAW_DDIR_VALID and self.PROCESSED_DDIR_VALID and self.PROBE_CONFIG_VALID))
    
    
    def make_ddir(self):
        # open file popup to create processed data folder
        dlg = FileDialog(init_ddir=self.processed_ddir, load_or_save='save', parent=self)
        res = dlg.exec()
        if res:
            self.processed_ddir = str(dlg.directory().path())
            self.update_processed_ddir()
    
    
    def update_processed_ddir(self):
        nexisting = len(os.listdir(self.processed_ddir))
        if nexisting > 0:
            txt = (f'The directory <code>{self.processed_ddir.split(os.sep)[-1]}</code> contains '
                   f'{nexisting} items.<br><br>I have taken away your overwrite '
                   'privileges for the time being.<br><br>Stop almost deleting important things!!')
            msg = '<center>{}</center>'.format(txt)
            res = QtWidgets.QMessageBox.warning(self, 'fuck you', msg, 
                                                QtWidgets.QMessageBox.NoButton, QtWidgets.QMessageBox.Close)
            if res == QtWidgets.QMessageBox.Yes:
                return True
            return False
            
        # update widgets
        self.ddw2.update_status(self.processed_ddir, True)
        
        self.PROCESSED_DDIR_VALID = True
        self.probe_gbox.setVisible(True)
        self.adjustSize()
        self.continue_btn.setEnabled(bool(self.RAW_DDIR_VALID and self.PROCESSED_DDIR_VALID and self.PROBE_CONFIG_VALID))
    
        
    def accept(self):
        print(f'Raw data folder: {self.raw_ddir}')
        print(f'Save folder: {self.processed_ddir}')
        super().accept()
        
        
        
class AnalysisBtns(QtWidgets.QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # View/save event channels; always available for all processed data folders
        self.option1_widget = self.create_widget('Select event channels', 'green')
        self.option1_btn = self.option1_widget.findChild(QtWidgets.QPushButton)
        # Classify dentate spikes; requires event channels and probe DS_DF files
        self.option2_widget = self.create_widget('Classify dentate spikes', 'blue')
        self.option2_btn = self.option2_widget.findChild(QtWidgets.QPushButton)
        
        #self.btn_grp.addButton(self.option1_btn)
        #self.btn_grp.addButton(self.option2_btn)
        self.option1_widget.setEnabled(False)
        self.option2_widget.setEnabled(False)
        
        #self.btn_grp.buttonToggled.connect(self.action_toggled)
    
    def create_widget(self, txt, c):
        widget = QtWidgets.QWidget()
        widget.setContentsMargins(0,0,0,0)
        hlay = QtWidgets.QHBoxLayout(widget)
        hlay.setContentsMargins(0,0,0,0)
        hlay.setSpacing(8)
        btn = QtWidgets.QPushButton()
        #btn.setCheckable(True)
        #clight = pyfx.hue(c, 0.7, 1); cdark = pyfx.hue(c, 0.4, 0)#; cdull = pyfx.hue(clight, 0.8, 0.5, alpha=0.5)
        btn.setStyleSheet(btn_ss % (pyfx.hue(c, 0.7, 1),  pyfx.hue(c, 0.4, 0)))
        lbl = QtWidgets.QLabel(txt)
        hlay.addWidget(btn)
        hlay.addWidget(lbl)
        #self.btn_grp.addButton(btn)
        return widget
        
    
    def ddir_toggled(self, ddir, current_probe=0):
        self.option1_widget.setEnabled(False)
        self.option2_widget.setEnabled(False)
        
        if not os.path.isdir(ddir):
            return
        
        files = os.listdir(ddir)
        # required: basic LFP files
        if all([bool(f in files) for f in ['lfp_bp.npz', 'lfp_time.npy', 'lfp_fs.npy']]):
            self.option1_widget.setEnabled(True)  # req: basic LFP data
        
        # required: event channels file, DS_DF file for current probe
        if f'DS_DF_{current_probe}' in files and f'theta_ripple_hil_chan_{current_probe}.npy' in files:
            self.option2_widget.setEnabled(True)


class ProcessedDirectorySelectionPopup(QtWidgets.QDialog):
    def __init__(self, init_ddir='', go_to_last=False, parent=None):
        super().__init__(parent)
        #self.info = None
        self.current_probe = -1
        
        if go_to_last == True:
            qfd = QtWidgets.QFileDialog()
            self.ddir = qfd.directory().path()
        else:
            if init_ddir == '':
                _, self.ddir, _ = ephys.base_dirs()
            else:
                self.ddir = init_ddir
        
        self.gen_layout()
        
        if os.path.isdir(self.ddir):
            self.update_ddir(self.ddir)
            
    
    def gen_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(20)
        
        ###   SELECT PROCESSED DATA FOLDER
        
        self.ddir_gbox = QtWidgets.QGroupBox()
        #ddir_gbox.setStyleSheet(gbox_ss)
        ddir_vbox = pyfx.InterWidgets(self.ddir_gbox, 'v')[2]
        # basic directory selection widget
        self.ddw = DirectorySelectionWidget(title='<b><u>Processed Data Folder</u></b>')
        self.qedit_hbox = self.ddw.qedit_hbox
        self.probe_dropdown = QtWidgets.QComboBox()
        self.probe_dropdown.hide()
        self.info_view_btn = QtWidgets.QPushButton('More Info')
        self.info_view_btn.hide()
        self.ddw.grid.addWidget(self.probe_dropdown, 2, 0, 1, 3)
        self.ddw.grid.addWidget(self.info_view_btn, 2, 3, 1, 3)
        ddir_vbox.addWidget(self.ddw)
        
        ###   ACTION BUTTONS
        
        self.ab = AnalysisBtns()
        intraline = pyfx.DividerLine()
        self.ddw.grid.addWidget(intraline, 3, 0, 1, 6)
        self.ddw.grid.addWidget(self.ab.option1_widget, 4, 1)
        self.ddw.grid.addWidget(self.ab.option2_widget, 5, 1)
        self.ddw.grid.setRowMinimumHeight(2, 20)

        # assemble layout
        self.layout.addWidget(self.ddir_gbox)
        
        # connect buttons
        self.ddw.ddir_btn.clicked.connect(self.select_ddir)
        self.probe_dropdown.currentTextChanged.connect(self.update_probe)
        self.info_view_btn.clicked.connect(self.show_info_popup)
    
    def show_info_popup(self):
        print('show_info_popup called --> under construction')
        #info_popup = InfoView(info=self.info, parent=self)
        #info_popup.show()
        #info_popup.raise_()
    
    # def info_popup(self):
    #     #dlg = QtWidgets.QDialog(self)
    #     #layout = QtWidgets.QVBoxLayout(dlg)
    #     self.
    # def emit_signal(self):
    #     if self.ab.option1_btn.isChecked():
    #         self.launch_ch_selection_signal.emit()
            
    #     elif self.ab.option2_btn.isChecked():
    #         self.launch_ds_class_signal.emit()
        
    def select_ddir(self):
        # open file popup to select processed data folder
        dlg = FileDialog(init_ddir=self.ddir, parent=self)
        res = dlg.exec()
        if res:
            self.update_ddir(str(dlg.directory().path()))
    
    def update_ddir(self, ddir):
        print('update_ddir called')
        self.ddir = ddir
        x = validate_processed_ddir(self.ddir)  # valid data folder?
        self.ddw.update_status(self.ddir, x)    # update folder path/icon style
        # reset probe dropdown, populate with probes in $info
        self.probe_dropdown.blockSignals(True)
        for i in reversed(range(self.probe_dropdown.count())):
            self.probe_dropdown.removeItem(i)
        self.probe_dropdown.setVisible(x)
        self.info_view_btn.setVisible(x)
        if x:
            # read in recording info, clear and update probe dropdown menu
            #self.info = ephys.load_recording_info(self.ddir)
            probe_group = prif.io.read_probeinterface(Path(self.ddir, 'probe_group'))
            items = [f'probe {i}' for i in range(len(probe_group.probes))]
            self.probe_dropdown.addItems(items)
        #else:
        #    self.info = None
        # probe index (e.g. 0,1,...) if directory is valid, otherwise -1
        self.probe_dropdown.blockSignals(False)
        #self.current_probe = self.probe_dropdown.currentIndex()
        self.update_probe()
    
    
    def update_probe(self):
        print('update_probe called')
        self.current_probe = self.probe_dropdown.currentIndex()
        self.ab.ddir_toggled(self.ddir, self.current_probe)  # update action widgets


class MatplotlibPopup(QtWidgets.QDialog):
    """ Simple popup window to display Matplotlib figure """
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.canvas_layout = QtWidgets.QVBoxLayout()
        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas_layout.addWidget(self.toolbar)
        self.canvas_layout.addWidget(self.canvas)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.canvas_layout)
        self.setLayout(self.layout)


def create_widget_row(key, val, param_type, description='', **kw):
    """
    key, val - parameter name, value
    param_type - 'num', 'int', 'keyword', 'text', 'bool', 'freq_band'
    """
    def set_val(w, val, **kw):
        if w.minimum() > val: w.setMinimum(val)
        if w.maximum() < val: w.setMaximum(val)
        if 'mmin' in kw    : w.setMinimum(kw['mmin'])
        if 'mmax' in kw    : w.setMaximum(kw['mmax'])
        if 'step' in kw   : w.setSingleStep(kw['step'])
        if 'suffix' in kw : w.setSuffix(kw['suffix'])
        if 'dec' in kw and w.__class__ == QtWidgets.QDoubleSpinBox:
            w.setDecimals(kw['dec'])
        w.setValue(val)
        return w
    
    lbl = QtWidgets.QLabel(key)
    lbl.setToolTip(description)
    if param_type in ['num','int']:
        if param_type == 'int':
            w = QtWidgets.QSpinBox()
            val = int(val)
        else: 
            w = QtWidgets.QDoubleSpinBox()
        w = set_val(w, val, **kw)
    elif param_type == 'keyword':
        w = QtWidgets.QComboBox()
        items = kw.get('opts', [])
        if val not in items: items.insert(0, val)
        w.addItems(items)
        w.setCurrentText(val)
    elif param_type == 'text':
        w = QtWidgets.QLineEdit()
        w.setText(val)
    else:
        # hbox = QtWidgets.QWidget()
        # hlay = QtWidgets.QHBoxLayout(hbox)
        # hlay.setSpacing(0)
        if param_type == 'bool':
            w0 = QtWidgets.QRadioButton('True')
            w0.setChecked(bool(val)==True)
            w1 = QtWidgets.QRadioButton('False')
            w1.setChecked(bool(val)==False)
        elif param_type == 'freq_band':
            w0 = QtWidgets.QDoubleSpinBox()
            w0 = set_val(w0, val[0], **kw)
            w1 = QtWidgets.QDoubleSpinBox()
            w1 = set_val(w1, val[1], **kw)
            # dash = QtWidgets.QLabel(' - ')
            # dash.setAlignment(QtCore.Qt.AlignCenter)
            # hlay.addWidget(dash)
        # hlay.insertWidget(0, w0)
        # hlay.insertWidget(-1, w1)
        w = [w0, w1]
    return lbl, w


def create_fband_row(w0, w1, mid=' - '):
    hlay = QtWidgets.QHBoxLayout()
    hlay.setSpacing(0)
    midw = QtWidgets.QLabel(mid)
    midw.setAlignment(QtCore.Qt.AlignCenter)
    hlay.addWidget(w0)
    hlay.addWidget(midw)
    hlay.addWidget(w1)
    return hlay

class ParamWidgets(object):
    
    def __init__(self, PARAMS):
        D = pd.Series(PARAMS)
        L = {}
        W = {}
        HLAY = {}
        
        # downsampled LFP
        L['lfp_fs'], W['lfp_fs'] = create_widget_row('lfp_fs', D.lfp_fs, 'int', mmax=30000, suffix=' Hz')
        
        # theta, slow gamma, and fast gamma bands, DS bandpass, and SWR bandpass
        for k in ['theta', 'slow_gamma', 'fast_gamma', 'ds_freq', 'swr_freq']:
            L[k], W[k] = create_widget_row(k, D[k], 'freq_band', dec=1, mmax=1000, suffix=' Hz')
            HLAY[k] = create_fband_row(*W[k])
        
        # DS detection params
        L['ds_height_thr'], W['ds_height_thr'] = create_widget_row('ds_height_thr', D['ds_height_thr'], 'num', dec=1, step=0.1, suffix=' STD')
        L['ds_dist_thr'], W['ds_dist_thr'] = create_widget_row('ds_dist_thr', D['ds_dist_thr'], 'num', dec=1, step=0.1, suffix=' s')
        L['ds_prom_thr'], W['ds_prom_thr'] = create_widget_row('ds_prom_thr', D['ds_prom_thr'], 'num', dec=1, step=0.1)
        L['ds_wlen'], W['ds_wlen'] = create_widget_row('ds_wlen', D['ds_wlen'], 'num', dec=3, step=0.005, suffix=' s')
        
        # SWR detection params
        L['swr_ch_bound'], W['swr_ch_bound'] = create_widget_row('swr_ch_bound', D['swr_ch_bound'], 'int', suffix=' channels')
        L['swr_height_thr'], W['swr_height_thr'] = create_widget_row('swr_height_thr', D['swr_height_thr'], 'num', dec=1, step=0.1, suffix=' STD')
        L['swr_min_thr'], W['swr_min_thr'] = create_widget_row('swr_min_thr', D['swr_min_thr'], 'num', dec=1, step=0.1, suffix=' STD')
        L['swr_dist_thr'], W['swr_dist_thr'] = create_widget_row('swr_dist_thr', D['swr_dist_thr'], 'num', dec=1, step=0.1, suffix=' s')
        L['swr_min_dur'], W['swr_min_dur'] = create_widget_row('swr_min_dur', D['swr_min_dur'], 'int', mmax=1000, suffix=' ms')
        L['swr_freq_thr'], W['swr_freq_thr'] = create_widget_row('swr_freq_thr', D['swr_freq_thr'], 'num', mmax=1000, dec=1, step=1, suffix=' Hz')
        L['swr_freq_win'], W['swr_freq_win'] = create_widget_row('swr_freq_win', D['swr_freq_win'], 'int', mmax=1000, suffix=' ms')
        L['swr_maxamp_win'], W['swr_maxamp_win'] = create_widget_row('swr_maxamp_win', D['swr_maxamp_win'], 'int', mmax=1000, suffix=' ms')
        
        # CSD calculation params
        methods = ['standard', 'delta', 'step', 'spline']
        filters = ['gaussian','identity','boxcar','hamming','triangular']
        L['csd_method'], W['csd_method'] = create_widget_row('csd_method', D['csd_method'], 'keyword',opts=methods)
        L['f_type'], W['f_type'] = create_widget_row('f_type', D['f_type'], 'keyword',opts=filters)
        L['f_order'], W['f_order'] = create_widget_row('f_order', D['f_order'], 'int')
        L['f_sigma'], W['f_sigma'] = create_widget_row('f_sigma', D['f_sigma'], 'num', dec=1, step=0.1)
        L['tol'], W['tol'] = create_widget_row('tol', D['tol'], 'num', dec=8, step=0.0000001)
        L['spline_nsteps'], W['spline_nsteps'] = create_widget_row('spline_nsteps', D['spline_nsteps'], 'int')
        L['vaknin_el'], W['vaknin_el'] = create_widget_row('vaknin_el', D['vaknin_el'], 'bool')

        self.LABELS = L
        self.WIDGETS = W
        self.HLAY = HLAY
        
# class ParamView(QtWidgets.QDialog):
#     def __init__(self, params=None, ddir=None, parent=None):
#         super().__init__(parent)
#         qrect = pyfx.ScreenRect(perc_height=0.5, perc_width=0.3, keep_aspect=False)
#         self.setGeometry(qrect)
        
#         params = None
#         # get info, convert to text string
#         if params is None:
#             params = ephys.read_param_file('default_params.txt')
#         if 'RAW_DATA_FOLDER' in params: rdf = params.pop('RAW_DATA_FOLDER')
#         if 'PROCESSED_DATA_FOLDER' in params: pdf = params.pop('PROCESSED_DATA_FOLDER')
        
        
#         keys, vals = zip(*params.items())
#         vstr = [*map(str,vals)]
#         klen = max(map(len, keys))  # get longest keyword
#         #vlen = max(map(len,vstr)) # get longest value string
        
#         rows3 = ['<pre>' + k + '_'*(klen-len(k)) + ' : ' + v + '</pre>' for k,v in zip(keys,vstr)]
#         ttxt = ''.join(rows3)
        
#         self.textwidget = QtWidgets.QTextEdit(ttxt)
        
        
#         # fmt = f'{{:^{klen}}} : {{:>{vlen}}}'
#         # rows = [fmt.format(k,v) for k,v in zip(keys,vstr)]
#         # TEXT = os.linesep.join(rows)
            
#         # a = 'i am a key'
#         # b = 'i am a value'
#         # TEXT = f'{a:<30} : {b:<60}'
        
#         qfont = QtGui.QFont("Monospace")
#         qfont.setStyleHint(QtGui.QFont.TypeWriter)
#         # qfont.setPointSize(10)
#         # qfm = QtGui.QFontMetrics(qfont)
            
#         # create QTextEdit for displaying file
        
#         #self.textwidget.setAlignment(QtCore.Qt.AlignCenter)
#         #self.textwidget.setFont(qfont)
        
#         # fm = self.textwidget.fontMetrics()
#         # klen2 = max(map(fm.horizontalAdvance, keys))
#         # vlen2 = max(map(fm.horizontalAdvance, vstr))
#         # fmt2 = f'{{:<{klen2}}} {{:<{vlen2}}}'
        
#         # rows3 = [f'{k:>20} : {v}' for k,v in zip(keys,vstr)]
        
#         # rows2 = [fmt2.format(k,v) for k,v in zip(keys,vstr)]
#         # TEXT2 = os.linesep.join(rows2)
        
        
#         # for row in rows3:
#         #     self.textwidget.append(row)
#             #self.textwidget.appendPlainText(row)
        
        
#         layout = QtWidgets.QVBoxLayout(self)
#         layout.addWidget(self.textwidget)
        
    
class InfoView(QtWidgets.QDialog):
    def __init__(self, info=None, ddir=None, parent=None):
        super().__init__(parent)
        qrect = pyfx.ScreenRect(perc_height=0.5, perc_width=0.3, keep_aspect=False)
        self.setGeometry(qrect)
        
        # get info, convert to text string
        # if info is None:
        #     info = ephys.load_recording_info(ddir)
        #self.info_text = info2text(info)
        self.info_text = 'Sorry - no text here :('
        
        # create QTextEdit for displaying file
        self.textwidget = QtWidgets.QTextEdit(self.info_text)
        self.textwidget.setAlignment(QtCore.Qt.AlignCenter)
        self.textwidget.setReadOnly(True)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.textwidget)


def create_filepath_row(txt, base_dir):
    fmt = '<code>{}</code>'
    w = QtWidgets.QWidget()
    vlay = QtWidgets.QVBoxLayout(w)
    row0 = QtWidgets.QHBoxLayout()
    row0.setContentsMargins(0,0,0,0)
    header = QtWidgets.QLabel(fmt.format(txt))
    row0.addWidget(header)
    row0.addStretch()
    row1 = QtWidgets.QHBoxLayout()
    row1.setContentsMargins(0,0,0,0)
    row1.setSpacing(5)
    qlabel = QtWidgets.QLabel(base_dir.format(base_dir))
    qlabel.setStyleSheet('QLabel {background-color:white;'
                         'border:1px solid gray; border-radius:4px; padding:5px;}')
    btn = QtWidgets.QPushButton()
    btn.setIcon(QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
    btn.setMinimumSize(30,30)
    btn.setIconSize(QtCore.QSize(20,20))
    row1.addWidget(qlabel, stretch=2)
    row1.addWidget(btn, stretch=0)
    vlay.addLayout(row0)
    vlay.addLayout(row1)
    return w,header,qlabel,btn
        
#%%
def create_hbox_rows():
    vlay = QtWidgets.QVBoxLayout()
    row0 = QtWidgets.QHBoxLayout()
    row0.setContentsMargins(0,0,0,0)
    row1 = QtWidgets.QHBoxLayout()
    row1.setContentsMargins(0,0,0,0)
    row1.setSpacing(5)
    vlay.addLayout(row0)
    vlay.addLayout(row1)
    return vlay, row0, row1


def folder_btns():
    btn_list = []
    for i in range(3):
        btn = QtWidgets.QPushButton()
        btn.setIcon(QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        btn.setMinimumSize(30,30)
        btn.setIconSize(QtCore.QSize(20,20))
        btn_list.append(btn)
    return btn_list


class thing(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        qrect = pyfx.ScreenRect(perc_height=0.5, perc_width=0.3, keep_aspect=False)
        self.setGeometry(qrect)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        
        self.BASE_FOLDERS = [str(x) for x in ephys.base_dirs()]
        qlabel_ss = ('QLabel {background-color:white;'
                             'border:1px solid gray; border-radius:4px; padding:5px;}')
        fmt = '<code>{}</code>'
        self.btn_list = folder_btns()
        
        ###   RAW BASE FOLDER
        self.raw_w = QtWidgets.QWidget()
        raw_vlay, raw_row0, raw_row1 = create_hbox_rows()
        self.raw_w.setLayout(raw_vlay)
        raw_header = QtWidgets.QLabel(fmt.format('RAW_DATA'))
        raw_row0.addWidget(raw_header)
        raw_row0.addStretch()
        self.raw_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[0]))
        self.raw_qlabel.setStyleSheet(qlabel_ss)
        self.raw_btn = self.btn_list[0]
        raw_row1.addWidget(self.raw_qlabel, stretch=2)
        raw_row1.addWidget(self.raw_btn, stretch=0)
        layout.addWidget(self.raw_w)
        
        ###   PROCESSED BASE FOLDER
        self.processed_w = QtWidgets.QWidget()
        processed_vlay, processed_row0, processed_row1 = create_hbox_rows()
        self.processed_w.setLayout(processed_vlay)
        processed_header = QtWidgets.QLabel(fmt.format('PROCESSED_DATA'))
        processed_row0.addWidget(processed_header)
        processed_row0.addStretch()
        self.processed_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[1]))
        self.processed_qlabel.setStyleSheet(qlabel_ss)
        self.processed_btn = self.btn_list[1]
        processed_row1.addWidget(self.processed_qlabel, stretch=2)
        processed_row1.addWidget(self.processed_btn, stretch=0)
        layout.addWidget(self.processed_w)
        
        ###   PROBE BASE FOLDER
        self.probe_w = QtWidgets.QWidget()
        probe_vlay, probe_row0, probe_row1 = create_hbox_rows()
        self.probe_w.setLayout(probe_vlay)
        probe_header = QtWidgets.QLabel(fmt.format('PROBE_DATA'))
        probe_row0.addWidget(probe_header)
        probe_row0.addStretch()
        self.probe_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[2]))
        self.probe_qlabel.setStyleSheet(qlabel_ss)
        self.probe_btn = self.btn_list[2]
        probe_row1.addWidget(self.probe_qlabel, stretch=2)
        probe_row1.addWidget(self.probe_btn, stretch=0)
        layout.addWidget(self.probe_w)
        
        self.raw_btn.clicked.connect(lambda: self.choose_base_ddir(0))
        self.processed_btn.clicked.connect(lambda: self.choose_base_ddir(1))
        self.probe_btn.clicked.connect(lambda: self.choose_base_ddir(2))
        
        bbox = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton('Save')
        self.save_btn.setStyleSheet('QPushButton {padding : 10px;}')
        bbox.addWidget(self.save_btn)
        layout.addLayout(bbox)
        
        self.save_btn.clicked.connect(self.save_base_folders)
        # rows = [(*x, *create_filepath_row(*x)) for x in ephys.base_dirs(keys=True)]
        
        # for i,(k,p,w,_,_,btn) in enumerate(rows):
        #     layout.addWidget(w)
            
        #     btn.clicked.connect(lambda: self.choose_base_ddir(i,p))
        #     self.BASE_FOLDERS.append(str(p))
            
        # get base folder widgets, put code tags around font
        
        
        # fx2: x -> p,fx(*x) -> (p,w,_,_,z)
        # fx: (k,v) -> (<k>,<v>) -> func() -> (w,x,y,z)
        #fx2 = lambda x: (p, fx(k,p))
  
    def choose_base_ddir(self, i):
        init_ddir = str(self.BASE_FOLDERS[i])
        fmt = 'Base folder for %s'
        titles = [fmt % x for x in ['raw data', 'processed data', 'probe files']]
        # when activated, initialize at ddir and save new base folder at index i
        dlg = FileDialog(init_ddir=init_ddir, parent=self)
        dlg.setWindowTitle(titles[i])
        print(i)
        res = dlg.exec()
        if res:
            print('success!')
            self.BASE_FOLDERS[i] = str(dlg.directory().path())
            self.update_base_ddir(i)
            
    
    def update_base_ddir(self, i):
        fmt = '<code>{}</code>'
        if i==0:
            self.raw_qlabel.setText(fmt.format(self.BASE_FOLDERS[0]))
        elif i==1:
            self.processed_qlabel.setText(fmt.format(self.BASE_FOLDERS[1]))
        elif i==2:
            self.probe_qlabel.setText(fmt.format(self.BASE_FOLDERS[2]))
    
    
    def save_base_folders(self):
        ddir_list = list(self.BASE_FOLDERS)
        ephys.write_base_dirs(ddir_list)
        
        msgbox = MsgboxSave(parent=self)
        res = msgbox.exec()
        if res == QtWidgets.QMessageBox.Yes:
            self.accept()
        
        
        
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    
    
    #ddir = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/NN_JG008')
    #popup = InfoView(ddir=ddir)
    
    nn_raw = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/raw_data/'
                'JG008_071124n1_neuronexus')
    #popup = RawDirectorySelectionPopup()
    #popup = ProbeFileSimple()
    #popup = thing()
    popup = AuxDialog(n=6)
    
    popup.show()
    popup.raise_()
    
    sys.exit(app.exec())
    
