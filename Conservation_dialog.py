# -*- coding: utf-8 -*-
"""
Created on Wed Sep 04 17:51:49 2019

@author: Pratiksha
"""

import os

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from PyQt5.QtWidgets import QFileDialog

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'Conservation_dialog_base.ui'))

class Conservation_Run:

    def Conservation_Run(self,rainfall_file,slop_file,flow_acc_file,soil_texture_file,land_use_file,aquifer_file,boundry_file,paremeter_file,output_folder,choose_measure,suitable_thresh):
        
        cwd = os.path.dirname(__file__)

        os.system("python3 "+cwd+"/Conservation_Module/cm_main.py " + str(rainfall_file)+" "+str(slop_file)+" "+str(flow_acc_file)+" "+str(soil_texture_file)+" "+str(land_use_file)+" "+str(aquifer_file)+" "+str(boundry_file)+" "+str(paremeter_file)+" "+str(output_folder)+" "+str(choose_measure)+" "+str(suitable_thresh))

        
class ConservationDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(ConservationDialog, self).__init__(parent)
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)
        self.rainfall_file = None
        self.slop_file = None
        self.flow_acc_file = None
        self.soil_texture_file = None
        self.land_use_file = None
        self.aquifer_file = None
        self.boundry_file = None
        self.paremeter_file = None
        self.output_folder = None

        self.PB_CM_1.clicked.connect(self.rainfall)
        self.PB_CM_2.clicked.connect(self.slop)
        self.PB_CM_3.clicked.connect(self.flow_acc)
        self.PB_CM_4.clicked.connect(self.soil_texture)
        self.PB_CM_5.clicked.connect(self.land_use)
        self.PB_CM_6.clicked.connect(self.aquifer)
        self.PB_CM_7.clicked.connect(self.boundry)
        self.PB_CM_8.clicked.connect(self.paremeter)
        self.PB_CM_9.clicked.connect(self.output_folder_call)
        self.MR_CM.clicked.connect(self.Con_fn)


    def rainfall(self):
        self.rainfall_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CM_1.setText(self.rainfall_file)

    def slop(self):
        self.slop_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CM_2.setText(self.slop_file)

    def flow_acc(self):
        self.flow_acc_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CM_3.setText(self.flow_acc_file)

    def soil_texture(self):
        self.soil_texture_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CM_4.setText(self.soil_texture_file)

    def land_use(self):
        self.land_use_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CM_5.setText(self.land_use_file)

    def aquifer(self):
        self.aquifer_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CM_6.setText(self.aquifer_file)

    def boundry(self):
        self.boundry_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CM_7.setText(self.boundry_file)

    def paremeter(self):
        self.paremeter_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CM_8.setText(self.paremeter_file)

    def output_folder_call(self):
        self.output_folder = QFileDialog.getExistingDirectory()
        self.LE_CM_9.setText(self.output_folder)

    def Con_fn(self):
        rainfall_file = self.rainfall_file
        slop_file = self.slop_file
        flow_acc_file = self.flow_acc_file
        soil_texture_file = self.soil_texture_file
        land_use_file = self.land_use_file
        aquifer_file = self.aquifer_file
        boundry_file = self.boundry_file
        paremeter_file = self.paremeter_file
        output_folder = self.output_folder
        choose_measure = self.CB_CM.currentText()
        suitable_thresh = self.LE_CM_10.text()

        self.ero = Conservation_Run()
        self.ero.Conservation_Run(rainfall_file,slop_file,flow_acc_file,soil_texture_file,land_use_file,aquifer_file,boundry_file,paremeter_file,output_folder,choose_measure,suitable_thresh)








