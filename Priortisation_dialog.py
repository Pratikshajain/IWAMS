# -*- coding: utf-8 -*-
"""
Created on Wed Sep 04 17:47:32 2019

@author: Pratiksha
"""

import os

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from PyQt5.QtWidgets import QFileDialog

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'Priortisation_dialog_base.ui'))

class Priority_Run:

    def Priority_Run(self,input_folder,output_folder):
        cwd = os.path.dirname(__file__)
        os.system("python3 "+cwd+"/Priortization_Module/pr_main.py " + str(input_folder) +" "+ str(output_folder))



class PriortisationDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(PriortisationDialog, self).__init__(parent)
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)

        self.input_folder = None
        self.output_folder = None

        self.PB_PR_1.clicked.connect(self.get_input_folder)
        self.PB_PR_2.clicked.connect(self.get_output_folder)
        self.PB_PR_MR.clicked.connect(self.PR_fn)


    def get_input_folder(self):
        self.input_folder = QFileDialog.getExistingDirectory()
        self.LE_PR_1.setText(self.input_folder)

    def get_output_folder(self):
        self.output_folder = QFileDialog.getExistingDirectory()
        self.LE_PR_2.setText(self.output_folder)

    def PR_fn(self):
        input_folder = self.input_folder
        output_folder = self.output_folder

        self.ero=Priority_Run()
        self.ero.Priority_Run(input_folder,output_folder)
