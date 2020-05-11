# -*- coding: utf-8 -*-
"""
Created on Wed Sep 04 17:51:49 2019

@author: Pratiksha
"""

import os

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from PyQt5.QtWidgets import QFileDialog

#importing necessary modules
global xlrd, csv, np, traceback, gdal, itemgetter, os, sys
global import_array, stream_order_from_flow_acc, excel_to_csv, para_array, rank_from_cm_id, cm_id_from_rank, cm_name_from_rank, stream_filter, raster_buffer, export_array
import sys
sys.path.append('')
from osgeo import gdal
import numpy as np, xlrd, csv, traceback
from operator import itemgetter

import time
start_time=time.time()
import warnings
warnings.filterwarnings("ignore")

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'Conservation_dialog_base.ui'))

class Conservation_Run:

    def Conservation_Run(self,rainfall_file,slop_file,flow_acc_file,soil_texture_file,land_use_file,aquifer_file,boundry_file,paremeter_file,output_folder,choose_measure,suitable_thresh,progress_bar):
        
        class Error1(Exception):
           """This is a custom exception."""
           pass
        
        def excel_to_csv(ExcelFile):
            workbook = xlrd.open_workbook(ExcelFile)
            sheet_names=workbook.sheet_names()
            for sheet in sheet_names[0:]:
                worksheet = workbook.sheet_by_name(sheet)
                csvfile = open(output_folder+"/"+sheet+".csv", 'w')
                wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                for rownum in range(worksheet.nrows):
                    wr.writerow(
                        list(x.encode('utf-8') if type(x) == type(u'') else x
                            for x in worksheet.row_values(rownum)))
                csvfile.close()
        
                
        def para_array(para_name,in_array,rank):
                """This class is used to create weighted arrays to represent the parameters like slope, rainfall, etc"""
                cm_id=cm_id_from_rank(rank)
                cm_name=cm_name_from_rank(rank)
                para_array=np.zeros_like(in_array).astype(float)
                with open(os.path.join(output_folder,para_name+".csv"),'r') as f:
                   reader = list(csv.reader(f))
                   for row in reader[1:]:
                       para_array=np.where(in_array>float(row[2]),row[cm_id+3],para_array)
                print(para_name + " para_array array created for " + cm_name)
                return para_array.astype(float)
                
        def import_array(filepath):
            """This function is used to import input raster map as an array."""
            d1=gdal.Open(filepath)
            if d1 is None:
                print("Error: Could not open image: " + filepath)
                return None
            inband=d1.GetRasterBand(1)
            in_array=np.zeros_like(array_ref)
            in_array= inband.ReadAsArray(0,0,col_ref,row_ref)
            in_array=in_array.astype(float)
            in_array[in_array == (inband.GetNoDataValue() or 0.0 or -999)]=np.nan
            return in_array
        
        def export_array(in_array,output_filepath):
            """This function is used to produce output of array as a map."""
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(output_filepath,col_ref,row_ref,1,gdal.GDT_Float32)
            outband=outdata.GetRasterBand(1)  
            outband.SetNoDataValue(np.nan)
            outband.WriteArray(in_array)
            outdata.SetGeoTransform(geotrans)
            outdata.SetProjection(proj)       
            outdata.FlushCache()
            
        def cm_name_from_rank(rank):
            if(rank==0):
                return "All"
            with open(os.path.join(output_folder,"ranking.csv"),'r') as f:
                reader = csv.reader(f)
                reader=list(reader)
                for row in reader[1:]:
                    if(int(float(row[8]))==rank):
                        return row[1]
        
        def rank_from_cm_id(cm_id):
            if(cm_id==0):
                return 0
            with open(os.path.join(output_folder,"ranking.csv"),'r') as f:
                reader=csv.reader(f)
                reader=list(reader)
                return int(reader[cm_id+1][8])
                        
                    
        def cm_id_from_rank(rank):
            if(rank==0):
                return 0
            with open(os.path.join(output_folder,"ranking.csv"),'r') as f:
                reader=csv.reader(f)
                reader=list(reader)
                for row in reader[1:]:
                    if(int(float(row[8]))==rank):
                        return int(float(row[0]))
                    
        def rank_from_cm_name(cm_name):
            if(cm_name=="All"):
                return 0
            with open(os.path.join(output_folder,"ranking.csv"),'r') as f:
                reader = csv.reader(f)
                reader=list(reader)
                for row in reader[1:]:
                    if(row[1]==cm_name):
                        return int(float(row[8]))
            
                    
        def stream_order_from_flow_acc(flow_acc_filepath):
            """This function creates stream order map from flow accumulation map using thresholds derived from main_table.xls's sheet named 'stream_order_def"""
            d1=gdal.Open(flow_acc_filepath)
            if d1 is None:
                print("Error: Could not open image: " + flow_acc_filepath)
                return None
            inband=d1.GetRasterBand(1)
            trans=d1.GetGeoTransform()
            Xcell_size=int(abs(trans[1]))
            Ycell_size=int(abs(trans[5]))
            global stream_cell_size
            stream_cell_size = (Xcell_size+Ycell_size)/2
            in_array= inband.ReadAsArray(0,0,col_ref,row_ref)
            in_array=in_array.astype(float)
            in_array[in_array == (inband.GetNoDataValue() or 0.0 or -999)]=np.nan
            in_array=np.absolute(in_array) #flow_acc is sometimes negative also.
            with open(os.path.join(output_folder,"stream_order_def.csv")) as f:
                reader=csv.reader(f)
                reader=list(reader)
                out_array=np.zeros_like(array_ref)
                out_array.fill(np.nan)
                for row in reader[1:]:
                    flow_acc_threshold = float(row[1])/(Xcell_size*Ycell_size)
                    out_array=np.where(in_array>=flow_acc_threshold,row[0],out_array)
            return out_array.astype(float)
                  
        def raster_buffer(raster_filepath,dist=1000):
            """This function creates a distance buffer around the given raster file with non-zero values.
            The value in output raster will have value of the cell to which it is close to."""
            print(raster_filepath)
            d=gdal.Open(raster_filepath)
            if d is None:
                print("Error: Could not open image " + raster_filepath)
                return None
            row=d.RasterYSize
            col=d.RasterXSize
            inband=d.GetRasterBand(1)
            cell_dist=dist/stream_cell_size
            in_array = inband.ReadAsArray(0,0,col,row).astype(float)
            in_array[in_array == (inband.GetNoDataValue() or 0 or -999)]=0
            in_array=import_array(raster_filepath)
            out_array=np.zeros_like(in_array)
            out_array.fill(np.nan)
            temp_array=np.zeros_like(in_array)
            print("temp array is ........................................")
            # for v in range(len(temp_array)):
            #     print(temp_array[v])
            # print("in_Arry is ........................")
            # print(in_array)
            h,k,i,j=0,0,0,0
            while(h<col):
                k=0
                # print("h is ..................")
                # print(h)
                while(k<row):
                    # print("k is ......") 
                    # print(k)
                    if(in_array[k][h]>0):
                        i=h-cell_dist
                        if(i<0):
                            i=0
                        while((i<=cell_dist+h) and i<col):
                            # print("i is ..............")
                            # print(i)
                            j=k-cell_dist
                            if(j<0):
                                j=0
                            while(j<=(cell_dist+k) and j<row):
                                print("j is  ......................")
        
                                if(((i-h)**2+(j-k)**2)<=cell_dist**2):
                                    print(i,j)
                                    # print(temp_array[j][i])
                                    i = int(i)
                                    j = int(j)
                                    if(temp_array[j][i]==0 or temp_array[j][i]>(((i-h)*(i-h))+((j-k)*(j-k)))):
                                        out_array[j][i]= in_array[k][h]
                                        temp_array[j][i]=(i-h)**2+(j-k)**2
                                j+=1
                            i+=1
                    k+=1
                h+=1
            return out_array.astype(float)
        
        def stream_filter(wtage_array, stream_order_array,dist=1000):
            cell_dist=dist/stream_cell_size
            wtage_array[np.isnan(wtage_array)]=0
            out_array=np.zeros_like(stream_order_array)
            out_array.fill(np.nan)
            i,j,h,k,n,s=0,0,0,0,0,0
            while(h<col_ref):
                k=0
                while(k<row_ref): 
                    if(stream_order_array[k][h]>=1):
                        i=h-cell_dist
                        if(i<0):
                            i=0
                        n,s=0,0
                        while((i<=(cell_dist+h)) and i<col_ref):
                            j=k-cell_dist
                            if(j<0):
                                j=0
                            while(j<=(cell_dist+k) and j<row_ref):
                                if(((i-h)**2+(j-k)**2)<=cell_dist**2):
                                    n+=1
                                    j = int(j)
                                    i = int(i)
                                    s=s+wtage_array[j][i]
                                j+=1
                            i+=1
                        out_array[k][h]=s/n
                    k+=1
                h+=1
            return out_array
        
        def raster_read(filename):
            d1= gdal.Open(filename)
            row1=d1.RasterYSize
            column1=d1.RasterXSize
            data= d1.ReadAsArray(0,0,column1,row1)
            transform = d1.GetGeoTransform()
            pixelWidth = transform[1]
            return (data,pixelWidth,row1,column1)
        
        def csv_to_data(filename):
            with open(filename,'r') as f:
                reader=csv.reader(f)
                A=list(reader)
                A=np.asarray([row[2:] for row in A[1:]])
                A=A.astype(float)
            return A
        
        try:
            print("\nProgram starting... \n")
            
            #Taking inputs
            # global input_folder,output_folder,data_folder
            # input_folder=Input_CM+"\\"
            # output_folder=Output_CM+"\\"
            suitability_threshold=float(suitable_thresh)
            cm_name=choose_measure
            print(suitability_threshold,cm_name)
            progress_bar=progress_bar
        
            #assigning filepaths
            #data_folder=os.path.dirname(os.path.abspath(__file__))+"\\Conservation_Measures\\data\\"
            #data_folder="C:\\Users\\Pratiksha\\.qgis2\\python\\plugins\\iwams\\Conservation_Module\\data\\"
            
            rainfall_filepath=rainfall_file
            slope_filepath=slop_file
            flow_acc_filepath=flow_acc_file
            soil_texture_filepath=soil_texture_file
            lulc_filepath=land_use_file
            reference_filepath=flow_acc_filepath
            aquifer_filepath=aquifer_file
            boundary_filepath = boundry_file
            main_table_filepath=paremeter_file
            output_filepath_final=os.path.join(output_folder,"final_suitability_map.tif")
        
            
            
            (flowacc_array, s2,r2,c2)= raster_read(flow_acc_filepath)
            (lulc_array,s4,r4,c4)= raster_read(lulc_filepath)
            (soil_texture_array,s5,r5,c5)=raster_read(soil_texture_filepath)
            (slope_array,s6,r6,c6)=raster_read(slope_filepath)
            (taluka,s7,r7,c7) = raster_read(boundry_file)
            
            excel_to_csv(rainfall_filepath)
            rainfall_data= csv_to_data(os.path.join(output_folder,"Annual.csv"))
            rainfall_array=taluka
            for i in range(len(rainfall_data)):
                rainfall_array[taluka == i+1] = float(rainfall_data[i][0])    
            
            #Conversion of data/main_table.xls excel files into usable csv files
            print("Generating csv files...")
            excel_to_csv(main_table_filepath)
            print("All excel files generated")
            
            
            
            
            #Creating log_file.csv which can be used by user to interpret the output map values
            
            log_file=open(os.path.join(output_folder,"log_file.csv"),'w+')
            log_file.write("Time,"+time.strftime("%H:%M:%S")+"\n")
            log_file.write("Date,"+time.strftime("%d/%m/%Y")+"\n\n")
            log_file.write("Value, Conservation structure\n")
            with open(os.path.join(output_folder,"ranking.csv"),'r') as f:
                reader = csv.reader(f)
                reader=list(reader)
                del reader[0]
                for row in reader:
                    row[8]=int(float(row[8]))
                reader=sorted(reader, key=itemgetter(8))
                for row in reader:
                    log_file.write(str(row[8])+','+str(row[1])+"\n")
            
            #Checking if the inputs are correct
            cm_rank=rank_from_cm_name(cm_name)
            max_rank=len(reader)-1
            if((cm_rank>max_rank) or (cm_rank<0)):   
                print("Wrong input for CM_ID. Give in the range as specified.")
                raise Error1     
            if(suitability_threshold > 1 or suitability_threshold<0.5):
                print("Wrong value for suitability threshold. Enter a value between 0.5 and 1.")
                raise Error1
            if(not os.path.exists(output_folder)):
                os.mkdir(output_folder)
            
            #Some map is loaded as a reference for rest of the arrays/maps
            d=gdal.Open(reference_filepath)
            if d is None:
                print("Error: Could not open image " + reference_filepath)
                raise Error1
            global proj,geotrans,row_ref,col_ref,array_ref
            inband=d.GetRasterBand(1)
            proj=d.GetProjection()
            geotrans=d.GetGeoTransform()
            row_ref=d.RasterYSize
            col_ref=d.RasterXSize
            array_ref = inband.ReadAsArray(0,0,col_ref,row_ref).astype(float)
            array_ref[array_ref == (inband.GetNoDataValue() or 0.0 or -999)]=np.nan
            d,inband=None,None
            progress_bar.setValue(5)
        
            #Creating stream_order map from flow_accumulation map
            print("Creating stream order map from flow accumulation map...")
            stream_order_array=stream_order_from_flow_acc(flow_acc_filepath)
            export_array(stream_order_array,os.path.join(output_folder,"stream_order.tif"))
            print("Stream_order map created.")
            progress_bar.setValue(10)
            
            #Creating distance buffer map around the streams.
            print("Creating distance buffer around streams...")
            stream_order_buffer_array=raster_buffer(os.path.join(output_folder,"stream_order.tif"))
            export_array(stream_order_buffer_array,os.path.join(output_folder,"stream_order_buffer.tif"))
            print("Buffer map created around the streams. \n")
            
            #Reading other input maps as arrays
            print("Reading input maps...")
            #rainfall_array=import_array(rainfall_filepath)
            #slope_array=import_array(slope_filepath)
            #soil_texture_array=import_array(soil_texture_filepath)
            #lulc_array=import_array(lulc_filepath)
            
            
            
            
            
            if os.path.exists(aquifer_filepath):
                aquifer_array=import_array(aquifer_filepath)
            print("Input maps read.")
            progress_bar.setValue(15)
            #Generation of suitability arrays.
            #Here, index in wtage array represents rank of conservation measure
            wtage_array=[]
            for i in range(max_rank+1):               
                wtage_array.append(np.nan)
            #If user has specified a single conservation measure, the following code will calculate the para_array,wtage_array arrays for that measure only.
            #If user has selected all conservation measure, the following code will calculate para_arrays,wtage_array for all conservation measures.
            for i in range(max_rank+1)[1:]:
                if (i!=cm_rank) and (cm_rank!=0):  
                    continue
                
                rainfall = para_array("rainfall",rainfall_array,i)
                slope= para_array("slope",slope_array,i)
                soil_texture = para_array("soil_texture",soil_texture_array,i)
                lulc= para_array("lulc",lulc_array,i)
                stream_order_buffer = para_array("stream_order",stream_order_buffer_array,i)
                progress_bar.setValue(progress_bar.value()+3)
               
                #Creation of combined map array
                cm_name=cm_name_from_rank(i)
                cm_id=cm_id_from_rank(i)
                wtage_array[i] = np.zeros_like(array_ref,dtype=float)
                wtage_array[i].fill(np.nan)
                with open(os.path.join(output_folder,"parameter_weightage.csv")) as wt:                
                    reader=csv.reader(wt)
                    reader=list(reader)
                rainfall_wt=float(reader[1][cm_id])
                slope_wt=float(reader[2][cm_id])
                soil_texture_wt=float(reader[3][cm_id])
                lulc_wt=float(reader[4][cm_id])
                stream_order_wt=float(reader[5][cm_id])
                wtage_array[i]=(rainfall_wt*rainfall+slope_wt*slope+soil_texture_wt*soil_texture+lulc_wt*lulc+stream_order_wt*stream_order_buffer)
            
                #Effect of type of aquifer
                if(os.path.exists(aquifer_filepath)):
                    aquifer=para_array("aquifer",aquifer_array,i)
                    wtage_array[i]=wtage_array[i]*aquifer
                
                #Placing hydraulic structures over stream sections.
                with open(os.path.join(output_folder,"location.csv")) as f:
                    reader=csv.reader(f)
                    reader=list(reader)
                    if(float(reader[1][cm_id])==1):
                        print("Placing "+cm_name+" in stream section...")
                        wtage_array[i]=stream_filter(wtage_array[i], stream_order_array,1000)   
                        print(cm_name+" placed in stream section.")
                    elif(float(reader[1][cm_id])==0):
                        print("Removing "+cm_name+" from streams...")
                        wtage_array[i]=np.where(stream_order_array>=1,0,wtage_array[i])
                        print(cm_name+" removed from streams.")
                wtage_array[i][np.isnan(array_ref)]=np.nan        
                print("Weightage array calculated for "+cm_name)
                #export the output suitability array as a tif map
                output_filepath=os.path.join(output_folder,"Suitability_map for " + cm_name+".tif")
                export_array(wtage_array[i],output_filepath)
                print("Suitability map created for "+cm_name + "\n")
                progress_bar.setValue(progress_bar.value()+2)
                
            #Creation of final_array combining all conservation measures
            progress_bar.setValue(95)
            print("\nCreating final map...")
            final_array = np.zeros_like(array_ref,dtype=float)
            final_array.fill(np.nan)
            r=max_rank
            while r>0:
                final_array=np.where(wtage_array[r]>=suitability_threshold,r,final_array)
                r-=1
            #export the output suitability array as a tif map
            export_array(final_array,output_filepath_final)
            print("Final suitability map created with suitability threshold at "+str(suitability_threshold))
            log_file.write("Status: Success\n")
            progress_bar.setValue(100)
            
        except Error1:
            print("\nThis is error due to wrong input. Program will now exit.")
            #log_file.write("Status: Failed.\nTraceback: "+"Wrong input\n")    
            #progress_bar.setValue(0)
        
        except:
            print("\nUnexpected error:")
            traceback.print_exc(file=sys.stdout)
            #log_file.write("Status: Failed.\nTraceback: "+str(traceback.print_exc(file=sys.stdout))+"\n")
            #progress_bar.setValue(0)
            
        finally:
            #Deletion of temporary files
            # workbook = xlrd.open_workbook(paremeter_file)
            # sheet_names=workbook.sheet_names()
            # for sheet in sheet_names[0:]:
            #     try:
            #         os.remove(output_folder+"/"+sheet+".csv")
            #     except:
            #         continue
            # workbook = xlrd.open_workbook(rainfall_file)
            # sheet_names=workbook.sheet_names()
            # for sheet in sheet_names[0:]:
            #     try:
            #         os.remove(output_folder+"/"+sheet+".csv")
            #     except:
            #         continue
            print("\nTime elapsed: " + str(time.time()-start_time))
                
            log_file.write("Time elapsed: " + str(time.time()-start_time)+"\n\n")
            log_file.close()
        
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
        progress_bar=self.PG_CM
        self.ero = Conservation_Run()
        self.ero.Conservation_Run(rainfall_file,slop_file,flow_acc_file,soil_texture_file,land_use_file,aquifer_file,boundry_file,paremeter_file,output_folder,choose_measure,suitable_thresh,progress_bar)








