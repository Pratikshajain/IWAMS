# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:29:58 2017

@author: Admin
"""
# Input_PR="C:\Users\Pratiksha\.qgis2\python\plugins\iwams\Priortization_Module\Input_PR"
# Output_PR= "C:\Users\Pratiksha\.qgis2\python\plugins\iwams\Priortization_Module\Output_PR"
#importing necessary modules
global xlrd, csv, np, traceback, gdal, itemgetter, os, sys
global import_array, stream_order_from_flow_acc, excel_to_csv, para_array, rank_from_cm_id, cm_id_from_rank, cm_name_from_rank, stream_filter, raster_buffer, export_array
import sys
import os
sys.path.append('')

import numpy as np, os, xlrd, csv, traceback


import time
start_time=time.time()
import warnings
warnings.filterwarnings("ignore")

input_folder = sys.argv[1]
output_folder = sys.argv[2]

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
        
    
def Consistancy_check(main_table_filepath):
    with open(os.path.join(output_folder,"AHP_ranking.csv"),'r') as f:
        reader=csv.reader(f)
        A=list(reader)
        A=np.asarray([row[1:] for row in A[1:]])
        A=A.astype(float)
        col_total=[sum(x) for x in zip(*A)]
        #normalize the matrix
        Aw=A/col_total
        #relative weights for each criteria’s
        W=np.mean(Aw, axis=1) 
        #consistency check
        #weight sum vector
        Ws=np.dot(A,W)
        #Consistency vector
        C= Ws/W
        #the average of the elements of {C}.
        lmda=np.mean(C)
        #defined standard random consistency index (RI) for no. of elements,
        RI_list= [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        #no. of criterias
        n=len(A[0])
        #random consistency index 
        RI=RI_list[n-1]
        #Consistency Index
        CI= (lmda-n)/(n-1)
        CR=CI/RI
        print("Consistency ratio is : "+ str(CR))
        if CR>0.1:
            return (9999)
        else:
            return CR, W  
def Topsis_ranking(main_table_filepath,W,Ranking):
        with open(os.path.join(output_folder,"Criteria_values.csv"),'r') as f:
            reader=csv.reader(f)
            #decision matrix (A)
            A=list(reader)
            B=np.asarray([row[0] for row in A[1:]])
            A=np.asarray([row[1:] for row in A[1:]])
            A=A.astype(float)  
            B=B.astype(float)  
            #Square of each element in decision matrix
            A1=np.power(A,2)
            S=[sum(x) for x in zip(*A1)]
            S1=np.power(S,0.5)
            #Standardised Decision Matrix
            A2=A/S1
            #Multiplied AHP weights
            A3=A2*W
            #positive-ideal and negative-ideal solutions.
            A_min=[min(x) for x in zip(*A3)]
            A_max=[max(x) for x in zip(*A3)]
            #DRNK_WAT_F(IS-Min)	GW_MEAN(IS-Max)	Per_Cult_Was(IS-Max)	Per_ST_SC(IS-Max)	Per_UN_IRR(IS-Max)	 Per_Agri(IS-Max)	SE_MEAN((IS-Max))
            #positive ideal solution and negative ideal solution
            IS= np.zeros(len(A_min))
            NIS= np.zeros(len(A_min))
            IS[0]=A_min[0]
            IS[1:]= A_max[1:]
            NIS[0]=A_max[0]
            NIS[1:]=A_min[1:]
            #Seperation from Ideal Solution
            A4=np.power((A3-IS),2)
            A5=np.power((A3-NIS),2)
            S4=[sum(x) for x in A4]
            S4=np.power(S4,0.5)
            S5=[sum(x) for x in A5]
            S5=np.power(S5,0.5)
            
            #relative closeness to the idea solution
            CC=S5/(S4+S5)
            
            R=np.column_stack((B,CC))
            R=sorted(R, key= lambda x: -x[1])
            R=np.asarray(R)
            a= R[:,0]
            b = np.arange(len(CC))+1
            
            f = open(Ranking, "w")
                       
            f.write("{},{}\n".format("Unit", "Ranking"))
            for x in zip(a, b):
                f.write("{},{}\n".format(x[0], x[1]))
            f.close()
            return (CC)
                
try:
    print("\nProgram starting... \n")
    
    #Taking inputs
    # global input_folder,output_folder,data_folder
    # input_folder=Input_PR+"\\"
    # output_folder=Output_PR+"\\"
    #progress_bar=progress_bar
    #assigning filepaths
    main_table_filepath=os.path.join(input_folder,"main_table.xlsx")
    Ranking= os.path.join(output_folder,"Ranking.csv")
    
    #Conversion of data/main_table.xls excel files into usable csv files
    print("Generating csv files...")
    excel_to_csv(main_table_filepath)
    print("All excel files generated")
    
    #Creating log_file.csv which can be used by user to interpret the output map values
    log_file=open(os.path.join(output_folder,"log_file.csv"),'w+')
    log_file.write("Time,"+time.strftime("%H:%M:%S")+"\n")
    log_file.write("Date,"+time.strftime("%d/%m/%Y")+"\n\n")
    
    
    #progress_bar.setValue(30)
    #Checking consistancy of AHP 
    print("Checking consistancy of AHP...")
    [CR,W]= Consistancy_check(main_table_filepath)
    #if W.any()==9999:
    if CR==9999:
        print ("inconsistent AHP matrix")
        print ("restart with new AHP ranking")
        raise Error1
    print ("Consistent AHP  matrix")
    log_file.write("Consistent AHP  matrix\n")
    log_file.write("Consistency Ratio is ")
    log_file.write(str(CR))
    log_file.write("\nAHP weight matrix is ")
    log_file.write(str(W))
    #progress_bar.setValue(60)
    #Implementing Topsis algorithm
    print ("Implementing Topsis algorithm")
    Topsis_ranking(main_table_filepath,W,Ranking)
    print ("Finished...")
    print ("Ranking saved in output folder")
    log_file.write("\nRanking saved in output folder\n")
    log_file.write("Status: Success\n")
    #progress_bar.setValue(100)
except Error1:
    print("\nThis is error due to wrong input. Program will now exit.")
    log_file.write("Status: Failed.\nTraceback: "+"Wrong input\n")    
    #progress_bar.setValue(0)
    
except:
    print("\nUnexpected error:")
    traceback.print_exc(file=sys.stdout)
    log_file.write("Status: Failed.\nTraceback: "+str(traceback.print_exc(file=sys.stdout))+"\n")
    #progress_bar.setValue(0)
finally:
    #Deletion of temporary files
    workbook = xlrd.open_workbook(main_table_filepath)
    sheet_names=workbook.sheet_names()
    for sheet in sheet_names[0:]:
        try:
            os.remove(input_folder+sheet+".csv")
        except:
            continue
    print("Time elapsed: " + str(time.time()-start_time))
    log_file.write("Time elapsed: " + str(time.time()-start_time)+"\n\n")
    log_file.close()