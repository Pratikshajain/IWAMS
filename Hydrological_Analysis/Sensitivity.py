# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 15:58:39 2019

@author: Pratiksha
"""


# Input_SR_R ="C:\Users\Pratiksha\.qgis2\python\plugins\iwams\Hydrological_Module\Inputs"
# Output_SR_R= "C:\Users\Pratiksha\.qgis2\python\plugins\iwams\Hydrological_Module\Outputs"
# Simulation_year_SR_R= "2012"
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
global xlrd, csv, np, traceback, gdal, itemgetter, os, sys
global import_array, stream_order_from_flow_acc, excel_to_csv, para_array, rank_from_cm_id, cm_id_from_rank, cm_name_from_rank, stream_filter, raster_buffer, export_array
global raster_read, excel_to_csv, csv_to_data, move_along_flow_dir, export_array
import sys
sys.path.append('')
from osgeo import gdal
import numpy as np, os, xlrd, csv, traceback
from operator import itemgetter

import time
start_time=time.time()
import warnings
warnings.filterwarnings("ignore")

print("args .........................................................")
Simulation_Year = sys.argv[1]
print(Simulation_Year)
dem_file = sys.argv[2]
print(dem_file)
slop_file = sys.argv[3]
print(slop_file)
flow_direction_file = sys.argv[4]
print(flow_direction_file)
flow_accumulation_file = sys.argv[5]
print(flow_direction_file)
land_use_file = sys.argv[6]
print(land_use_file)
soil_texture_file = sys.argv[7]
print(soil_texture_file)
rainfall_data_file = sys.argv[8]
print(rainfall_data_file)
boundry_file = sys.argv[9]
print(boundry_file)
paremeter_file = sys.argv[10]
print(paremeter_file)
output_folder = sys.argv[11]
print(output_folder)
chek_box_arr = sys.argv[12]
print(chek_box_arr)

class Error1(Exception):
   """This is a custom exception."""
   pass




def raster_read(filename):
    d1= gdal.Open(filename)
    row1=d1.RasterYSize
    column1=d1.RasterXSize
    data= d1.ReadAsArray(0,0,column1,row1)
    transform = d1.GetGeoTransform()
    pixelWidth = transform[1]
    return (data,pixelWidth,row1,column1)

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

def csv_to_data(filename):
    with open(filename,'r') as f:
        reader=csv.reader(f)
        A=list(reader)
        A=np.asarray([row[2:] for row in A[1:]])
        A=A.astype(float)
    return A
 
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

def move_along_flow_dir(i,j,flowdir):
    if flowdir == 1:
        i2 = i
        j2 = j + 1
        
    elif flowdir == 2:
        i2 = i + 1;
        j2 = j + 1;
        
    elif flowdir == 4:
        i2 = i + 1;
        j2 = j;
        
    elif flowdir == 8:
        i2 = i + 1;
        j2 = j - 1;
        
    elif flowdir == 16:
        i2 = i;
        j2 = j - 1;
        
    elif flowdir == 32:
        i2 = i - 1;
        j2 = j - 1;
        
    elif flowdir == 64:
        i2 = i - 1;
        j2 = j;
        
    elif flowdir == 128:
        i2 = i - 1;
        j2 = j + 1;
    return i2,j2






   
try:
    print("\nProgram starting... \n")
    #print (Input_SR_R)
    #Taking inputs
    # data_folder="C:\Users\\Pratiksha\\.qgis2\\python\\plugins\\iwams\\Hydrological_Module\\Data\\"
    # main_table_filepath=data_folder+"Parameter_file.xlsx"
    main_table_filepath = paremeter_file
    # global input_folder,output_folder
    # #global data_folder
    # input_folder=Input_SR_R+"\\"
    # output_folder=Output_SR_R+"\\"
    # Y=str(Simulation_year_SR_R)
    #data_folder=os.path.dirname(os.path.abspath(__file__))+"\\python3 MMMF\\Data\\"
    #data_folder="C:\\Users\\Pratiksha\\AppData\\Roaming\\QGIS\\QGIS3\\profiles\\default\\python\\plugins\\iwams\\Hydrological_Analysis\\Data\\"
    #main_table_filepath=data_folder+"Parameter_file.xlsx"
    log_file=open(os.path.join(output_folder,"log_file.csv"),'w+')
    log_file.write(output_folder)
    log_file.write("\nTime,"+time.strftime("%H:%M:%S")+"\n")
    log_file.write("Date,"+time.strftime("%d/%m/%Y")+"\n\n")
    #progress_bar=progress_bar
    #reading raster tiff files as array
    print("Reading Input tiff files as array...")
    log_file.write("Reading Input tiff files as array...")
    (dem, s1,r1,c1)=raster_read(dem_file)
    (flowacc, s2,r2,c2)= raster_read(flow_accumulation_file)
    (flowdir, s3,r3,c3)= raster_read(flow_direction_file)
    (lulc,s4,r4,c4)= raster_read(land_use_file) #land used
    (Soil_texture,s5,r5,c5)=raster_read(soil_texture_file)
    (Slope,s6,r6,c6)=raster_read(slop_file)
    (taluka,s7,r7,c7) = raster_read(boundry_file)#boundry
    log_file.write("\nRead all")
    s=[s1,s2,s3,s4,s5,s6,s7]
    r=[r1,r2,r3,r4,r5,r6,r7]
    c=[c1,c2,c3,c4,c5,c6,c7]
    sp=s1
    #Conversion of data/main_table.xls excel files into usable csv files
    print("Generating csv files...")
    excel_to_csv(main_table_filepath)
    excel_to_csv(rainfall_data_file)
    print("All csv files generated")
    log_file.write("\nGenerated csv files...\n")
    #progress_bar.setValue(5)    
    #reading txt files as paramereters
    soil_para=csv_to_data(os.path.join(output_folder,"Soil.csv"))#output 
    cover= csv_to_data(os.path.join(output_folder,"Landuse.csv"))
    soil_char= csv_to_data(os.path.join(output_folder,"Soil_char.csv"))
    flow_char=csv_to_data(os.path.join(output_folder,"Flow_char.csv"))
    other_para= csv_to_data(os.path.join(output_folder,"Other.csv"))
    
    #rainfall data
    rainfall_data= csv_to_data(os.path.join(output_folder,"Annual.csv"))
    
    log_file.write("\nRead CSV to data\n")
    
    print("Preparing rainfall layers...")
    
    Rainfall=np.tile(np.nan, dem.shape)
    rn=np.tile(np.nan, dem.shape)
    for i in range(len(rainfall_data)):
        Rainfall[taluka == i+1] = float(rainfall_data[i][0])
        rn[taluka == i+1] = float(rainfall_data[i][1])
        
    
        
    print("Preparing parameter layers...")
    #Define parameters
    #Soil Parameters
    c=np.zeros(dem.shape,dtype='f')
    silt=np.zeros(dem.shape,dtype='f')
    s =np.zeros(dem.shape,dtype='f')
    ms=np.zeros(dem.shape,dtype='f')
    bd=np.zeros(dem.shape,dtype='f')
    lp=np.zeros(dem.shape,dtype='f')
 
    #preparing soil parameter layers
       
    for i in range(1,len(soil_para)):
        c[Soil_texture==i] = float(soil_para[i-1][0])      #percentage clay
        silt[Soil_texture==i] = float(soil_para[i-1][1])        #percentage silt
        s[Soil_texture==i] = float(soil_para[i-1][2])        #percentage sand
        ms[Soil_texture==i] = float(soil_para[i-1][3])        #soil moisture content at field capacity
        bd[Soil_texture==i] = float(soil_para[i-1][4])        #bulk density (mg/m^3)
        lp[Soil_texture==i] = float(soil_para[i-1][5])        #saturated lateral permeability of soil (m/day)
 
    #preparing land-use parameter layers
    ehd = np.zeros(dem.shape,dtype='f')
    per_incp =np.zeros(dem.shape,dtype='f')
    eto = np.zeros(dem.shape,dtype='f')
    cc = np.zeros(dem.shape,dtype='f')
    gc = np.zeros(dem.shape,dtype='f')
    ph = np.zeros(dem.shape,dtype='f')
    nv = np.zeros(dem.shape,dtype='f')
    d = np.zeros(dem.shape,dtype='f')
    st = np.zeros(dem.shape,dtype='f')
    #Other parameters
    intensity = float( other_para[0][0])#Intensity of erosive rain (mm/hr)
    t = float( other_para[1][0])#Mean annual temperature (C)
    #rn = 50; %No. of rainy days %input from user
    thres=float( other_para[2][0])# flowaccumulation threashold
    g = float( other_para[3][0])#Gravitational acceleration
    sp = float( other_para[4][0]) #Cell spacing %input from user
    
    #LU/LC Parameters
    lulc[flowacc>thres]=16

    for i in range(1,len(cover)):
        ehd[lulc==i]  = float(cover[i-1][0])        #effective hidrological depth (m)
        per_incp[lulc==i] = float(cover[i-1][1])    #permanent interception
        eto[lulc==i]  = float(cover[i-1][2])        #ratio of actual to potential evapotranspiration
        cc[lulc==i]  = float(cover[i-1][3])         #canopy cover (cover for rainfall)
        gc[lulc==i]  = float(cover[i-1][4])         #ground cover (cover for runoff)
        ph[lulc==i]  = float(cover[i-1][5])         #plant height (m)
        nv[lulc==i]  = float(cover[i-1][6])         #number of plants per unit area (/m^2)
        d[lulc==i]  = float(cover[i-1][7])  
        st[lulc==i] = float(cover[i-1][8])
    #Soil Characteristics
    kc =float( soil_char[0][0])#Detachability of soil by raindrop for clay (g/J)
    kz = float(soil_char[1][0])#Detachability of soil by raindrop for silt (g/J)
    ks = float(soil_char[2][0])#Detachability of soil by raindrop for sand (g/J)
    drc = float(soil_char[0][1])#Detachability of soil by runoff for clay (g/mm)
    drz = float(soil_char[1][1])#Detachability of soil by runoff for silt (g/mm)
    drs = float(soil_char[2][1])#Detachability of soil by runoff for sand (g/mm)
    v_s_c = float(soil_char[0][2])# particle fall velocity for clay(m/s)
    v_s_z = float(soil_char[1][2])# particle fall velocity for Silt(m/s)
    v_s_s = float(soil_char[2][2])# particle fall velocity for Sand(m/s)

    #Flow Characteristics
    n = float(flow_char[0][0])#mannings coefficient
    ys = float(flow_char[1][0])#sediment density
    y = float(flow_char[2][0])#flow density
    eta = float(flow_char[3][0])#fluid viscosity
    dtu = float(flow_char[4][0])#Depth of flow for unchanneled flow
    dtw = float(flow_char[5][0])#Depth of flow for shallow rill
    dtd = float(flow_char[6][0])#Depth of flow for deeper rill

    d_o_f= np.zeros(dem.shape,dtype='f')
    d_o_f[flowacc<=10*thres]=dtu
    d_o_f[flowacc>10*thres]=dtw
    d_o_f[flowacc>100*thres]=dtd
    
    #progress_bar.setValue(10)   
    log_file.write("\nPrepared parameter layer\n")
    
    #Grid odering
    print("Creating Grid order...")
    log_file.write("\nCreating Grid order...\n")
    filename = os.path.join(output_folder,"grid_order_var.npz")
    filename = "r"+filename
    if os.path.isfile(filename)==True:
        data=np.load(filename)
        print("yeppii")
        (grid_order, start, no_ce,slope_length)= (data['grid_order'],data['start'],data['no_ce'],data['slope_length'])
        
    else:
        ##No of contributing elements
        slope_length= np.empty((dem.shape[0],dem.shape[1]),dtype='f')
        slope_length[:]=np.NaN
        no_ce = np.zeros((flowdir.shape[0],flowdir.shape[1]),dtype='f')
        for i in range(1,len(flowdir)-1):
            for j in range(1,len(flowdir[0])-1):                                        
                if np.isnan(flowdir[i][j])==0:
                    (i2,j2) = move_along_flow_dir(i,j,flowdir[i][j])
                    no_ce[i2][j2]=no_ce[i2][j2]+1
                    if abs(i-i2) == 1 and abs(j-j2) == 1:
                        slope_length[i][j] = sp * (2**0.5); #for diagonals
                    else:
                        slope_length[i][j] = sp;
                    
        ##No. of start cells & junction cells
        no_start = 0
        no_junc = 0
        
        for i in range(1,len(flowdir)-1):
            for j in range(1,len(flowdir[0]-1)):                                        
                if np.isnan(flowdir[i][j])==0:
                    if no_ce[i][j] == 0:
                        no_start = no_start + 1
                    elif no_ce[i][j] > 1:
                        no_junc = no_junc + 1
        ##Automated Grid Element Ordering
        start_no = 0
        junc_no = 0
        start = np.zeros((no_start, 2))#list of start cells
        junction = np.zeros((no_junc, 2))#list of junction cells
        junction_s = np.zeros((no_junc, 2))#ordered list of junctions
        inflow =  np.zeros((flowdir.shape[0],flowdir.shape[1]),dtype='f')
        inflow_junc = np.zeros((no_junc, 1))#inflow values for junction cells only. required for sorting.
        grid_order = np.zeros((no_start + no_junc, 2))#ordered list of start and junction cells together
        for i in range(1,len(flowdir)-1):
            for j in range(1,len(flowdir[0])-1):                                        
                if np.isnan(flowdir[i][j])==0:
                    if no_ce[i][j] == 0:
                        i1 = i
                        j1 = j
                        start[start_no][0] = i1
                        start[start_no][1] = j1              
                        start_no = start_no +1
                        while i1 > 0 and i1 < len(flowdir)-1 and j1 > 0 and j1 < len(flowdir[0])-1 and  np.isnan(flowdir[i1][j1]) == 0:
                            if no_ce[i1][j1] > 1:
                                if inflow[i1][j1] == 0:
                                    junction[junc_no][0] = i1
                                    junction[junc_no][1] = j1
                                    junc_no = junc_no + 1                            
                                    #print i1, j1
                                inflow[i1][j1] = inflow[i1][j1] + 1
                                #print 1
                            (i1,j1) = move_along_flow_dir(i1,j1,flowdir[i1][j1])
                            
        for i in range(no_junc):
            ii=int(junction[i][0])
            jj=int(junction[i][1])
            inflow_junc[i][0] = inflow[ii][jj]
        
        inflow_junc_sort=sorted(inflow_junc)
        inflow_junc_index = sorted(range(len(inflow_junc)),key=lambda x,inflow_junc=inflow_junc:inflow_junc[x])
        
        for i in range(no_junc):
           junction_s[i][:] = junction[inflow_junc_index[i]][:]
        
        grid_order=np.concatenate((start,junction_s),axis=0)
        
        np.savez(os.path.join(output_folder,"grid_order_var.npz"), grid_order,start, no_ce, slope_length)
     
    
    #progress_bar.setValue(50)
    
    print("Performing Sensitivity Analysis...")
    log_file.write("\nPerforming Sensitivity Analysis...\n")
    
    nn=0
    inc= ([0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.00009,   0.00035],[0.00035,  0.00313],[0.00426,  0.03475],[0.1, 1.5],[0.5,5.15],[0.15, 4.15],[0.02,2],[0.016,1.6],[0.015,1.5]);
    

    CB_Data = chek_box_arr.split(",")
    print("Checked data is ")
    print(CB_Data)


    CB=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
    LS_soil_loss= np.zeros((len(CB)+1, 2))
    LS_runoff=np.zeros((len(CB)+1, 2))
    Input=np.zeros((len(CB)+1,2))
    objects=np.chararray(len(CB)+1,itemsize=3)
    # if 1 in CB:
    if 'MS' in CB_Data:
        print("in MS ..................")
        log_file.write("\n In MS .....................")
        for ii in range(2):
            ms = ms * inc[0][ii]; 
            ms[ms<0.05]=0.05;
            ms[ms>0.45]=0.45;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= float(sl_outl_MT);
            LS_runoff[nn][ii]= float(q_outlet_mm);
            Input[nn][ii]=inc[0][ii];
            objects[nn]='MS'
        nn=nn+1;
        for i in range(1,len(soil_para)):      
            ms[Soil_texture==i] = float(soil_para[i-1][3])
        
                #soil moisture content at field capacity
              
    
    if 'BD' in CB_Data:
        print("in BD ..................")
        for ii in range(2):
            bd = bd * inc[1][ii]; 
            bd[bd<0.8]=0.8; 
            bd[bd>1.6]=1.6;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[1][ii];
            objects[nn]='BD'
        nn=nn+1;
        for i in range(1,len(soil_para)):
            bd[Soil_texture==i] = float(soil_para[i-1][4])        #bulk density (mg/m^3)

            
             
    if 'LP' in CB_Data:
        print("in LP ..................")
        for ii in range(2):
            lp= lp * inc[2][ii]; 
            lp[lp<1]=1; 
            lp[lp>230]=230;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[2][ii];
            objects[nn]='LP'
        nn=nn+1;
        for i in range(1,len(soil_para)):
            lp[Soil_texture==i] = float(soil_para[i-1][5]) 
          
    if 'EHD' in CB_Data:
        print("in EHD ..................")
        for ii in range(2):
            ehd= ehd * inc[3][ii]; 
            ehd[ehd<0.05]=0.05; 
            ehd[ehd>0.17]=0.17;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[3][ii];
            objects[nn]='EHD'
        nn=nn+1;
        for i in range(1,len(cover)):
                ehd[lulc==i]  = float(cover[i-1][0])        #effective hidrological depth (m)
        
            
    if 'PI' in CB_Data:
        print("in PI ..................")
        for ii in range(2):
            per_incp  = per_incp * inc[4][ii]; 
            per_incp[per_incp<0.05]=0.05;
            per_incp[per_incp>1]=1;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm; 
            Input[nn][ii]=inc[4][ii];
            objects[nn]='PI'
        nn=nn+1;
        for i in range(1,len(cover)):
            per_incp[lulc==i] = float(cover[i-1][1])    #permanent interception
        
            
    if 'ETO' in CB_Data:
        print("in ETO ..................")
        for ii in range(2):
            eto  = eto * inc[5][ii]; 
            eto[eto<0.1]=0.1; 
            eto[eto>0.9]=0.9;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[5][ii];
            objects[nn]='ETO'
        nn=nn+1;
        for i in range(1,len(cover)):
            eto[lulc==i]  = float(cover[i-1][2])        #ratio of actual to potential evapotranspiration
        
    if 'CC' in CB_Data:
        print("in CC ..................")
        for ii in range(2):
            cc  = cc * inc[6][ii]; 
            cc[cc<0]=0; 
            cc[cc>1]=1;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[6][ii];
            objects[nn]='CC'
        nn=nn+1;
        for i in range(1,len(cover)):
            cc[lulc==i]  = float(cover[i-1][3])         #canopy cover (cover for rainfall)
        
    if 'GC' in CB_Data:
        print("in GC ..................")
        for ii in range(2):
            gc  = gc * inc[7][ii]; 
            gc[gc<0]=0; 
            gc[gc>1]=1;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[7][ii];
            objects[nn]='GC'
        nn=nn+1;
        for i in range(1,len(cover)):
            gc[lulc==i]  = float(cover[i-1][4])         #ground cover (cover for runoff)
        

    if 'PH' in CB_Data:
        print("in PH ..................")
        for ii in range(2):
            ph  = ph * inc[8][ii]; 
            ph[ph<0]=0; 
            ph[ph>30]=30;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[8][ii];
            objects[nn]='PH'
        nn=nn+1;
        for i in range(1,len(cover)):
            ph[lulc==i]  = float(cover[i-1][5])         #plant height (m)
        

    if 'NV' in CB_Data:
        print("in NV ..................")
        for ii in range(2):
            nv  = nv * inc[9][ii]; 
            nv[nv<0.00001]=0.00001; 
            nv[nv>2000]=2000;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[9][ii];
            objects[nn]='NV'
        nn=nn+1;
        for i in range(1,len(cover)):
            nv[lulc==i]  = float(cover[i-1][6])         #number of plants per unit area (/m^2)
        
    
    if 'D' in CB_Data:
        print("in D ..................")
        for ii in range(2):
            d  = d * inc[10][ii]; 
            d[d<0.00001]=0; 
            d[d>3]=3;
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;  
            Input[nn][ii]=inc[10][ii];
            objects[nn]='D'
        nn=nn+1;
        for i in range(1,len(cover)):
            d[lulc==i]  = float(cover[i-1][7])  
        

            
    if 'VSc' in CB_Data:
        print("in VSc ..................")
        for ii in range(2):
            v_s_c = inc[11][ii];
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[11][ii]= sl_outl_MT;
            LS_runoff[11][ii]= q_outlet_mm; 
            Input[nn][ii]=inc[11][ii];
            objects[nn]='VSC'
        nn=nn+1;
        v_s_c = float(soil_char[0][2])# particle fall velocity for clay(m/s)
    

            
    if 'VSz' in CB_Data:
        print("in VSz ..................")
        for ii in range(2):
            v_s_z = inc[12][ii];
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            objects[nn]='VSZ'
        nn=nn+1;
        v_s_z = float(soil_char[1][2])# particle fall velocity for Silt(m/s)
    
          
    if 'VSs' in CB_Data:
        print("in VSs ..................")
        for ii in range(2):
            v_s_s = inc[13][ii];
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[13][ii];
            objects[nn]='VSS'
            nn=nn+1;
            v_s_s = float(soil_char[2][2])# particle fall velocity for Sand(m/s)
    
              
    if 'Kc' in CB_Data:
        print("in Kc ..................")
        for ii in range(2):
            kc = inc[14][ii];
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[14][ii];
            objects[nn]='KC'
        nn=nn+1;
        kc =float( soil_char[0][0])#Detachability of soil by raindrop for clay (g/J)
    
        
    if 'Kz' in CB_Data:
        print("in Kz ..................")
        for ii in range(2):
            kz = inc[15][ii];
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[15][ii];
            objects[nn]='KZ'
        nn=nn+1;
        kz = float(soil_char[1][0])#Detachability of soil by raindrop for silt (g/J)
    

    if 'Ks' in CB_Data:
        print("in Ks ..................")
        for ii in range(2):
            ks = inc[16][ii];
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[16][ii];
            objects[nn]='KS'
        nn=nn+1;
        ks = float(soil_char[2][0])#Detachability of soil by raindrop for sand (g/J)
    
        
    if 'DRc' in CB_Data:
        print("in DRc ..................")
        for ii in range(2):
            drc =  inc[17][ii];
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[17][ii];
            objects[nn]='DRC'
        nn=nn+1;
        drc = float(soil_char[0][1])#Detachability of soil by runoff for clay (g/mm)
     
    if 'DRz' in CB_Data:
        print("in DRz ..................")
        for ii in range(2):
            drz =  inc[18][ii];
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[18][ii];
            objects[nn]='DRZ'
        nn=nn+1;
        drz = float(soil_char[1][1])#Detachability of soil by runoff for silt (g/mm)
    
        
    if 'DRs' in CB_Data:
        print("in DRs ..................")
        for ii in range(2):
            drs =  inc[19][ii];
            cwd = os.path.dirname(__file__)
            exec(open(os.path.join(cwd,"MMMF.py")).read())
            LS_soil_loss[nn][ii]= sl_outl_MT;
            LS_runoff[nn][ii]= q_outlet_mm;
            Input[nn][ii]=inc[19][ii];
            objects[nn]='DRS'
        nn=nn+1;
        drs = float(soil_char[2][1])#Detachability of soil by runoff for sand (g/mm)

    log_file.write("\n calculate soil loss.................")
    
    O_soil_loss=(LS_soil_loss[:,1]-LS_soil_loss[:,0])/((LS_soil_loss[:,1]+LS_soil_loss[:,0])/2);
    print (O_soil_loss)
    O_runoff = (LS_runoff[:,1]-LS_runoff[:,0])/((LS_runoff[:,1]+LS_runoff[:,0])/2);
    I = (Input[:,1]-Input[:,0])/((Input[:,1]+Input[:,0])/2);
    
    ALS_soil_loss = O_soil_loss/I;
    ALS_runoff = O_runoff/I;
    
    log_file.write("\n plot start.................")

    y_pos = np.arange(len(objects))

    plt.bar(y_pos, ALS_runoff, align='center', alpha=0.5)
    plt.xticks(y_pos, objects,rotation=90)
    plt.ylabel('ALS')
    plt.title('Average Linear Sensitivity Plot for Runoff')
    
    plt.savefig(os.path.join(output_folder,"ALS_runoff.jpg"))
    # plt.show()
    
    
    plt.bar(y_pos, ALS_soil_loss, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation=90)
    plt.ylabel('ALS')
    plt.title('Average Linear Sensitivity Plot for Soil Loss')
    
    plt.savefig(os.path.join(output_folder,'ALS_soil_loss.jpg'))
    # plt.show()
    
    log_file.write("\n plot done.................")
#    %I=(range(:,1)-range(:,2))./((range(:,1)+range(:,2))/2);
#    ALS=O./I;
#    clearvars -except ALS LS_soil_loss LS_runoff
#            
#    %         for ii=1 : numel(inc) %loop for sensitivity
#    %             if inc(ii)~= 1
#    %                 clearvars -except dem flow_acc flow_dir grid_order no_start no_junc start no_ce LS_soil_loss LS_runoff Y ii jj inc rn rnn;
#    %                 load('F:\My_Work\Model_Implementation\Modf_MMF_modelling\input_marol_2013_RG.mat')
#    %                 run input_param.m;
#    %                 slope_length = slope_length * inc(ii);
#    %                 run soil_loss_modf.m;
#    %                 LS_soil_loss(21,ii)= sl_outl_kg;
#    %                 LS_runoff(21,ii)= q_outl_m3;
#    %                           
#    %             end                
#    %         end
#    %  for ii=1 : numel(inc) %loop for sensitivity
#    %    if inc(ii)~= 1
#    %       clearvars -except dem flow_acc flow_dir grid_order no_start no_junc start no_ce LS_soil_loss LS_runoff Y ii jj inc rn rnn;
#    %       load('F:\My_Work\Model_Implementation\Modf_MMF_modelling\input_marol_2013_RG.mat')
#    %       run input_param.m;
#    %       rn = rn .* inc(ii);
#    %       run soil_loss_modf.m;
#    %       LS_soil_loss(22,ii)= sl_outl_kg;
#    %       LS_runoff(22,ii)= q_outl_m3;
#    %                           
#    %    end                
#    % end
#    toc
#    %save('sens_anal_2013_RG.mat')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #progress_bar.setValue(90)
        
    # print("Preparing output maps...")
    # log_file.write("\nPreparing output maps...\n")
    # #Some map is loaded as a reference for rest of the arrays/maps
    # reference_filepath= os.path.join(output_folder,"dem.tif")
    # d=gdal.Open(reference_filepath)
    # if d is None:
    #     print("Error: Could not open image " + reference_filepath)
    #     raise Error1
    # global proj,geotrans,row_ref,col_ref,array_ref
    # inband=d.GetRasterBand(1)
    # proj=d.GetProjection()
    # geotrans=d.GetGeoTransform()
    # row_ref=d.RasterYSize
    # col_ref=d.RasterXSize
    # array_ref = inband.ReadAsArray(0,0,col_ref,row_ref).astype(float)
    # array_ref[array_ref == (inband.GetNoDataValue() or 0.0 or -999)]=np.nan
    # d,inband=None,None
    # export_array(qe, os.path.join(output_folder,'Runoff.tiff'))
    # log_file.write("\n Runoff Map created has unit of mm")
    # export_array(ke, os.path.join(output_folder,'Kinetic_Energy.tiff'))
    # log_file.write("\n Kinetic Energy map created has unit of J/m^2")
    # export_array(f_total, os.path.join(output_folder,'Detachment_by_Rain.tiff'))
    # log_file.write("\n Detachment due to Raindrop map created has unit of Kg/m^2")
    # export_array(h_total, os.path.join(output_folder,'Detachment_by_Rain.tiff'))
    # log_file.write("\n Detachment due to Runoff map created has unit of Kg/m^2")
    # export_array(ero_dep, os.path.join(output_folder,'Erosion_Deposition.tiff'))
    # log_file.write("\n Net Erosion/Net Deposition map created has unit of Kg/m^2")
    # export_array(sum_f_h, os.path.join(output_folder,'Soil_Erosion.tiff'))
    # log_file.write("\n \n Soil Erosion map created has unit of Kg/m^2")
    # export_array(ero_dep, os.path.join(output_folder,'Erosion_Deposition.tiff'))
    # log_file.write("\n Kinetic Energy map created has unit of Kg/m^2")
    log_file.write("\nSucessfully Finished... \n")
    print("\nSucessfully Finished... \n")
    # log_file.write("\nRunoff at outlet is ")
    # log_file.write(str(q_outl_m3))
    # log_file.write(" m^3")
    # log_file.write("\nSediment Yield at outlet is ")
    # log_file.write(str(sl_outl_MT))
    # log_file.write(" Million tonnes")
    log_file.write("\n\nStatus: Success\n")  
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
    
    print("\nTime elapsed: " + str(time.time()-start_time))
        
    log_file.write("Time elapsed: " + str(time.time()-start_time)+"\n\n")
    log_file.close()
    

