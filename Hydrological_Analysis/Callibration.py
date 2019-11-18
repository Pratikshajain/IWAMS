# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 15:58:39 2019

@author: Pratiksha
"""


# Input_SR_R ="C:\Users\Pratiksha\.qgis2\python\plugins\iwams\Hydrological_Module\Inputs"
# Output_SR_R= "C:\Users\Pratiksha\.qgis2\python\plugins\iwams\Hydrological_Module\Outputs"
# Start_year= "2008"
# End_year ="2012"

RB=2
import xlwt 
from xlwt import Workbook

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
global xlrd, csv, np, traceback, gdal, itemgetter, os, sys
global import_array, stream_order_from_flow_acc, excel_to_csv, para_array, rank_from_cm_id, cm_id_from_rank, cm_name_from_rank, stream_filter, raster_buffer, export_array
global raster_read, excel_to_csv, csv_to_data, move_along_flow_dir, export_array
import sys
import os
sys.path.append('')
from osgeo import gdal
import numpy as np, os, xlrd, csv, traceback
from operator import itemgetter

import time
start_time=time.time()
import warnings
warnings.filterwarnings("ignore")

Start_year = sys.argv[1]
End_year = sys.argv[2]
dem_file = sys.argv[3]
slop_file = sys.argv[4]
flow_direction_file = sys.argv[5]
flow_accumulation_file = sys.argv[6]
soil_texture_file = sys.argv[7]
land_use_folder = sys.argv[8]
rain_fall_data_file_cal = sys.argv[9]
boundry_file = sys.argv[10]
paremeter_file = sys.argv[11]
calibration_range_file = sys.argv[12]
observed_file = sys.argv[13]
output_folder = sys.argv[14]
CB_data = sys.argv[15]
print(CB_data)
CB=list()
if 'MS' in CB_data:
    print("Hi")
    CB.append(1)
                
if 'BD' in CB_data:
    CB.append(2)
    
if 'LP' in CB_data:
    CB.append(3)
    
if 'EHD' in CB_data:
    CB.append(4)
    
if 'PI' in CB_data:
    CB.append(5)

#ratio of actual to potential evapotranspiration
if 'ETO' in CB_data:
    CB.append(6)

if 'CC' in CB_data:
    CB.append(7)

if 'GC' in CB_data:
    CB.append(8)

if 'PH' in CB_data:
    CB.append(9)

if 'NV' in CB_data:
    CB.append(10)

if 'D' in CB_data:
    CB.append(11)

if 'VSc' in CB_data:
    CB.append(12)

if 'VSz' in CB_data:
    CB.append(13)

if 'VSs' in CB_data:
    CB.append(14)
    
if 'Kc' in CB_data:
    CB.append(15)

if 'Kz' in CB_data:
    CB.append(16)

if 'Ks' in CB_data:
    CB.append(17)

if 'DRc' in CB_data:
    CB.append(18)

if 'DRz' in CB_data:
    CB.append(19)

if 'DRs' in CB_data:
    CB.append(20)

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


def MMMF_Cal(x):
#try:
    #print("\nProgram starting... \n")
    #print (Input_SR_R)
    #Taking inputs
    # data_folder="C:\Users\\Pratiksha\\.qgis2\\python\\plugins\\iwams\\Hydrological_Module\\Data\\"
    # main_table_filepath=data_folder+"Parameter_file.xlsx"
    main_table_filepath = paremeter_file
    global input_folder,output_folder
    #global data_folder
    # input_folder=Input_SR_R+"\\"
    # output_folder=Output_SR_R+"\\"
    LULC_folder= land_use_folder
    #data_folder=os.path.dirname(os.path.abspath(__file__))+"\\MMMF\\Data\\"
    #data_folder="C:\\Users\\Pratiksha\\AppData\\Roaming\\QGIS\\QGIS3\\profiles\\default\\python\\plugins\\iwams\\Hydrological_Analysis\\Data\\"
    #main_table_filepath=data_folder+"Parameter_file.xlsx"
    
    #progress_bar=progress_bar
    #reading raster tiff files as array
    #print("Reading Input tiff files as array...")
    #log_file.write("Reading Input tiff files as array...")
    (dem, s1,r1,c1)=raster_read(dem_file)
    (flowacc, s2,r2,c2)= raster_read(flow_accumulation_file)
    (flowdir, s3,r3,c3)= raster_read(flow_direction_file)
    
    (Soil_texture,s5,r5,c5)=raster_read(soil_texture_file)
    (Slope,s6,r6,c6)=raster_read(slop_file)
    (taluka,s7,r7,c7) = raster_read(boundry_file)
    #log_file.write("\nRead all")
    s=[s1,s2,s3,s5,s6,s7]
    r=[r1,r2,r3,r5,r6,r7]
    c=[c1,c2,c3,c5,c6,c7]
    sp=s1
    #Conversion of data/main_table.xls excel files into usable csv files
    #print("Generating csv files...")
    excel_to_csv(main_table_filepath)
    excel_to_csv(rain_fall_data_file_cal)
    excel_to_csv(observed_file)
    #print("All csv files generated")
    #log_file.write("\nGenerated csv files...\n")
    #progress_bar.setValue(5)    
    #reading txt files as paramereters
    soil_para=csv_to_data(os.path.join(output_folder,"Soil.csv"))
    cover= csv_to_data(os.path.join(output_folder,"Landuse.csv"))
    soil_char= csv_to_data(os.path.join(output_folder,"Soil_char.csv"))
    flow_char=csv_to_data(os.path.join(output_folder,"Flow_char.csv"))
    other_para= csv_to_data(os.path.join(output_folder,"Other.csv"))
    
    
    #Define parameters
    #Soil Parameters
    c=np.zeros(dem.shape,dtype='f')
    silt=np.zeros(dem.shape,dtype='f')
    s =np.zeros(dem.shape,dtype='f')
    ms=np.zeros(dem.shape,dtype='f')
    bd=np.zeros(dem.shape,dtype='f')
    lp=np.zeros(dem.shape,dtype='f')
 
    
       
    
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
    
    #Grid odering
    filename = os.path.join(output_folder,"grid_order_var.npz")
    filename = "r"+filename
    if os.path.isfile(filename)==True:
        data=np.load(filename)
        
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
        
    
    est_sl_mt=np.zeros(int(End_year)-int(Start_year)+1)
    est_q_mm=np.zeros(int(End_year)-int(Start_year)+1)
    for ii in range(int(End_year)-int(Start_year)+1):        
        filename=(os.path.join(output_folder,str(ii+int(Start_year))+".csv"))
        rainfall_data= csv_to_data(filename)
        Rainfall=np.tile(np.nan, dem.shape)
        rn=np.tile(np.nan, dem.shape)
        for i in range(1,len(soil_para)):
            c[Soil_texture==i] = float(soil_para[i-1][0])      #percentage clay
            silt[Soil_texture==i] = float(soil_para[i-1][1])        #percentage silt
            s[Soil_texture==i] = float(soil_para[i-1][2])        #percentage sand
            ms[Soil_texture==i] = float(soil_para[i-1][3])        #soil moisture content at field capacity
            bd[Soil_texture==i] = float(soil_para[i-1][4])        #bulk density (mg/m^3)
            lp[Soil_texture==i] = float(soil_para[i-1][5])        #saturated lateral permeability of soil (m/day)
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

        
        for i in range(len(rainfall_data)):
            Rainfall[taluka == i+1] = float(rainfall_data[i][0])
            rn[taluka == i+1] = float(rainfall_data[i][1])   
        (lulc,s4,r4,c4)= raster_read(os.path.join(LULC_folder, "Lulc_" + str(ii+int(Start_year)) + ".tif"))
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
        
 #changing variable according to argument x       
        nn=0;
        print("x is .................................")
        print(x)
        print("Check Box data is ....................")
        print(CB_data)
     #soil moisture content at field capacity    
        if 'MS' in CB_data:
            ms = ms * x[nn]
            ms[ms<0.05]=0.05;
            ms[ms>0.45]=0.45;
            nn=nn+1;
            # CB.append[1]
            
     #bulk density (mg/m^3)               
        if 'BD' in CB_data:
            bd = bd * x[nn]; 
            bd[bd<0.8]=0.8; 
            bd[bd>1.6]=1.6;
            nn=nn+1;
            # CB.append[2]
            

        if 'LP' in CB_data:
            lp= lp * x[nn]; 
            lp[lp<1]=1; 
            lp[lp>230]=230;
            nn=nn+1;
            # CB.append[3]
            
        if 'EHD' in CB_data:
            ehd= ehd * x[nn]; 
            ehd[ehd<0.05]=0.05; 
            ehd[ehd>0.17]=0.17;
            nn=nn+1;
            # CB.append[4]
            
        if 'PI' in CB_data:
            per_incp  = per_incp * x[nn]; 
            per_incp[per_incp<0.05]=0.05;
            per_incp[per_incp>1]=1;
            nn=nn+1;
            # CB.append[5]
        
        #ratio of actual to potential evapotranspiration
        if 'ETO' in CB_data:
            eto  = eto * x[nn]; 
            eto[eto<0.1]=0.1; 
            eto[eto>0.9]=0.9;
            nn=nn+1;
            # CB.append[6]
 
        if 'CC' in CB_data:
            cc  = cc * x[nn]; 
            cc[cc<0]=0; 
            cc[cc>1]=1;
            nn=nn+1;
            # CB.append[7]
        
        if 'GC' in CB_data:
            gc  = gc * x[nn]; 
            gc[gc<0]=0; 
            gc[gc>1]=1;
            nn=nn+1;
            # CB.append[8]
    
        if 'PH' in CB_data:
            ph  = ph * x[nn]; 
            ph[ph<0]=0; 
            ph[ph>30]=30;
            nn=nn+1;
            # CB.append[9]
        
        if 'NV' in CB_data:
            nv  = nv * x[nn]; 
            nv[nv<0.00001]=0.00001; 
            nv[nv>2000]=2000;
            nn=nn+1;
            # Cb.append[10]
        
    
        if 'D' in CB_data:
            d  = d * x[nn]; 
            d[d<0.00001]=0; 
            d[d>3]=3;
            nn=nn+1;
            # CB.append[11]
        
        if 'VSc' in CB_data:
            v_s_c = x[nn]; 
            nn=nn+1;
            # CB.append[12]

        if 'VSz' in CB_data:
            v_s_z = x[nn];
            nn=nn+1;
            # CB.append[13]
        
        if 'VSs' in CB_data:
            v_s_s = x[nn];
            nn=nn+1;
            # CB.append[14]
            
        if 'Kc' in CB_data:
            kc = x[nn];
            nn=nn+1;
            # CB.append[15]
        
        if 'Kz' in CB_data:
            kz = x[nn];
            nn=nn+1;
            # CB.append[16]

        if 'Ks' in CB_data:
            ks = x[nn];
            nn=nn+1;
            # CB.append[17]
        
        if 'DRc' in CB_data:
            drc =  x[nn];
            nn=nn+1;
            # CB.append[18]
        
        if 'DRz' in CB_data:
            drz =  x[nn]
            nn=nn+1;
            # Cb.append[19]
        
        if 'DRs' in CB_data:
            drs =  x[nn];
            nn=nn+1;
            # CB.append[20]
        
         #effective rainfall
        rf = Rainfall * (1 - per_incp) * np.cos(Slope)#effective rainfall excluding permanent interception (mm)
        ld = rf * cc#leaf drainage (mm)
        dt = rf - ld#direct through fall (mm)
        
        #Kinetic energy of effective rainfall
        ke_dt = dt * 20.2 * (1 - 0.5 * np.exp(-0.067 * intensity))# kinetic energy equation cosidered as in IIRS raj paper for indian region
        ke_ld= np.tile(np.nan, dem.shape)
        ke_ld = ((15.8 * np.sqrt(ph)) - 5.87)
        ke_ld[ph < 0.15] = 0
        
        ke = ke_dt + ke_ld#total kinetic energy (J  m^-2)
        ro = Rainfall / rn #mean rain per day (mm / day)
        
        e = Rainfall /(0.9 + (Rainfall ** 2 / (300 + 25 * t + 0.05 * t ** 2) ** 2)) ** 0.5#evapotranspiration (mm)
        
        Slope[Slope==0]=np.nanmin(Slope[Slope!=0])
        
        #runoff
        q = np.zeros(dem.shape,dtype='f')
        inter_flow = np.zeros(dem.shape,dtype='f')
        rc = np.zeros(dem.shape,dtype='f')
        q_ce = np.zeros(dem.shape,dtype='f')
        qe = np.zeros(dem.shape,dtype='f')
        if_ce = np.zeros(dem.shape,dtype='f')
        
        no_start=len(start)
        no_junc=len(grid_order)-no_start
        
        for i in range(no_start + no_junc):
            i1 = int(grid_order[i][0])
            j1 = int(grid_order[i][1]) 
            i2 = int(start[0][0])
            j2 = int(start[0][1])
            while i1 > 0 and i1 < len(flowdir)-1 and j1 > 0 and j1 < len(flowdir[0])-1 and  np.isnan(flowdir[i1][j1]) == 0 and no_ce[i2][j2] < 2:
                rc[i1][j1] = (1000 * ms[i1][j1] * bd[i1][j1] * ehd[i1][j1] * (eto[i1][j1] ** 0.5)) - (if_ce[i1][j1])
                if rc[i1][j1] < 0:
                    rc[i1][j1] = 0
                    
                qe[i1][j1] = (rf[i1][j1] * np.exp(-1 * rc[i1][ j1] / ro[i1][ j1])) * (slope_length[i1][ j1] / 10) ** 0.1
                q[i1][ j1] = (((rf[i1][ j1] + q_ce[i1][ j1]) * np.exp(-1 * rc[i1][ j1] / ro[i1][ j1])) * (slope_length[i1][j1] / 10) ** 0.1)
                            
                if q[i1][ j1] > (rf[i1][ j1] + q_ce[i1][ j1]):
                    q[i1][ j1]= (rf[i1][ j1] + q_ce[i1][ j1])
                inter_flow[i1][ j1] = ((Rainfall[i1][ j1] - e[i1][ j1] - qe[i1][ j1])/ rn[i1][ j1]) * ((lp[i1][ j1] * np.sin(Slope[i1][j1]))/sp)
                if inter_flow[i1][ j1] < 0:
                    inter_flow[i1][ j1] = 0
                if_ce[i1][ j1] = inter_flow[i1][ j1] + if_ce[i1][ j1]
        
                [i2, j2] = move_along_flow_dir(i1, j1, flowdir[i1][ j1])
                q_ce[i2][ j2] = q_ce[i2][ j2] + q[i1][ j1]    # adds up runoff from all contributing element
        
                if_ce[i2][ j2] = if_ce[i2][ j2] + if_ce[i1][ j1]
        
                i1 = i2
                j1 = j2
        
        #Detachment by raindrop
        f_c = c * (1 - st) * ke * kc / 10 ** 5 #clay
        f_z = silt * (1 - st) * ke * kz / 10 ** 5#silt
        f_s = s * (1 - st) * ke * ks / 10 ** 5#sand
        f_c[lulc == 16] = 0; f_z[lulc == 16] = 0; f_s[lulc == 16] = 0; # for water bodies
        f_total = f_c + f_z + f_s
        
        #Detachment by runoff
        
        h_c = (c * (q ** 1.5) * (1 - gc - st) * (np.sin(Slope) ** 0.3) * drc / 10 ** 5)#clay
        h_z = (silt * (q ** 1.5) * (1 - gc - st) * (np.sin(Slope) ** 0.3) * drz / 10 ** 5)#silt
        h_s = (s * (q ** 1.5) * (1 - gc - st) * (np.sin(Slope) ** 0.3) * drs / 10 ** 5)#sand
        h_c[lulc == 16] = 0; h_z[lulc == 16] = 0; h_s[lulc == 16] = 0 #for waterbodies
        h_total = h_c + h_z + h_s
           
        #Flow velocity
        v_b = (Slope ** 0.5) * (dtu ** 0.67) / n#unchanneled flow (m/s)
        v_a = (Slope ** 0.5) * (d_o_f ** 0.67) * np.exp(st * -0.018) / n#actual flow
        v_v = (((2 * g) / (d * nv)) ** 0.5) * (Slope ** 0.5)#effects of vegetation cover
        v_v[d==0]=1; v_v[nv==0]=1; # to avoid infinity values
        v_v[np.isnan(dem)==1]=np.nan; v_v[lulc == 16] = 1;
        v_v[lulc == 1] = 1
        
        v_t = np.ones((dem.shape[0],dem.shape[1]))#effects of roughness
            
        #Particle Fall Number
        nf_c = (slope_length * v_s_c) / (v_b * d_o_f)#clay
        nf_z = (slope_length * v_s_z) / (v_b * d_o_f)#silt
        nf_s = (slope_length * v_s_s) / (v_b * d_o_f)#sand
        
        #Immediate deposition of detatched particles
        dep_c = np.tile(np.nan, dem.shape)
        dep_z = np.tile(np.nan, dem.shape)
        dep_s = np.tile(np.nan, dem.shape)
        dep_c= (nf_c** 0.29)* 44.1    #clay
        dep_c[dep_c>100]=100
        dep_z= (nf_z** 0.29) * 44.1    #silt
        dep_z[dep_z>100]=100
        dep_s= (nf_s ** 0.29) * 44.1    #sand
        dep_s[dep_s>100]=100
        
        #Transport capacity of runoff
        
        v_fact = (v_a * v_v * v_t / v_b**3)
        v_fact[v_fact > 1] = 1
        tc_c = (v_fact) * c * (q ** 2) * np.sin(Slope) / 10 ** 5#clay
        tc_z = (v_fact) * silt * (q ** 2) * np.sin(Slope) / 10 ** 5#silt
        tc_s = (v_fact) * s * (q ** 2) * np.sin(Slope) / 10 ** 5#sand
        
        Transportation_capacity = np.empty((dem.shape[0],dem.shape[1]),dtype='f')
        Transportation_capacity[:]=np.NaN
        Transportation_capacity = ((tc_c+tc_z+tc_s)/3)
        
        #Soil Loss
        sl_c = np.tile(np.nan, dem.shape)
        sl_z = np.tile(np.nan, dem.shape)
        sl_s = np.tile(np.nan, dem.shape)
        g_c = np.tile(np.nan, dem.shape)
        g_z = np.tile(np.nan, dem.shape)
        g_s = np.tile(np.nan, dem.shape)
        g_total = np.zeros(dem.shape,dtype='f')
        g_c_n = np.tile(np.nan, dem.shape)
        g_z_n = np.tile(np.nan, dem.shape)
        g_s_n = np.tile(np.nan, dem.shape)
        g_total_n = np.zeros(dem.shape,dtype='f')
        g_c1 = np.tile(np.nan, dem.shape)
        g_z1 = np.tile(np.nan, dem.shape)
        g_s1 = np.tile(np.nan, dem.shape)
        sl_ce_c = np.zeros(dem.shape,dtype='f')
        sl_ce_z = np.zeros(dem.shape,dtype='f')
        sl_ce_s = np.zeros(dem.shape,dtype='f')
        Dep_post_c= np.zeros(dem.shape,dtype='f')
        Dep_post_s= np.zeros(dem.shape,dtype='f')
        Dep_post_z= np.zeros(dem.shape,dtype='f')
        dep_imd_c = np.zeros(dem.shape,dtype='f')
        dep_imd_z = np.zeros(dem.shape,dtype='f')
        dep_imd_s = np.zeros(dem.shape,dtype='f')
               
        g_c_n = ((f_c + h_c) * (1 - (dep_c / 100))) #clay
        g_z_n = ((f_z + h_z) * (1 - (dep_z / 100))) #silt
        g_s_n = ((f_s + h_s) * (1 - (dep_s / 100))) #sand
        dep_imd_c = ((f_c + h_c) * ((dep_c / 100)))
        dep_imd_z = ((f_z + h_z) * ((dep_z / 100)))
        dep_imd_s=  ((f_s + h_s) * ((dep_s / 100)))
        
        for i in range(no_start+no_junc):
            i1 = int(grid_order[i][0])
            j1 = int(grid_order[i][1])
            i2 = int(start[0][0])
            j2 = int(start[0][1])
            while i1 > 0 and i1 < len(flowdir)-1 and j1 > 0 and j1 < len(flowdir[0])-1 and  np.isnan(flowdir[i1][j1]) == 0 and no_ce[i2][j2] < 2:
                
                g_c[i1][j1] = ((f_c[i1][j1] + h_c[i1][j1]) * (1 - (dep_c[i1][j1] / 100))) + sl_ce_c[i1][j1] #clay
                g_z[i1][j1] = ((f_z[i1][j1] + h_z[i1][j1]) * (1 - (dep_z[i1][j1] / 100))) + sl_ce_z[i1][j1] #silt
                g_s[i1][j1] = ((f_s[i1][j1] + h_s[i1][j1]) * (1 - (dep_s[i1][j1] / 100))) + sl_ce_s[i1][j1] #sand
                g_total[i1][j1]= g_c[i1][j1]+g_z[i1][j1] + g_s[i1][j1]
                if tc_c[i1][j1] >= g_c[i1][j1]:
                    sl_c[i1][j1] = g_c[i1][j1]
                    Dep_post_c[i1][j1] = 0
                else:
                    g_c1[i1][j1] = g_c[i1][j1] * (1 - dep_c[i1][j1]/100)
                    if tc_c[i1][j1] >= g_c1[i1][j1]:
                        sl_c[i1][j1] = tc_c[i1][j1]
                        Dep_post_c[i1][j1] = g_c[i1][j1]-tc_c[i1][j1]
                    else:
                        sl_c[i1][j1] = g_c1[i1][j1]
                        Dep_post_c[i1][j1] = g_c[i1][j1]- g_c1[i1][j1]
        
                if tc_z[i1][j1] >= g_z[i1][j1]:
                    sl_z[i1][j1] = g_z[i1][j1]
                    Dep_post_z[i1][j1] = 0
                else:
                    g_z1[i1][j1] = g_z[i1][j1] * (1 - dep_z[i1][j1]/100)
                    if tc_z[i1][j1] >= g_z1[i1][j1]:
                        sl_z[i1][j1] = tc_z[i1][j1]
                        Dep_post_z[i1][j1] = g_z[i1][j1]-tc_z[i1][j1]
                    else:
                        sl_z[i1][j1] = g_z1[i1][j1]
                        Dep_post_z[i1][j1] = g_z[i1][j1]- g_z1[i1][j1]
        
                if tc_s[i1][j1] >= g_s[i1][j1]:
                    sl_s[i1][j1] = g_s[i1][j1]
                    Dep_post_s[i1][j1] = 0
                else:
                    g_s1[i1][j1] = g_s[i1][j1] * (1 - dep_s[i1][j1]/100)
                    if tc_s[i1][j1] >= g_s1[i1][j1]:
                        sl_s[i1][j1] = tc_s[i1][j1]
                        Dep_post_s[i1][j1] = g_s[i1][j1]-tc_s[i1][j1]
                    else:
                        sl_s[i1][j1] = g_s1[i1][j1]
                        Dep_post_s[i1][j1] =g_s[i1][j1]- g_s1[i1][j1]
        
                [i2,j2] = move_along_flow_dir(i1,j1,flowdir[i1][j1])
                sl_ce_c[i2][j2] = sl_ce_c[i2][j2] + sl_c[i1][j1]
                sl_ce_z[i2][j2] = sl_ce_z[i2][j2] + sl_z[i1][j1]
                sl_ce_s[i2][j2] = sl_ce_s[i2][j2] + sl_s[i1][j1]
                
                i1 = i2
                j1 = j2
        sl_total = sl_c + sl_z + sl_s #total soil loss at outlet in kg/m^2
        #tolat deposition
        dep_total_c = dep_imd_c + Dep_post_c; 
        dep_total_z = dep_imd_z + Dep_post_z; 
        dep_total_s = dep_imd_s + Dep_post_s; 
        dep_total = dep_total_c + dep_total_s +dep_total_z;
        
        #Total annual detachment of each pixel(kg/m^2)
        sum_f_h = f_total + h_total;
        
        #net erosion and deposition
        ero_dep= sum_f_h - dep_total; #+ve erosion -ve deposition
        
        iii = int(grid_order[no_junc+no_start-1][0])
        jjj = int(grid_order[no_junc+no_start-1][1])
        q_outl_m3= (q[iii][jjj]*sp*sp)/1000;
        sl_outl_MT = (sl_total[iii][jjj]*sp*sp)/10**9;
        sl_outlet_kg_m2=sl_total[iii][jjj];
        q_outlet_mm=q[iii][jjj];
    
        est_q_mm[ii]=q_outlet_mm/flowacc[iii][jjj]; #in mm
        est_sl_mt[ii]=sl_outl_MT;
        
    #print ('est_q',est_q_mm)
    #print ('est_sl',est_sl_mt)
    #progress_bar.setValue(90)
    observation_data= csv_to_data(os.path.join(output_folder,"Observation.csv"))
    #for i in range(1,len(observation_data)):
    obs_q_mm=np.transpose(observation_data[:,0])
    obs_sl_mt = np.transpose(observation_data[:,1])
    #print np.mean(est_q_mm)
    #print np.mean(est_sl_mt)
    print ('X',x)
    if RB==1:
        error_q=((est_q_mm-obs_q_mm)/obs_q_mm)*100;
        NSE_q=1-( sum((obs_q_mm-est_q_mm)**2)/sum((obs_q_mm-np.mean(obs_q_mm))**2));
        PBIAS_q= (sum(est_q_mm-obs_q_mm)*100)/sum(obs_q_mm);
        RSR_q = np.sqrt(sum((obs_q_mm-est_q_mm)**2))/np.sqrt(sum((obs_q_mm-np.mean(obs_q_mm))**2));
        opt= NSE_q; 
        print ('Error:',error_q,'NSE:',NSE_q,'PBIAS:',PBIAS_q,'RSR:',RSR_q)
    if RB==2:
        error_sl=((est_sl_mt-obs_sl_mt)/obs_sl_mt)*100;
        NSE_sl=1-( sum((obs_sl_mt-est_sl_mt)**2)/sum((obs_sl_mt-np.mean(obs_sl_mt))**2));
        PBIAS_sl= (sum(est_sl_mt-obs_sl_mt)*100)/sum(obs_sl_mt);
        RSR_sl = np.sqrt(sum((obs_sl_mt-est_sl_mt)**2))/np.sqrt(sum((obs_sl_mt-np.mean(obs_sl_mt))**2));
        opt= NSE_sl; 
        print ('Error:',error_sl,'NSE:',NSE_sl,'PBIAS:',PBIAS_sl,'RSR:',RSR_sl)
    #progress_bar.setValue(100)
    
    return opt

try:
#    fun = @cali_func_q;
#    MS, BD, LP1,LP2,LP3,EHD, Et/Eo
#    for runoff callibration
#    lb = [0.65, 0.65, 0.19, 0.77, 1.92, 0.65, 0.65];
#    ub = [1.35, 1.35, 23.08, 73.08, 96.15, 1.35, 1.35]; 
#    for soilloss callibration
#    GC, VSc, DRc
#    lb = [0.65, 0.00009, 0.02];
#    ub = [1.35, 0.00035, 2];
    # input_folder=Input_SR_R+"\\"
    # output_folder=Output_SR_R+"\\"
    log_file=open(os.path.join(output_folder,"log_file.csv"),'w+')
    log_file.write(output_folder)
    log_file.write("\nTime,"+time.strftime("%H:%M:%S")+"\n")
    log_file.write("Date,"+time.strftime("%d/%m/%Y")+"\n\n")
    
    excel_to_csv(calibration_range_file)
    Range = csv_to_data(os.path.join(output_folder,"Range.csv"))
    lb= np.zeros(len(CB))
    ub= np.zeros(len(CB))
    for i in range(len(CB)):
        lb[i] = Range[CB[i]-1,2]
        ub[i] = Range[CB[i]-1,3]
    
    
    if RB==1:
#   Runoff
        print("Starting Runoff Callibration")
        #lb = [0.65, 0.65, 0.19, 0.77, 1.92, 0.65, 0.65];
        #ub = [1.35, 1.35, 23.08, 73.08, 96.15, 1.35, 1.35]; 

        xopt, fopt = pso(MMMF_Cal, lb, ub,args=(), kwargs={},swarmsize=2, omega=0.5, phip=0.5, phig=0.5, maxiter=4, minstep=1e-8,minfunc=1e-8, debug=False)
        #print (xopt)
        #print (fopt)
        log_file.write(xopt,fopt)
    if RB ==2:
#   Soil Loss
#    GC, VSc, DRc
#        lb = [0.65, 0.00009, 0.02];
#        ub = [1.35, 0.00035, 2];
        print("Starting soil loss Callibration")
        xopt, fopt = pso(MMMF_Cal, lb, ub,args=(), kwargs={}, swarmsize=2, omega=0.5, phip=0.5, phig=0.5, maxiter=4, minstep=1e-8, minfunc=1e-8, debug=False)
        #print (xopt)
        #print (fopt)
        log_file.write(str(xopt))
    Parameter=['MS','BD','LP', 'EHD', 'per_incp','Et/Eo','CC','GC','PH','NV','D','VSc','VSs','VSz','Kc','Kz','Ks','DRc','DRz','DRs']

    wb1 = Workbook()
    sheet1 = wb1.add_sheet('Calibrated_values')
    sheet1.write(0, 0, 'ID')
    sheet1.write(0, 1, 'Parameter')
    sheet1.write(0, 2, 'Parameter_ID')
    sheet1.write(0, 3, 'Callibrated value')
    for i in range(len(CB)):
        sheet1.write(i+1, 0, i )
        sheet1.write(i+1, 1, Parameter[CB[i]-1] )
        sheet1.write(i+1, 2, CB[i] )
        sheet1.write(i+1, 3, xopt[i] )
    wb1.save(os.path.join(output_folder,'Calibration_parameters.xls'))
    
    
    
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
    





