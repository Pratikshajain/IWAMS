# -*- coding: utf-8 -*-
"""
Created on Wed Sep 04 17:51:49 2019

@author: Pratiksha
"""

import os
import sys 

sys.path.append ('/Users/pratikshajain/opt/anaconda3/lib/python3.7/site-packages')

from qgis.PyQt import uic 
from qgis.PyQt import QtWidgets

from PyQt5.QtWidgets import QFileDialog

global xlrd, csv, np, traceback, gdal, itemgetter, os
global import_array, stream_order_from_flow_acc, excel_to_csv, para_array, rank_from_cm_id, cm_id_from_rank, cm_name_from_rank, stream_filter, raster_buffer, export_array
global raster_read, excel_to_csv, csv_to_data, move_along_flow_dir, export_array, CB_data

from osgeo import gdal
import numpy as np, os, xlrd, csv, traceback
from operator import itemgetter

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import time
from pyswarm import pso
import xlwt 
from xlwt import Workbook


start_time=time.time()
import warnings
warnings.filterwarnings("ignore")

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'Hydrological_dialog_base.ui'))

def excel_to_csv(ExcelFile,output_folder):
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


def MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs):
            
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

    ms = ms * x_MS; ms[ms<0.05]=0.05;ms[ms>0.45]=0.45;
    bd = bd * x_BD; bd[bd<0.8]=0.8; bd[bd>1.6]=1.6;
    lp= lp * x_LP; lp[lp<1]=1; lp[lp>230]=230;
    ehd= ehd * x_EHD; ehd[ehd<0.05]=0.05;ehd[ehd>0.17]=0.17;
    per_incp  = per_incp *  x_PI; per_incp[per_incp<0.05]=0.05;per_incp[per_incp>1]=1;
    eto  = eto * x_ETO; eto[eto<0.1]=0.1; eto[eto>0.9]=0.9;
    cc  = cc * x_CC; cc[cc<0]=0; cc[cc>1]=1;
    gc  = gc * x_GC; gc[gc<0]=0; gc[gc>1]=1;
    ph  = ph * x_PH; ph[ph<0]=0; ph[ph>30]=30;
    nv  = nv * x_NV; nv[nv<0.00001]=0.00001; nv[nv>2000]=2000;
    d  = d * x_D; d[d<0.00001]=0; d[d>3]=3;
    v_s_z = x_VSz;
    v_s_s = x_VSs;
    kc = x_Kc;          
    kz = x_Kz;
    ks = x_Ks;
    drc =  x_DRc;
    drz = x_DRz
    drs =  x_DRs;

  

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
    print("Creating Grid order...")
    filename = os.path.join(output_folder,"grid_order_var.npz")
    filename = "r"+filename
    if os.path.isfile(filename)==True:
        data=np.load(filename)
        print(data)
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
    
    print("Performing Soil Erosion modelling...")
    
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
    q_outlet_mm=q[iii][jjj]/flowacc[iii][jjj];
    return (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3,q_outlet_mm, sl_outl_MT)



class Simple_Run:            

    def Simple_Run(self,Simulation_Year,dem_file,slop_file,flow_direction_file,flow_accumulation_file,land_use_file,soil_texture_file,rainfall_data_file,boundry_file,paremeter_file,output_folder,progress_bar):

        print("In Simple Run ............................")
        #cwd = os.path.dirname(__file__)

        #print("python3 "+ cwd +"/Hydrological_Analysis/Simple_Run.py")
        #os.system("python3 "+ cwd +"/Hydrological_Analysis/Simple_Run.py")
        # exec(open(r"/home/viral/.local/share/QGIS/QGIS3/profiles/default/python/plugins/iwams/Hydrological_Analysis/Simple_Run.py").read())
        #os.system("python3 "+ cwd +"/Hydrological_Analysis/Simple_Run.py "+ str(Simulation_Year) +" "+ str(dem_file) +" "+ str(slop_file) +" "+ str(flow_direction_file) +" "+ str(flow_accumulation_file) +" "+ str(land_use_file) +" "+ str(soil_texture_file) +" "+ str(rainfall_data_file) +" "+ str(boundry_file) +" "+ str(paremeter_file) +" "+ str(output_folder))
        #exec("C:\\Users\\Pratiksha\\AppData\\Roaming\\QGIS\\QGIS3\\profiles\\default\\python\\plugins\\iwams\\Hydrological_Analysis\\Simple_Run.py")
        # -*- coding: utf-8 -*-
        
        #reading args
        
        
        class Error1(Exception):
           """This is a custom exception."""
           pass
        
           
        try:
            print("\nProgram starting... \n")
            #Taking inputs
        
        
            # global input_folder,output_folder,data_folder
            # input_folder=Input_SR_R
            # output_folder=Output_SR_R
            # Y=str(Simulation_year_SR_R)
            #data_folder=os.path.dirname(os.path.abspath(__file__))+"\\MMMF\\Data\\"
            # data_folder="/home/viral/.local/share/QGIS/QGIS3/profiles/default/python/plugins/iwams/Hydrological_Analysis/Data/"
            # main_table_filepath=data_folder+"Parameter_file.xlsx"
            progress_bar=progress_bar
            progress_bar.setValue(0)
            main_table_filepath = paremeter_file
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
            excel_to_csv(main_table_filepath,output_folder)
            excel_to_csv(rainfall_data_file,output_folder)
            print("All csv files generated")
            log_file.write("\nGenerated csv files...\n")
            progress_bar.setValue(10)    
            #reading txt files as paramereters
            soil_para=csv_to_data(os.path.join(output_folder,"Soil.csv"))#output 
            cover= csv_to_data(os.path.join(output_folder,"Landuse.csv"))
            soil_char= csv_to_data(os.path.join(output_folder,"Soil_char.csv"))
            flow_char=csv_to_data(os.path.join(output_folder,"Flow_char.csv"))
            other_para= csv_to_data(os.path.join(output_folder,"Other.csv"))
            
            #rainfall data
            rainfall_data= csv_to_data(os.path.join(output_folder,"Annual.csv"))
            
            log_file.write("\nRead CSV to data\n")

            x_MS=1;x_BD=1;x_LP=1;x_EHD=1; x_PI=1;x_ETO=1;x_CC=1;x_GC=1;x_PH=1;x_NV=1;x_D=1;

            x_Kc =float( soil_char[0][0])#Detachability of soil by raindrop for clay (g/J)
            x_Kz = float(soil_char[1][0])#Detachability of soil by raindrop for silt (g/J)
            x_Ks = float(soil_char[2][0])#Detachability of soil by raindrop for sand (g/J)
            x_DRc = float(soil_char[0][1])#Detachability of soil by runoff for clay (g/mm)
            x_DRz = float(soil_char[1][1])#Detachability of soil by runoff for silt (g/mm)
            x_DRs = float(soil_char[2][1])#Detachability of soil by runoff for sand (g/mm)
            x_VSc = float(soil_char[0][2])# particle fall velocity for clay(m/s)
            x_VSz = float(soil_char[1][2])# particle fall velocity for Silt(m/s)
            x_VSs = float(soil_char[2][2])# particle fall velocity for Sand(m/s)
           


            (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm, sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)



            progress_bar.setValue(90)
                
            print("Preparing output maps...")
            log_file.write("\nPreparing output maps...\n")
            #Some map is loaded as a reference for rest of the arrays/maps
            reference_filepath= dem_file
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
            export_array(qe, os.path.join(output_folder,'Runoff.tiff'))
            log_file.write("\n Runoff Map created has unit of mm")
            export_array(ke, os.path.join(output_folder,'Kinetic_Energy.tiff'))
            log_file.write("\n Kinetic Energy map created has unit of J/m^2")
            export_array(f_total, os.path.join(output_folder,'Detachment_by_Rain.tiff'))
            log_file.write("\n Detachment due to Raindrop map created has unit of Kg/m^2")
            export_array(h_total, os.path.join(output_folder,'Detachment_by_Rain.tiff'))
            log_file.write("\n Detachment due to Runoff map created has unit of Kg/m^2")
            export_array(ero_dep, os.path.join(output_folder,'Erosion_Deposition.tiff'))
            log_file.write("\n Net Erosion/Net Deposition map created has unit of Kg/m^2")
            export_array(sum_f_h, os.path.join(output_folder,'Soil_Erosion.tiff'))
            log_file.write("\n \n Soil Erosion map created has unit of Kg/m^2")
            export_array(ero_dep, os.path.join(output_folder,'Erosion_Deposition.tiff'))
            log_file.write("\n Kinetic Energy map created has unit of Kg/m^2")
            log_file.write("\nSucessfully Finished... \n")
            print("\nSucessfully Finished... \n")
            log_file.write("\nRunoff at outlet is ")
            log_file.write(str(q_outl_m3))
            log_file.write(" m^3")
            log_file.write("\nSediment Yield at outlet is ")
            log_file.write(str(sl_outl_MT))
            log_file.write(" Million tonnes")
            log_file.write("\n\nStatus: Success\n")  
            progress_bar.setValue(100)
        
        
        
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
            workbook = xlrd.open_workbook(rainfall_data_file)
            sheet_names=workbook.sheet_names()
            for sheet in sheet_names[0:]:
                try:
                    os.remove(input_folder+sheet+".csv")
                except:
                    continue
            print("\nTime elapsed: " + str(time.time()-start_time))
                
            log_file.write("Time elapsed: " + str(time.time()-start_time)+"\n\n")
            log_file.close()
            
        

class Sensitivity_Run:

    def Sensitivity_Run(self,Simulation_Year,dem_file,slop_file,flow_direction_file,flow_accumulation_file,land_use_file,soil_texture_file,rainfall_data_file,boundry_file,paremeter_file,output_folder,cb_arr,progress_bar):

        print("In Sensitivity_Run .......................")
        

        class Error1(Exception):
           """This is a custom exception."""
           pass
        
        
          
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
            progress_bar=progress_bar
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
            excel_to_csv(main_table_filepath,output_folder)
            excel_to_csv(rainfall_data_file,output_folder)
            print("All csv files generated")
            log_file.write("\nGenerated csv files...\n")
            progress_bar.setValue(5)    
            #reading txt files as paramereters
            soil_para=csv_to_data(os.path.join(output_folder,"Soil.csv"))#output 
            cover= csv_to_data(os.path.join(output_folder,"Landuse.csv"))
            soil_char= csv_to_data(os.path.join(output_folder,"Soil_char.csv"))
            flow_char=csv_to_data(os.path.join(output_folder,"Flow_char.csv"))
            other_para= csv_to_data(os.path.join(output_folder,"Other.csv"))
            
            #rainfall data
            rainfall_data= csv_to_data(os.path.join(output_folder,"Annual.csv"))
            progress_bar.setValue(10) 
            log_file.write("\nRead CSV to data\n")
 
            x_MS=1;x_BD=1;x_LP=1;x_EHD=1; x_PI=1;x_ETO=1;x_CC=1;x_GC=1;x_PH=1;x_NV=1;x_D=1;

            x_Kc =float( soil_char[0][0])#Detachability of soil by raindrop for clay (g/J)
            x_Kz = float(soil_char[1][0])#Detachability of soil by raindrop for silt (g/J)
            x_Ks = float(soil_char[2][0])#Detachability of soil by raindrop for sand (g/J)
            x_DRc = float(soil_char[0][1])#Detachability of soil by runoff for clay (g/mm)
            x_DRz = float(soil_char[1][1])#Detachability of soil by runoff for silt (g/mm)
            x_DRs = float(soil_char[2][1])#Detachability of soil by runoff for sand (g/mm)
            x_VSc = float(soil_char[0][2])# particle fall velocity for clay(m/s)
            x_VSz = float(soil_char[1][2])# particle fall velocity for Silt(m/s)
            x_VSs = float(soil_char[2][2])# particle fall velocity for Sand(m/s)
           

             
            progress_bar.setValue(20)
            
            print("Performing Sensitivity Analysis...")
            log_file.write("\nPerforming Sensitivity Analysis...\n")
            
            nn=0
            inc = ([0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.00009,   0.00035],[0.00035,  0.00313],[0.00426,  0.03475],[0.1, 1.5],[0.5,5.15],[0.15, 4.15],[0.02,2],[0.016,1.6],[0.015,1.5]);
            
            inc= ([0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.7,1.3],[0.00009,   0.00035],[0.00035,  0.00313],[0.00426,  0.03475],[0.1, 1.5],[0.5,5.15],[0.15, 4.15],[0.02,2],[0.016,1.6],[0.015,1.5]);
            
        
            CB_Data = cb_arr.split(",")
            print("Checked data is ")
            print(CB_Data)
        
        
            CB=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
            LS_soil_loss= np.zeros((len(CB)+1, 2))
            LS_runoff=np.zeros((len(CB)+1, 2))
            Input=np.zeros((len(CB)+1,2))
            
            objects=np.chararray(len(CB)+1,itemsize=5,unicode=True)
            # if 1 in CB:
#soil moisture content at field capacity
            if 'MS' in CB_Data:
                print("in MS ..................")
                log_file.write("\n In MS .....................")
                for ii in range(2):
                    x_MS = inc[0][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm, sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= float(sl_outl_MT);
                    LS_runoff[nn][ii]= float(q_outlet_mm);
                    Input[nn][ii]=inc[0][ii];
                    objects[nn]='MS'
                nn=nn+1;
                x_MS = 1 ;
                
            
            if 'BD' in CB_Data:
                print("in BD ..................")
                for ii in range(2):
                    x_BD = inc[1][ii]; 
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3,q_outlet_mm, sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[1][ii];
                    objects[nn]='BD'
                nn=nn+1;
                x_BD = 1;       #bulk density (mg/m^3)
        
                    
                  
            if 'LP' in CB_Data:
                print("in LP ..................")
                for ii in range(2):
                    x_LP= inc[2][ii]; 

                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm, sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[2][ii];
                    objects[nn]='LP'
                nn=nn+1;
                x_LP = 1;
                  
            progress_bar.setValue(30)
            if 'EHD' in CB_Data:
                print("in EHD ..................")
                for ii in range(2):
                    x_EHD= inc[3][ii]; 

                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm, sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[3][ii];
                    objects[nn]='EHD'
                nn=nn+1;
                x_EHD = 1;       #effective hidrological depth (m)
                
                    
            
            if 'PI' in CB_Data:
                print("in PI ..................")
                for ii in range(2):
                    x_PI = inc[4][ii]; 

                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm, sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm; 
                    Input[nn][ii]=inc[4][ii];
                    objects[nn]='PI'
                nn=nn+1;
                x_PI=1;
                    
            
            if 'ETO' in CB_Data:
                print("in ETO ..................")
                for ii in range(2):
                    x_ETO = inc[5][ii]; 
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm, sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[5][ii];
                    objects[nn]='ETO'
                nn=nn+1;
                x_ETO = 1;       #ratio of actual to potential evapotranspiration
                
            progress_bar.setValue(40)
            if 'CC' in CB_Data:
                print("in CC ..................")
                for ii in range(2):
                    x_CC = inc[6][ii]; 
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3,q_outlet_mm, sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[6][ii];
                    objects[nn]='CC'
                nn=nn+1;
                x_CC=1;        #canopy cover (cover for rainfall)
                
            
            if 'GC' in CB_Data:
                print("in GC ..................")
                for ii in range(2):
                    x_GC = inc[7][ii]; 

                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[7][ii];
                    objects[nn]='GC'
                nn=nn+1;
                x_GC=1;        #ground cover (cover for runoff)
                
        
            
            if 'PH' in CB_Data:
                print("in PH ..................")
                for ii in range(2):
                    x_PH = inc[8][ii]; 

                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3,sq_outlet_mm,l_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[8][ii];
                    objects[nn]='PH'
                nn=nn+1;
                x_PH= 1;        #plant height (m)
                
        
            progress_bar.setValue(50)
            if 'NV' in CB_Data:
                print("in NV ..................")
                for ii in range(2):
                    x_NV  = inc[9][ii]; 

                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[9][ii];
                    objects[nn]='NV'
                nn=nn+1;
                x_NV = 1;         #number of plants per unit area (/m^2)
                
            
            
            if 'D' in CB_Data:
                print("in D ..................")
                for ii in range(2):
                    x_D = inc[10][ii]; 
                    d[d<0.00001]=0; 
                    d[d>3]=3;
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;  
                    Input[nn][ii]=inc[10][ii];
                    objects[nn]='D'
                nn=nn+1;
                x_D = 1;
                
        
            progress_bar.setValue(60)       
            if 'VSc' in CB_Data:
                print("in VSc ..................")
                for ii in range(2):
                    x_VSc= inc[11][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[11][ii]= sl_outl_MT;
                    LS_runoff[11][ii]= q_outlet_mm; 
                    Input[nn][ii]=inc[11][ii];
                    objects[nn]='VSC'
                nn=nn+1;
                x_VSc= float(soil_char[0][2])# particle fall velocity for clay(m/s)
            
        
                    
            
            
            if 'VSz' in CB_Data:
                print("in VSz ..................")
                for ii in range(2):
                    x_VSz= inc[12][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    objects[nn]='VSZ'
                nn=nn+1;
                x_VSz = float(soil_char[1][2])# particle fall velocity for Silt(m/s)
            
                  
            
            if 'VSs' in CB_Data:
                print("in VSs ..................")
                for ii in range(2):
                    x_VSs = inc[13][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[13][ii];
                    objects[nn]='VSS'
                    nn=nn+1;
                    x_VSs = float(soil_char[2][2])# particle fall velocity for Sand(m/s)
            
                      
            progress_bar.setValue(70)
            if 'Kc' in CB_Data:
                print("in Kc ..................")
                for ii in range(2):
                    x_Kc= inc[14][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[14][ii];
                    objects[nn]='KC'
                nn=nn+1;
                x_Kc =float( soil_char[0][0])#Detachability of soil by raindrop for clay (g/J)
            
                
            if 'Kz' in CB_Data:
                print("in Kz ..................")
                for ii in range(2):
                    x_Kz = inc[15][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[15][ii];
                    objects[nn]='KZ'
                nn=nn+1;
                x_Kz = float(soil_char[1][0])#Detachability of soil by raindrop for silt (g/J)
            
        
            if 'Ks' in CB_Data:
                print("in Ks ..................")
                for ii in range(2):
                    x_Ks = inc[16][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[16][ii];
                    objects[nn]='KS'
                nn=nn+1;
                x_Ks = float(soil_char[2][0])#Detachability of soil by raindrop for sand (g/J)
            
            progress_bar.setValue(80)
                
            if 'DRc' in CB_Data:
                print("in DRc ..................")
                for ii in range(2):
                    x_DRc =  inc[17][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[17][ii];
                    objects[nn]='DRC'
                nn=nn+1;
                x_DRc = float(soil_char[0][1])#Detachability of soil by runoff for clay (g/mm)
             
            if 'DRz' in CB_Data:
                print("in DRz ..................")
                for ii in range(2):
                    x_DRz =  inc[18][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[18][ii];
                    objects[nn]='DRZ'
                nn=nn+1;
                x_DRz = float(soil_char[1][1])#Detachability of soil by runoff for silt (g/mm)
            
                
            if 'DRs' in CB_Data:
                print("in DRs ..................")
                for ii in range(2):
                    x_DRs =  inc[19][ii];
                    (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                    LS_soil_loss[nn][ii]= sl_outl_MT;
                    LS_runoff[nn][ii]= q_outlet_mm;
                    Input[nn][ii]=inc[19][ii];
                    objects[nn]='DRS'
                nn=nn+1;
                x_DRs = float(soil_char[2][1])#Detachability of soil by runoff for sand (g/mm)
        
            log_file.write("\n calculate soil loss.................")
            
            O_soil_loss=(LS_soil_loss[:,1]-LS_soil_loss[:,0])/((LS_soil_loss[:,1]+LS_soil_loss[:,0])/2);
            print (O_soil_loss)
            O_runoff = (LS_runoff[:,1]-LS_runoff[:,0])/((LS_runoff[:,1]+LS_runoff[:,0])/2);
            I = (Input[:,1]-Input[:,0])/((Input[:,1]+Input[:,0])/2);
            
            ALS_soil_loss = O_soil_loss/I;
            ALS_runoff = O_runoff/I;
            
            log_file.write("\n plot start.................")
                   
            y_pos = np.arange(len(objects))
            plt.bar(y_pos, ALS_runoff, align='center', color='blue',alpha=0.5)
            plt.xticks(y_pos, objects ,rotation=90)
            plt.ylabel('ALS')
            plt.title('Average Linear Sensitivity Plot for Runoff')
            
            plt.savefig(os.path.join(output_folder,"ALS_runoff.jpg"))
            plt.clf()
            
            
            plt.bar(y_pos, ALS_soil_loss, align='center',color='blue', alpha=0.5)
            plt.xticks(y_pos, objects, rotation=90)
            plt.ylabel('ALS')
            plt.title('Average Linear Sensitivity Plot for Soil Loss')
            
            plt.savefig(os.path.join(output_folder,'ALS_soil_loss.jpg'))
            plt.clf()
            
            log_file.write("\n plot done.................")

            progress_bar.setValue(100)
                
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
            
        




        
class Calibration_Run:

    def Calibration_Run(self,Start_year,End_year,dem_file,slop_file,flow_direction_file,flow_accumulation_file,soil_texture_file,land_use_folder,rain_fall_data_file_cal,boundry_file,paremeter_file,calibration_range_file,observed_file,output_folder,cb_arr, progress_bar):
        print("In Callibration_Run .......................")
        CB_data = cb_arr.split(",")
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
        
        def MMMF_Cal(x):
            main_table_filepath = paremeter_file
            
           
            LULC_folder= land_use_folder
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
            excel_to_csv(main_table_filepath,output_folder)
            excel_to_csv(rain_fall_data_file_cal,output_folder)
            excel_to_csv(observed_file,output_folder)
            #print("All csv files generated")
            #log_file.write("\nGenerated csv files...\n")
            #progress_bar.setValue(5)    
            #reading txt files as paramereters
            soil_para=csv_to_data(os.path.join(output_folder,"Soil.csv"))
            cover= csv_to_data(os.path.join(output_folder,"Landuse.csv"))
            soil_char= csv_to_data(os.path.join(output_folder,"Soil_char.csv"))
            flow_char=csv_to_data(os.path.join(output_folder,"Flow_char.csv"))
            other_para= csv_to_data(os.path.join(output_folder,"Other.csv"))

            x_MS=1;x_BD=1;x_LP=1;x_EHD=1; x_PI=1;x_ETO=1;x_CC=1;x_GC=1;x_PH=1;x_NV=1;x_D=1;

            x_Kc =float( soil_char[0][0])#Detachability of soil by raindrop for clay (g/J)
            x_Kz = float(soil_char[1][0])#Detachability of soil by raindrop for silt (g/J)
            x_Ks = float(soil_char[2][0])#Detachability of soil by raindrop for sand (g/J)
            x_DRc = float(soil_char[0][1])#Detachability of soil by runoff for clay (g/mm)
            x_DRz = float(soil_char[1][1])#Detachability of soil by runoff for silt (g/mm)
            x_DRs = float(soil_char[2][1])#Detachability of soil by runoff for sand (g/mm)
            x_VSc = float(soil_char[0][2])# particle fall velocity for clay(m/s)
            x_VSz = float(soil_char[1][2])# particle fall velocity for Silt(m/s)
            x_VSs = float(soil_char[2][2])# particle fall velocity for Sand(m/s)
    
        #changing variable according to argument x       
            nn=0;
            print("x is .................................")
            print(x)
            print("Check Box data is ....................")
            print(CB_data)
         #soil moisture content at field capacity    
            if 'MS' in CB_data:
                x_MS = x[nn]
                nn=nn+1;
                # CB.append[1]
              
         #bulk density (mg/m^3)               
            if 'BD' in CB_data:
                x_BD = x[nn]
                nn=nn+1;
                # CB.append[2]
                
        
            if 'LP' in CB_data:
                x_LP = x[nn]; 
                nn=nn+1;
                # CB.append[3]
              
            if 'EHD' in CB_data:
                x_EHD=x[nn];
                nn=nn+1;
                # CB.append[4]
                
            if 'PI' in CB_data:
                x_PI=x[nn];
                nn=nn+1;
                # CB.append[5]
             
            #ratio of actual to potential evapotranspiration
            if 'ETO' in CB_data:
                x_ETO = x[nn]
                nn=nn+1;
                # CB.append[6]
        
            if 'CC' in CB_data:
                x_CC = x[nn]
                nn=nn+1;
                # CB.append[7]
            
            if 'GC' in CB_data:
                x_GC = x[nn]
                nn=nn+1;
                # CB.append[8]
                
            if 'PH' in CB_data:
                x_PH =x[nn];
                nn=nn+1;
                # CB.append[9]
            
            if 'NV' in CB_data:
                x_NV=x[nn];
                nn=nn+1;
                # Cb.append[10]
            
        
            if 'D' in CB_data:
                x_D =x[nn];
                nn=nn+1;
                # CB.append[11]
            
            if 'VSc' in CB_data:
                x_VSc=x[nn]
                nn=nn+1;
                # CB.append[12]
        
            if 'VSz' in CB_data:
                x_VSz=x[nn];
                nn=nn+1;
                # CB.append[13]
            
            if 'VSs' in CB_data:
                x_VSs=x[nn];
                nn=nn+1;
                # CB.append[14]
        
                
            if 'Kc' in CB_data:
                x_Kc= x[nn]
                nn=nn+1;
                # CB.append[15]
            
            if 'Kz' in CB_data:
                x_Kz= x[nn]
                nn=nn+1;
                # CB.append[16]
        
            if 'Ks' in CB_data:
                x_Ks= x[nn]
                nn=nn+1;
                # CB.append[17]
            
            if 'DRc' in CB_data:
                x_DRc =  x[nn];
                nn=nn+1;
                # CB.append[18]
            
            if 'DRz' in CB_data:
                x_DRz =  x[nn];
                nn=nn+1;
                # Cb.append[19]
            
            if 'DRs' in CB_data:
                x_DRs =  x[nn];
                nn=nn+1;
                # CB.append[20]
            est_sl_mt=np.zeros(int(End_year)-int(Start_year)+1)
            est_q_mm=np.zeros(int(End_year)-int(Start_year)+1)
                
            for ii in range(int(End_year)-int(Start_year)+1):
                filename=(os.path.join(output_folder,str(ii+int(Start_year))+".csv"))
                rainfall_data= csv_to_data(filename)          
                (lulc,s4,r4,c4)= raster_read(os.path.join(LULC_folder, "Lulc_" + str(ii+int(Start_year)) + ".tif"))
               
                (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                est_q_mm[ii]=q_outlet_mm; #in mm
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
            RB = 2;
 
            progress_bar=progress_bar
            log_file=open(os.path.join(output_folder,"log_file.csv"),'w+')
            log_file.write(output_folder)
            log_file.write("\nTime,"+time.strftime("%H:%M:%S")+"\n")
            log_file.write("Date,"+time.strftime("%d/%m/%Y")+"\n\n")
            
            excel_to_csv(calibration_range_file,output_folder)
            Range = csv_to_data(os.path.join(output_folder,"Range.csv"))
            lb= np.zeros(len(CB))
            ub= np.zeros(len(CB))
            for i in range(len(CB)):
                lb[i] = Range[CB[i]-1,2]
                ub[i] = Range[CB[i]-1,3]
            
            progress_bar.setValue(0)
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
            wb1.save(os.path.join(output_folder,'Calibrated_values.xls'))
            progress_bar.setValue(100)
            
            
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
            

 
class Validation_Run:
    def Validation_Run(self,Start_year,End_year,dem_file,slop_file,flow_direction_file,flow_accumulation_file,soil_texture_file,land_use_folder,rain_fall_data_file_cal,boundry_file,paremeter_file,observed_file,calibrated_file,output_folder,progress_bar):
        
        print("In Validation Run ........................")
        class Error1(Exception):
           """This is a custom exception."""
           pass
        
        
        try:
            print("\nProgram starting... \n")
            progress_bar=progress_bar
            progress_bar.setValue(0)
            log_file=open(os.path.join(output_folder,"log_file.csv"),'w+')
            log_file.write(output_folder)
            log_file.write("\nTime,"+time.strftime("%H:%M:%S")+"\n")
            log_file.write("Date,"+time.strftime("%d/%m/%Y")+"\n\n")
            main_table_filepath=paremeter_file
            LULC_folder = land_use_folder
            (dem, s1,r1,c1)=raster_read(dem_file)
            (flowacc, s2,r2,c2)= raster_read(flow_accumulation_file)
            (flowdir, s3,r3,c3)= raster_read(flow_direction_file)
            progress_bar.setValue(20)
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
            excel_to_csv(main_table_filepath,output_folder)
            excel_to_csv(rain_fall_data_file_cal,output_folder)
            excel_to_csv(observed_file,output_folder)
            excel_to_csv(calibrated_file,output_folder)
            #print("All csv files generated")
            #log_file.write("\nGenerated csv files...\n")
            #progress_bar.setValue(5)    
            #reading txt files as paramereters
            soil_para=csv_to_data(os.path.join(output_folder,"Soil.csv"))
            cover= csv_to_data(os.path.join(output_folder,"Landuse.csv"))
            soil_char= csv_to_data(os.path.join(output_folder,"Soil_char.csv"))
            flow_char=csv_to_data(os.path.join(output_folder,"Flow_char.csv"))
            other_para= csv_to_data(os.path.join(output_folder,"Other.csv"))
            Callibrated_para= csv_to_data(os.path.join(output_folder,"Calibrated_values.csv"))
            print (Callibrated_para)
            CB= Callibrated_para[:,0]
            x=Callibrated_para[:,1]
            
            x_MS=1;x_BD=1;x_LP=1;x_EHD=1; x_PI=1;x_ETO=1;x_CC=1;x_GC=1;x_PH=1;x_NV=1;x_D=1;

            x_Kc =float( soil_char[0][0])#Detachability of soil by raindrop for clay (g/J)
            x_Kz = float(soil_char[1][0])#Detachability of soil by raindrop for silt (g/J)
            x_Ks = float(soil_char[2][0])#Detachability of soil by raindrop for sand (g/J)
            x_DRc = float(soil_char[0][1])#Detachability of soil by runoff for clay (g/mm)
            x_DRz = float(soil_char[1][1])#Detachability of soil by runoff for silt (g/mm)
            x_DRs = float(soil_char[2][1])#Detachability of soil by runoff for sand (g/mm)
            x_VSc = float(soil_char[0][2])# particle fall velocity for clay(m/s)
            x_VSz = float(soil_char[1][2])# particle fall velocity for Silt(m/s)
            x_VSs = float(soil_char[2][2])# particle fall velocity for Sand(m/s)
            nn=0;
         #soil moisture content at field capacity    
            if 1 in CB:
                x_MS = x[nn]
                nn=nn+1;
                # CB.append[1]
              
         #bulk density (mg/m^3)               
            if 2 in CB:
                x_BD = x[nn]
                nn=nn+1;
                # CB.append[2]
                
        
            if 3 in CB:
                x_LP = x[nn]; 
                nn=nn+1;
                # CB.append[3]
              
            if 4 in CB:
                x_EHD=x[nn];
                nn=nn+1;
                # CB.append[4]
                
            if 5 in CB:
                x_PI=x[nn];
                nn=nn+1;
                # CB.append[5]
             
            #ratio of actual to potential evapotranspiration
            if 6 in CB:
                x_ETO = x[nn]
                nn=nn+1;
                # CB.append[6]
        
            if 7 in CB:
                x_CC = x[nn]
                nn=nn+1;
                # CB.append[7]
            
            if 8 in CB:
                x_GC = x[nn]
                nn=nn+1;
                # CB.append[8]
                
            if 9 in CB:
                x_PH =x[nn];
                nn=nn+1;
                # CB.append[9]
            
            if 10 in CB:
                x_NV=x[nn];
                nn=nn+1;
                # Cb.append[10]
            
        
            if 11 in CB:
                x_D =x[nn];
                nn=nn+1;
                # CB.append[11]
            
            if 12 in CB:
                x_VSc=x[nn]
                nn=nn+1;
                # CB.append[12]
        
            if 13 in CB:
                x_VSz=x[nn];
                nn=nn+1;
                # CB.append[13]
            
            if 14 in CB:
                x_VSs=x[nn];
                nn=nn+1;
                # CB.append[14]
        
                
            if 15 in CB:
                x_Kc= x[nn]
                nn=nn+1;
                # CB.append[15]
            
            if 16 in CB:
                x_Kz= x[nn]
                nn=nn+1;
                # CB.append[16]
        
            if 17 in CB:
                x_Ks= x[nn]
                nn=nn+1;
                # CB.append[17]
            
            if 18 in CB:
                x_DRc =  x[nn];
                nn=nn+1;
                # CB.append[18]
            
            if 19 in CB:
                x_DRz =  x[nn];
                nn=nn+1;
                # Cb.append[19]
            
            if 20 in CB:
                x_DRs =  x[nn];
                nn=nn+1;
                
            progress_bar.setValue(30)
            
            observation_data= csv_to_data(os.path.join(output_folder,"Observation.csv"))
            #for i in range(1,len(observation_data)):
            obs_q_mm=np.transpose(observation_data[:,0])
            obs_sl_mt = np.transpose(observation_data[:,1])
            
            est_sl_mt=np.zeros(int(End_year)-int(Start_year)+1)
            est_q_mm=np.zeros(int(End_year)-int(Start_year)+1)
                
            for ii in range(int(End_year)-int(Start_year)+1):
                filename=(os.path.join(output_folder,str(ii+int(Start_year))+".csv"))
                rainfall_data= csv_to_data(filename)          
                (lulc,s4,r4,c4)= raster_read(os.path.join(LULC_folder, "Lulc_" + str(ii+int(Start_year)) + ".tif"))
               
                (qe,ke,f_total,h_total,ero_dep,sum_f_h,ero_dep,q_outl_m3, q_outlet_mm,sl_outl_MT)=MMMF(dem,Slope,rainfall_data,taluka,Soil_texture,soil_para,other_para,lulc,flowdir,flowacc,cover,soil_char,flow_char,output_folder,x_MS, x_BD, x_LP, x_EHD, x_PI,x_ETO,x_CC,x_GC,x_PH, x_NV, x_D,x_VSc, x_VSz, x_VSs, x_Kc, x_Kz,x_Ks, x_DRc,x_DRz, x_DRs)
                est_q_mm[ii]=q_outlet_mm; #in mm
                est_sl_mt[ii]=sl_outl_MT;
            
            
            progress_bar.setValue(90)
            error_q=((est_q_mm-obs_q_mm)/obs_q_mm)*100;
            NSE_q=1-( sum((obs_q_mm-est_q_mm)**2)/sum((obs_q_mm-np.mean(obs_q_mm))**2));
            PBIAS_q= (sum(est_q_mm-obs_q_mm)*100)/sum(obs_q_mm);
            RSR_q = np.sqrt(sum((obs_q_mm-est_q_mm)**2))/np.sqrt(sum((obs_q_mm-np.mean(obs_q_mm))**2));
            opt= 1-NSE_q; 
        
            error_sl=((est_sl_mt-obs_sl_mt)/obs_sl_mt)*100;
            NSE_sl=1-( sum((obs_sl_mt-est_sl_mt)**2)/sum((obs_sl_mt-np.mean(obs_sl_mt))**2));
            PBIAS_sl= (sum(est_sl_mt-obs_sl_mt)*100)/sum(obs_sl_mt);
            RSR_sl = np.sqrt(sum((obs_sl_mt-est_sl_mt)**2))/np.sqrt(sum((obs_sl_mt-np.mean(obs_sl_mt))**2));
            opt= 1-NSE_sl; 
            
            Parameter=['MS','BD','LP', 'EHD', 'per_incp','Et/Eo','CC','GC','PH','NV','D','VSc','VSs','VSz','Kc','Kz','Ks','DRc','DRz','DRs']
        
            wb1 = Workbook()
            sheet1 = wb1.add_sheet('Error_Results')
            
            
            sheet1.write(0, 0, 'Year') 
            sheet1.write(0, 1, 'Runoff') 
            sheet1.write(0, 2, 'Soilloss') 
            for ii in range(int(End_year)-int(Start_year)+1):
                sheet1.write(ii+1, 0, int(Start_year)+ii) 
                sheet1.write(ii+1, 1, error_q[ii]) 
                sheet1.write(ii+1, 2, error_sl[ii]) 
            
        
            sheet2 = wb1.add_sheet('Statistics')
            sheet2.write(1, 0, 'NSE') 
            sheet2.write(2, 0, 'PBIAS') 
            sheet2.write(3, 0, 'RSR') 
            sheet2.write(0, 1, 'Runoff') 
            sheet2.write(0, 2, 'Soilloss') 
            
            sheet2.write(1, 1, NSE_q) 
            sheet2.write(1, 2, NSE_sl) 
            sheet2.write(2, 1, PBIAS_q) 
            sheet2.write(2, 2, PBIAS_sl) 
            sheet2.write(3, 1, RSR_q) 
            sheet2.write(3, 2, RSR_sl) 
            
            
            wb1.save(os.path.join(output_folder,'Validation_results.xls'))
            
            #progress_bar.setValue(100)
            print("Finised Sucessfully")
            progress_bar.setValue(100)
            
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
    



class HydrologicalDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""


        super(HydrologicalDialog, self).__init__(parent)
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)

        self.dem_file = None
        self.slop_file = None
        self.flow_direction_file = None
        self.flow_accumulation_file = None
        self.land_use_file = None
        self.soil_texture_file = None
        self.rainfall_data_file = None
        self.boundry_file = None
        self.paremeter_file = None
        self.output_folder = None
        self.land_use_folder = None
        self.rain_fall_data_file_cal = None
        self.calibration_range_file = None
        self.observed_file = None
        self.calibrated_file = None

        #simple run 
        self.PB_SR_1.clicked.connect(self.dem)
        self.PB_SR_2.clicked.connect(self.slop)
        self.PB_SR_3.clicked.connect(self.flow_direction)
        self.PB_SR_4.clicked.connect(self.flow_accumulation)
        self.PB_SR_5.clicked.connect(self.land_use)
        self.PB_SR_6.clicked.connect(self.soil_texture)
        self.PB_SR_7.clicked.connect(self.rainfall_data)
        self.PB_SR_8.clicked.connect(self.boundry)
        self.PB_SR_9.clicked.connect(self.paremeter_file_call)
        self.PB_SR_10.clicked.connect(self.output_folder_call)

        #sensitivity
        self.PB_SA_1.clicked.connect(self.dem)
        self.PB_SA_2.clicked.connect(self.slop)
        self.PB_SA_3.clicked.connect(self.flow_direction)
        self.PB_SA_4.clicked.connect(self.flow_accumulation)
        self.PB_SA_5.clicked.connect(self.land_use)
        self.PB_SA_6.clicked.connect(self.soil_texture)
        self.PB_SA_7.clicked.connect(self.rainfall_data)
        self.PB_SA_8.clicked.connect(self.boundry)
        self.PB_SA_9.clicked.connect(self.paremeter_file_call)
        self.PB_SA_10.clicked.connect(self.output_folder_call)

        #calibration
        self.PB_CA_1.clicked.connect(self.dem)
        self.PB_CA_2.clicked.connect(self.slop)
        self.PB_CA_3.clicked.connect(self.flow_direction)
        self.PB_CA_4.clicked.connect(self.flow_accumulation)
        self.PB_CA_5.clicked.connect(self.soil_texture)
        self.PB_CA_6.clicked.connect(self.land_use_folder_call)
        self.PB_CA_7.clicked.connect(self.rain_fall_data_cal_call)
        self.PB_CA_8.clicked.connect(self.boundry)
        self.PB_CA_9.clicked.connect(self.paremeter_file_call)
        self.PB_CA_10.clicked.connect(self.calibration_range)
        self.PB_CA_11.clicked.connect(self.observed)
        self.PB_CA_12.clicked.connect(self.output_folder_call)

        #validation
        self.PB_VA_1.clicked.connect(self.dem)
        self.PB_VA_2.clicked.connect(self.slop)
        self.PB_VA_3.clicked.connect(self.flow_direction)
        self.PB_VA_4.clicked.connect(self.flow_accumulation)
        self.PB_VA_5.clicked.connect(self.soil_texture)
        self.PB_VA_6.clicked.connect(self.land_use_folder_call)
        self.PB_VA_7.clicked.connect(self.rain_fall_data_cal_call)
        self.PB_VA_8.clicked.connect(self.boundry)
        self.PB_VA_9.clicked.connect(self.paremeter_file_call)
        self.PB_VA_10.clicked.connect(self.observed)
        self.PB_VA_11.clicked.connect(self.calibrated)
        self.PB_VA_12.clicked.connect(self.output_folder_call)

#        self.PB_Output_SR_A.clicked.connect(self.Output_SR_A)
        self.PB_SR_MR.clicked.connect(self.SR_fn)
        self.PB_SA_MR.clicked.connect(self.Sensitivity_fn)
        self.MR_CA.clicked.connect(self.Calibration_fn)
        self.Cal_MR_A_3.clicked.connect(self.Val_fn)
        
# Modified MMF Annual (_A)  

    def Simulation_year_SR(self):
        return self.LE_SR_0.text()

    def Simulation_year_SA(self):
        return self.LE_SA_0.text()

    def dem(self):
        self.dem_file,_filter = QFileDialog.getOpenFileName()
        self.LE_SR_1.setText(self.dem_file)
        self.LE_SA_1.setText(self.dem_file)
        self.LE_CA_1.setText(self.dem_file)
        self.LE_VA_1.setText(self.dem_file)

    def slop(self):
        self.slop_file,_filter = QFileDialog.getOpenFileName()
        self.LE_SR_2.setText(self.slop_file)
        self.LE_SA_2.setText(self.slop_file)
        self.LE_CA_2.setText(self.slop_file)
        self.LE_VA_2.setText(self.slop_file)

    def flow_direction(self):
        self.flow_direction_file,_filter = QFileDialog.getOpenFileName()
        self.LE_SR_3.setText(self.flow_direction_file)
        self.LE_SA_3.setText(self.flow_direction_file)
        self.LE_CA_3.setText(self.flow_direction_file)
        self.LE_VA_3.setText(self.flow_direction_file)

    def flow_accumulation(self):
        self.flow_accumulation_file,_filter = QFileDialog.getOpenFileName()
        self.LE_SR_4.setText(self.flow_accumulation_file)
        self.LE_SA_4.setText(self.flow_accumulation_file)
        self.LE_CA_4.setText(self.flow_accumulation_file)
        self.LE_VA_4.setText(self.flow_accumulation_file)

    def land_use(self):
        self.land_use_file,_filter = QFileDialog.getOpenFileName()
        self.LE_SR_5.setText(self.land_use_file)
        self.LE_SA_5.setText(self.land_use_file)
        

    def land_use_folder_call(self):
        self.land_use_folder = QFileDialog.getExistingDirectory()
        self.LE_CA_6.setText(self.land_use_folder)
        self.LE_VA_6.setText(self.land_use_folder)


    def rain_fall_data_cal_call(self):
        self.rain_fall_data_file_cal,_filter = QFileDialog.getOpenFileName()
        self.LE_CA_7.setText(self.rain_fall_data_file_cal)
        self.LE_VA_7.setText(self.rain_fall_data_file_cal)

    def soil_texture(self):
        self.soil_texture_file,_filter = QFileDialog.getOpenFileName()
        self.LE_SR_6.setText(self.soil_texture_file)
        self.LE_SA_6.setText(self.soil_texture_file)
        self.LE_CA_5.setText(self.soil_texture_file)
        self.LE_VA_5.setText(self.soil_texture_file)

    def rainfall_data(self):
        self.rainfall_data_file,_filter = QFileDialog.getOpenFileName()
        self.LE_SR_7.setText(self.rainfall_data_file)
        self.LE_SA_7.setText(self.rainfall_data_file)
        

    def boundry(self):
        self.boundry_file,_filter = QFileDialog.getOpenFileName()
        self.LE_SR_8.setText(self.boundry_file)
        self.LE_SA_8.setText(self.boundry_file)
        self.LE_CA_8.setText(self.boundry_file)
        self.LE_VA_8.setText(self.boundry_file)

    def paremeter_file_call(self):
        self.paremeter_file,_filter = QFileDialog.getOpenFileName()
        self.LE_SR_9.setText(self.paremeter_file)
        self.LE_SA_9.setText(self.paremeter_file)
        self.LE_CA_9.setText(self.paremeter_file)
        self.LE_VA_9.setText(self.paremeter_file)

    def output_folder_call(self):
        self.output_folder = QFileDialog.getExistingDirectory()
        self.LE_SR_10.setText(self.output_folder)
        self.LE_SA_10.setText(self.output_folder)
        self.LE_CA_12.setText(self.output_folder)
        self.LE_VA_12.setText(self.output_folder)

    def calibration_range(self):
        self.calibration_range_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CA_10.setText(self.calibration_range_file)

    def observed(self):
        self.observed_file,_filter = QFileDialog.getOpenFileName()
        self.LE_CA_11.setText(self.observed_file)
        self.LE_VA_10.setText(self.observed_file)

    def calibrated(self):
        self.calibrated_file,_filter = QFileDialog.getOpenFileName()
        self.LE_VA_11.setText(self.calibrated_file)


    def SR_fn(self, parent=None):

        Simulation_Year = self.Simulation_year_SR()
        dem_file = self.dem_file
        slop_file = self.slop_file
        flow_direction_file = self.flow_direction_file
        flow_accumulation_file = self.flow_accumulation_file 
        land_use_file = self.land_use_file 
        soil_texture_file = self.soil_texture_file 
        rainfall_data_file = self.rainfall_data_file 
        boundry_file = self.boundry_file 
        paremeter_file = self.paremeter_file 
        output_folder = self.output_folder 
        progress_bar=self.PG_SR
        self.ero=Simple_Run()
        self.ero.Simple_Run(Simulation_Year,dem_file,slop_file,flow_direction_file,flow_accumulation_file,land_use_file,soil_texture_file,rainfall_data_file,boundry_file,paremeter_file,output_folder,progress_bar) 

    def Sensitivity_fn(self,parent=None):

        Simulation_Year = self.Simulation_year_SA()
        dem_file = self.dem_file
        slop_file = self.slop_file
        flow_direction_file = self.flow_direction_file
        flow_accumulation_file = self.flow_accumulation_file 
        land_use_file = self.land_use_file 
        soil_texture_file = self.soil_texture_file 
        rainfall_data_file = self.rainfall_data_file 
        boundry_file = self.boundry_file 
        paremeter_file = self.paremeter_file 
        output_folder = self.output_folder 
        cb_arr = ""
        if(self.RB_SA_MS.isChecked()):
            cb_arr+='MS,'
        if(self.RB_SA_BD.isChecked()):
            cb_arr+='BD,'
        if(self.RB_SA_LP.isChecked()):
            cb_arr+='LP,'
        if(self.RB_SA_EHD.isChecked()):
            cb_arr+='EHD,'
        if(self.RB_SA_PI.isChecked()):
            cb_arr+='PI,'
        if(self.RB_SA_ETO.isChecked()):
            cb_arr+='ETO,'
        if(self.RB_SA_NV.isChecked()):
            cb_arr+='NV,'
        if(self.RB_SA_D.isChecked()):
            cb_arr+='D,'
        if(self.RB_SA_VSc.isChecked()):
            cb_arr+='VSc,'
        if(self.RB_SA_GC.isChecked()):
            cb_arr+='GC,'
        if(self.RB_SA_PH.isChecked()):
            cb_arr+='PH,'
        if(self.RB_SA_CC.isChecked()):
            cb_arr+='CC,'
        if(self.RB_SA_Kc.isChecked()):
            cb_arr+='Kc,'
        if(self.RB_SA_VSs.isChecked()):
            cb_arr+='VSs,'
        if(self.RB_SA_Kz.isChecked()):
            cb_arr+='Kz,'
        if(self.RB_SA_VSz.isChecked()):
            cb_arr+='VSz,'
        if(self.RB_SA_all.isChecked()):
            cb_arr+='all,'
        if(self.RB_SA_Ks.isChecked()):
            cb_arr+='Ks,'
        if(self.RB_SA_DRc.isChecked()):
            cb_arr+='DRc,'
        if(self.RB_SA_DRz.isChecked()):
            cb_arr+='DRz,'
        if(self.RB_SA_DRs.isChecked()):
            cb_arr+='DRs,'

        progress_bar=self.PG_SA
        self.ero=Sensitivity_Run()
        self.ero.Sensitivity_Run(Simulation_Year,dem_file,slop_file,flow_direction_file,flow_accumulation_file,land_use_file,soil_texture_file,rainfall_data_file,boundry_file,paremeter_file,output_folder,cb_arr,progress_bar) 

    def Calibration_fn(self,parent=None):

        Start_Year = self.LE_CA_SY.text()
        End_Year = self.LE_CA_EY.text()
        dem_file = self.dem_file
        slop_file = self.slop_file
        flow_direction_file = self.flow_direction_file
        flow_accumulation_file = self.flow_accumulation_file
        soil_texture_file = self.soil_texture_file
        land_use_folder = self.land_use_folder
        rain_fall_data_file_cal = self.rain_fall_data_file_cal
        boundry_file = self.boundry_file
        paremeter_file = self.paremeter_file
        calibration_range_file = self.calibration_range_file
        observed_file = self.observed_file
        output_folder = self.output_folder

        cb_arr = ""
        if(self.RB_CA_MS.isChecked()):
            cb_arr+='MS,'
        if(self.RB_CA_BD.isChecked()):
            cb_arr+='BD,'
        if(self.RB_CA_LP.isChecked()):
            cb_arr+='LP,'
        if(self.RB_CA_EHD_2.isChecked()):
            cb_arr+='EHD,'
        if(self.RB_CA_PI_2.isChecked()):
            cb_arr+='PI,'
        if(self.RB_CA_ETO.isChecked()):
            cb_arr+='ETO,'
        if(self.RB_CA_NV_2.isChecked()):
            cb_arr+='NV,'
        if(self.RB_CA_D.isChecked()):
            cb_arr+='D,'
        if(self.RB_CA_VSc.isChecked()):
            cb_arr+='VSc,'
        if(self.RB_CA_GC_2.isChecked()):
            cb_arr+='GC,'
        if(self.RB_CA_PH_2.isChecked()):
            cb_arr+='PH,'
        if(self.RB_CA_CC.isChecked()):
            cb_arr+='CC,'
        if(self.RB_CA_KSc.isChecked()):
            cb_arr+='Kc,'
        if(self.RB_CA_VSs.isChecked()):
            cb_arr+='VSs,'
        if(self.RB_CA_KSz.isChecked()):
            cb_arr+='Kz,'
        if(self.RB_CA_VSz.isChecked()):
            cb_arr+='VSz,'
        if(self.RB_CA_all.isChecked()):
            cb_arr+='all,'
        if(self.RB_CA_KSs.isChecked()):
            cb_arr+='Ks,'
        if(self.RB_CA_DRc.isChecked()):
            cb_arr+='DRc,'
        if(self.RB_CA_DRz.isChecked()):
            cb_arr+='DRz,'
        if(self.RB_CA_DRs.isChecked()):
            cb_arr+='DRs,'

        progress_bar=self.PG_CA
        print(cb_arr)
        
        self.ero=Calibration_Run()
        self.ero.Calibration_Run(Start_Year,End_Year,dem_file,slop_file,flow_direction_file,flow_accumulation_file,soil_texture_file,land_use_folder,rain_fall_data_file_cal,boundry_file,paremeter_file,calibration_range_file,observed_file,output_folder,cb_arr,progress_bar)



    def Val_fn(self,parent=None):

        Start_Year = self.LE_VA_SY.text()
        End_Year = self.LE_VA_EY.text()
        dem_file = self.dem_file
        slop_file = self.slop_file
        flow_direction_file = self.flow_direction_file
        flow_accumulation_file = self.flow_accumulation_file
        soil_texture_file = self.soil_texture_file
        land_use_folder = self.land_use_folder
        rain_fall_data_file_cal = self.rain_fall_data_file_cal
        boundry_file = self.boundry_file
        paremeter_file = self.paremeter_file
        calibrated_file = self.calibrated_file
        observed_file = self.observed_file
        output_folder = self.output_folder
        progress_bar=self.PG_VA
        self.ero = Validation_Run()
        self.ero.Validation_Run(Start_Year,End_Year,dem_file,slop_file,flow_direction_file,flow_accumulation_file,soil_texture_file,land_use_folder,rain_fall_data_file_cal,boundry_file,paremeter_file,observed_file,calibrated_file,output_folder,progress_bar)




