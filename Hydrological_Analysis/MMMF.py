# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 16:04:45 2019

@author: Pratiksha
"""

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
   