# -*- coding: utf-8 -*-
"""
Created on 0701 monday 2019

@author: zhuwq14
"""
import scipy as sp
import codecs
import random
import math
import numpy as np 
import os
import re
from multiprocessing import Pool 
from functools import partial
import sys

class DataCylinder:
    def __init__(self, Nx, Nr, Nq,var_number):
        self._Nx = Nx
        self._Nr = Nr
        self._Nq = Nq
        self._var_number = var_number
        self._xyz = sp.zeros([3, self._Nx,self._Nr,self._Nq])
        self._xlist = sp.zeros(self._Nx)
        self._rlist = sp.zeros(self._Nr)
        self._qlist = sp.zeros(self._Nq)
        self._vars = sp.zeros([9,Nx,Nr,Nq])
        # 0,1,2: for U, V, W; 3-5 for Reynolds stress
        self._Uandrms = sp.zeros([6,self._Nx,self._Nr,self._Nq])
        self._circum_averaged = sp.zeros([6,self._Nx,self._Nr])
        self._max_2D = sp.zeros([3,self._Nx])

    def establish_cylinder(self):
        for i_x in range(self._Nx):
            self._xlist[i_x] = 0.1*(i_x + 1)
        self._rlist[0] = 0.0
        Drdis = sp.zeros([self._Nr - 1])
        for i_r in range(1,self._Nr):
            if i_r <= 16:
                Drdis[i_r-1] = 0.02
                self._rlist[i_r] = self._rlist[i_r-1] + Drdis[i_r - 1]
            elif i_r >16 and i_r <=40:
                Drdis[i_r - 1] = Drdis[i_r - 2]/1.1
                self._rlist[i_r] = self._rlist[i_r-1] + Drdis[i_r - 1]
            elif i_r >40 and i_r <=64:
                Drdis[i_r - 1] = Drdis[i_r - 2]*1.1
                self._rlist[i_r] = self._rlist[i_r-1] + Drdis[i_r - 1]
            else:
                Drdis[i_r-1] = Drdis[i_r - 2]*1.06
                self._rlist[i_r] = self._rlist[i_r-1] + Drdis[i_r - 1] 
        for i_q in range(self._Nq):
            self._qlist[i_q] = 2.0*math.pi/self._Nq * i_q
        for i_x in range(self._Nx):
            self._xyz [0,i_x,:,:] = self._xlist[i_x]
            for i_q in range(self._Nq):
                for i_r in range(self._Nr):
                    self._xyz[1,i_x,i_r,i_q] = self._rlist[i_r]*math.cos(self._qlist[i_q])
                    self._xyz[2,i_x,i_r,i_q] = self._rlist[i_r]*math.sin(self._qlist[i_q])
    
    def transform(self):
        self._Uandrms[0,:,:,:] = self._vars[0,:,:,:]

        for i_q in range(self._Nq):
            for i_r in range(self._Nr):
                for i_x in range(self._Nx):
                    self._Uandrms[1,i_x,i_r,i_q] = self._vars[1,i_x,i_r,i_q]*math.cos(self._qlist[i_q]) + self._vars[2,i_x,i_r,i_q]*math.sin(self._qlist[i_q])
                    self._Uandrms[2,i_x,i_r,i_q] = self._vars[1,i_x,i_r,i_q]*math.sin(self._qlist[i_q]) - self._vars[2,i_x,i_r,i_q]*math.cos(self._qlist[i_q])
                    self._Uandrms[3,i_x,i_r,i_q] = math.sqrt(abs(self._vars[3,i_x,i_r,i_q]))
                    self._Uandrms[4,i_x,i_r,i_q] = math.sqrt(abs(self._vars[4,i_x,i_r,i_q]))*math.cos(self._qlist[i_q]) + math.sqrt(abs(self._vars[5,i_x,i_r,i_q]))*math.sin(self._qlist[i_q])
                    self._Uandrms[5,i_x,i_r,i_q] = self._vars[6,i_x,i_r,i_q]*math.cos(self._qlist[i_q]) + self._vars[7,i_x,i_r,i_q]*math.sin(self._qlist[i_q])
    
    def circum_average(self):
        self._circum_averaged = np.average(self._Uandrms,-1)

    def find_max(self):
        for i_x in range(self._Nx):
            self._max_2D[0:3,i_x] = self._circum_averaged[3:6,i_x,:].max()

    def save3Dfield(self,filename):
        var_names = 'Variables = "x", "y","z", "U", "V", "W", "urms", "vrms"\n'
        zone_name = ''.join(['ZONE T="Newmesh"\n'])
        zone_ijK = ' '.join(['I=', str(self._Nx), 'J=', str(self._Nr), 'K=', str(self._Nq), '\n'])
        datapacking = 'DATAPACKING = POINT\n'
        with open(filename, 'w') as fout:
            fout.write(var_names)
            fout.write(zone_name)
            fout.write(zone_ijK)
            fout.write(datapacking)
            for i_q in range(self._Nq):
                for i_r in range(self._Nr):
                    for i_x in range(self._Nx):
                        x_curr = self._xyz[0,i_x,i_r,i_q]
                        y_curr = self._xyz[1,i_x,i_r,i_q]
                        z_curr = self._xyz[2,i_x,i_r,i_q]
                        u_curr = self._vars[0,i_x,i_r,i_q]
                        v_curr = self._vars[1,i_x,i_r,i_q]
                        w_curr = self._vars[2,i_x,i_r,i_q]
                        urms_curr = self._vars[3,i_x,i_r,i_q]
                        vrms_curr = self._vars[4,i_x,i_r,i_q]
                        fout.write('{} {} {} {} {} {} {} {}\n' .format(x_curr, y_curr, z_curr,u_curr,v_curr,w_curr,urms_curr,vrms_curr))
        print("Saving 3D completed")
  
    def outputprofile(self,filename):
        var_names = 'Variables = "x", "r", "U", "V", "W", "urms", "vrms", "uv"\n'
        zone_name = ''.join(['ZONE T="averaged"\n'])
        zone_ijK = ' '.join(['I=', str(self._Nx), 'J=', str(self._Nr),'\n'])
        datapacking = 'DATAPACKING = POINT\n'
        with open(filename, 'w') as fout:
            fout.write(var_names)
            fout.write(zone_name)
            fout.write(zone_ijK)
            fout.write(datapacking)
            for i_r in range(self._Nr):
                r_curr = self._rlist[i_r]
                for i_x in range(self._Nx):
                    x_curr = self._xlist[i_x]
                    var_curr = self._circum_averaged[:,i_x,i_r]
                    fout.write('{} {} {} {} {} {} {} {}\n' .format(x_curr, r_curr, var_curr[0],var_curr[1],var_curr[2],var_curr[3],var_curr[4],var_curr[5]))
        print("Saving 2Daveraged completed")

    def outputcenterprofile(self,filename):
        var_names = 'Variables = "x", "urms_peak", "vrms_peak", "uv_peak"\n'
        zone_name = ''.join(['ZONE T="Reynolds_stress"\n'])
        with open(filename,'w') as fout:
            fout.write(var_names)
            fout.write(zone_name)
            for i_x in range(self._Nx):
                x_curr = self._xlist[i_x]
                data = self._max_2D[:,i_x]
                fout.write('{} {} {} {}\n'.format(x_curr,data[0],data[1],data[2]))
        print("Saving centerline completed")
        


def read_inputdata(filename,var_number):
    head_line = var_number + 6
    with codecs.open(filename,'r','utf-8') as f:
        count = 1
        scurr = f.readline()
        while scurr and count < head_line:
            count += 1
            if count == var_number + 5:
                
                ne_list = re.findall(r"\d+",scurr)
                
                node_number = int(ne_list[0])
                element_number = int(ne_list[1])
                
                inputdata = sp.zeros([node_number, var_number])
                cell_info = sp.zeros([element_number,4], dtype = 'int')
            scurr = f.readline()
        while scurr:
            scurr = f.readline()
            count += 1
            if count > head_line and count <= head_line + node_number:
                index_curr = count - head_line - 1
                var_currline = scurr.split(" ",var_number+1)
                for ivar in range(var_number):
                    inputdata[index_curr,ivar] = float(var_currline[ivar+1])
            elif count > head_line + node_number and count <= head_line + node_number + element_number:
                index_curr = count - head_line - node_number -1        
                var_currline = scurr.split(" ",5)
                for ivertex in range(4):
                    cell_info[index_curr,ivertex] = int(var_currline[ivertex+1])
    data_used = sp.zeros([node_number,9])
    xyz_original = sp.zeros([node_number,3])
    xyz_original = inputdata[:,0:3]
    data_used[:,0:3] = inputdata[:,3:6]
    data_used[:,3:] = inputdata[:,13:19] + inputdata[:,19:25]
    return element_number, xyz_original,data_used, cell_info

def search_index(yz_target, element_number, xyz_ori, cell_info, MESH_SCALE):
    # search the location of every xrq point in the original data
    S_difference_min = 10000.0
    for i_m in range(element_number):
        point_index = cell_info[i_m,:] - 1
        yz_point_1 = xyz_ori[point_index[0],1:3]
        l1 = yz_point_1 - yz_target
        distance = math.sqrt(l1.dot(l1))
        if distance <= MESH_SCALE:
            yz_point_2 = xyz_ori[point_index[1],1:3]
            yz_point_3 = xyz_ori[point_index[2],1:3]
            yz_point_4 = xyz_ori[point_index[3],1:3]
            l2 = yz_point_2 - yz_target
            l3 = yz_point_3 - yz_target
            l4 = yz_point_4 - yz_target
            S_part1 = 0.5*abs((l1[0]*l4[1] - l1[1]*l4[0]))
            S_part2 = 0.5*abs((l1[0]*l2[1] - l1[1]*l2[0]))
            S_part3 = 0.5*abs((l2[0]*l3[1] - l2[1]*l3[0]))
            S_part4 = 0.5*abs((l3[0]*l4[1] - l3[1]*l4[0]))

            cross1 = yz_point_1 - yz_point_3
            cross2 = yz_point_2 - yz_point_4
            S_all = 0.5*abs(cross1[0]*cross2[1] - cross1[1]*cross2[0])

            S_difference = S_part1 + S_part2 + S_part3 + S_part4 - S_all
            if S_difference < S_difference_min:
                S_difference_min = S_difference
                cell_target = i_m
        else:
            S_difference_min = 0.0
    if S_difference_min > 0.0001:
        sys.exit()
        print("The target cell can not be found")

    return cell_target

def cal_coefficient(element_number,xyz_ori,inputdata,cell_info,var_used,search_range):
    coefficient = sp.zeros([3,var_used,element_number])
    for i_m in range(element_number):
        point_index = cell_info[i_m,:] - 1
        # the point index from grid begin from 1, not 0
        yz_point_1 = xyz_ori[point_index[0],1:3]
        yz_point_2 = xyz_ori[point_index[1],1:3]
        yz_point_3 = xyz_ori[point_index[2],1:3]
        yz_point_4 = xyz_ori[point_index[3],1:3]

        # find the distance from center. the far-field is not calculated
        r_curr = math.sqrt(yz_point_1.dot(yz_point_1))
        if r_curr > search_range:
            coefficient[:,:,i_m] = 0.0
        else:
            dyz_1 = yz_point_2 - yz_point_1
            dyz_2 = yz_point_3 - yz_point_1
            dyz_3 = yz_point_4 - yz_point_1
            Acoefficient = sp.zeros([3,3])

            Acoefficient[0,0] = dyz_1[0]*dyz_1[0]
            Acoefficient[0,1] = dyz_1[1]*dyz_1[1]
            Acoefficient[0,2] = dyz_1[0]*dyz_1[1]

            Acoefficient[1,0] = dyz_2[0]*dyz_2[0]
            Acoefficient[1,1] = dyz_2[1]*dyz_2[1]
            Acoefficient[1,2] = dyz_2[0]*dyz_2[1]

            Acoefficient[2,0] = dyz_3[0]*dyz_3[0]
            Acoefficient[2,1] = dyz_3[1]*dyz_3[1]
            Acoefficient[2,2] = dyz_3[0]*dyz_3[1]
            dvar = np.array([0.0,0.0,0.0])
        
            for i_var in range(var_used):
                dvar[0] = inputdata[point_index[1],i_var] - inputdata[point_index[0],i_var]
                dvar[1] = inputdata[point_index[2],i_var] - inputdata[point_index[0],i_var]
                dvar[2] = inputdata[point_index[3],i_var] - inputdata[point_index[0],i_var]

                coefficient[:,i_var,i_m] = np.linalg.solve(Acoefficient,dvar)
    return coefficient

def InterpolationProcess(i_x, Nr, Nq, yz_newmesh, var_number, var_used, search_range1, search_range2,floder_path):
    new_vars = sp.zeros([var_used,Nr,Nq])
    filename = ''.join([floder_path,'/Slice_',str(i_x+1),'.dat'])
    element_number, xyz_ori, data_point , cell_info = read_inputdata(filename,var_number)
    interpolation_coefficient = cal_coefficient(element_number,xyz_ori,data_point,cell_info,var_used,search_range1)
    for i_q in range(Nq):
        print(filename,i_q)
        for i_r in range(Nr):
            yz_curr = yz_newmesh[:,i_r,i_q]
            cell_target = search_index(yz_curr,element_number,xyz_ori,cell_info,search_range2)
            point_index = cell_info[cell_target,:] - 1
            dyz = yz_curr - xyz_ori[point_index[0],1:3]
            dr = np.array([dyz[0]*dyz[0], dyz[1]*dyz[1], dyz[0]*dyz[1]])
            for i_var in range(var_used):
                coeff_curr = interpolation_coefficient[:,i_var,cell_target]
                new_vars[i_var,i_r,i_q] = data_point[point_index[0],i_var] + np.dot(coeff_curr,dr)
    filename_out = ''.join([floder_path,'/D_xslice_',str(i_x+1),'.dat'])
    savenewmesh(filename_out,yz_newmesh,new_vars,Nr,Nq)
    file_binary = ''.join([floder_path,'/B_xslice_',str(i_x+1)])
    np.save(file_binary,new_vars)
    
def savenewmesh(filename,xy_new, var_new,Nr,Nq):
    var_names = 'Variables = "y", "z", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", v"9" \n'
    zone_name = ''.join(['ZONE T="interpolation"\n'])
    zone_ijK = ' '.join(['I=', str(Nr), 'J=', str(Nq),'\n'])
    datapacking = 'DATAPACKING = POINT\n'
    with open(filename, 'w') as fout:
        fout.write(var_names)
        fout.write(zone_name)
        fout.write(zone_ijK)
        fout.write(datapacking)
        for i_q in range(Nq):
            for i_r in range(Nr):
                y_curr = xy_new[0,i_r,i_q]
                z_curr = xy_new[1,i_r,i_q]

                var_curr = var_new[:,i_r,i_q]
                fout.write('{} {} {} {} {} {} {} {} {} {} {}\n' .format(y_curr, z_curr, var_curr[0],var_curr[1],var_curr[2],var_curr[3],var_curr[4],var_curr[5],var_curr[6],var_curr[7],var_curr[8]))
    print("Saving newmesh2D completed")   

def loadbinary(file_binary):
    data = np.load(file_binary)
    return data


if __name__ == '__main__':
    DATABINARY = False
    floder = 'Slice_Jetexit'
    floder_path = os.path.join('./',floder)
    number_xslice = 1
    var_number = 26
    var_used = 9

    #demension of new mesh
    Nx = number_xslice
    Nr = 100
    Nq = 60

    search_range1 = 3.5
    search_range2 = 0.4

    NewMesh = DataCylinder(Nx,Nr,Nq,var_used)
    NewMesh.establish_cylinder()
    if DATABINARY:
        for i_x in range(Nx):
            filename = ''.join([floder_path,'/B_xslice_',str(i_x+1),'.npy'])
            NewMesh._vars[:,i_x,:,:] = loadbinary(filename)
    else:
        yz_newmesh = NewMesh._xyz[1:3,0,:,:]
        x_vector = sp.zeros([Nx], dtype=int)
        for i_x in range(Nx):
            x_vector[i_x] = i_x
    
        partial_func = partial(InterpolationProcess, Nr=Nr, Nq=Nq, yz_newmesh=yz_newmesh, var_number=var_number, var_used=var_used, search_range1 = search_range1, search_range2 = search_range2, floder_path=floder_path)
        pool = Pool(6)
        pool.map(partial_func, x_vector)
        pool.close()
        pool.join()

        for i_x in range(Nx):
            filename = ''.join([floder_path,'/B_xslice_',str(i_x+1),'.npy'])
            NewMesh._vars[:,i_x,:,:] = loadbinary(filename)

    NewMesh.transform()
    NewMesh.circum_average()
    NewMesh.find_max()

    filename = 'ThreeD_field.plt'
    NewMesh.save3Dfield(filename)

    filename = 'Average_2D.plt'
    NewMesh.outputprofile(filename)
    filename = 'Stress_max.plt'
    NewMesh.outputcenterprofile(filename)












