# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:48:35 2019
Sum of residual of each block
@author: zhuwq14
"""
import codecs
import re
import scipy as sp
NumBlock = 98
Line_his = 90
Number_cell = 0
averged_residual = sp.zeros([Line_his,16])
for iblock in range(NumBlock):
    index = str(iblock + 1)
    name_dimension = 'grd\dimension_blk' + index.zfill(3) + '_m.dat'
    with codecs.open(name_dimension,'r') as f:
        scurr = f.readline()
        scurr = f.readline()
        scurr = f.readline()
        
        ne_list = re.findall(r"\d+",scurr)
        NI = int(ne_list[0])
        NJ = int(ne_list[1])
        NK = int(ne_list[2])
        Ncell = NI*NJ*NK    
    Number_cell += Ncell
    
    name_his = 'His\HYBRIDSST' + index.zfill(3) + '.his'
    hisdata = sp.loadtxt(name_his)
    averged_residual += hisdata*Ncell


averged_residual = averged_residual/Number_cell
filename_out = 'All_residual.dat'
sp.savetxt(filename_out, averged_residual)   
        