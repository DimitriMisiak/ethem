#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sum up all the possibility of the ethem package first applied to the
nbsi_solo and nbsi_duo detectors.

@author: misiak
"""

# adding ethem module path to the pythonpath
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import csv

import ethem as eth

import config_red_toy as config
from config_red_toy import syst, evad

plt.close('all')

#==============================================================================
# PARAMETERS
#==============================================================================
L = 1
fs = 1e4

#==============================================================================
# IV PLOT
#==============================================================================
i_array = np.array([0.1, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]) * 1e-9
v_array = evad[config.load.resistivity] * i_array
t_array = np.linspace(0.015, 0.050, 10)


param_sym = (
        syst.Voltstat_b.voltage,
        syst.Thermostat_b.temperature
)

ss_point = eth.solve_sse_param(syst, param_sym, evad)

iv_dict = dict()
for p in tqdm.tqdm(t_array):

    v_list = list()
    for volt in v_array:

        sol = ss_point((volt, p))
        v_list.append(sol.x[-1])

    iv_array = np.array(v_list)

    iv_dict[p] = np.vstack((i_array, iv_array))


fig = plt.figure('Toy IV data')

keys_sorted = np.sort(list(iv_dict.keys()))

for t in keys_sorted:
    i_array, v_array = iv_dict[t]
    plt.errorbar(i_array, v_array, yerr=v_array*0.1,
                 lw=1., ls='-',
                 label='{0:.1f} mK'.format(t*1e3))

with open('output/lol.csv',mode='w') as f:
    f.write('# current I\tvoltageV\tTemperature T\n')

    w = csv.writer(f, delimiter='\t')
    for t in keys_sorted:

        for i,v in iv_dict[t].T:
            w.writerow([i, v, t])


plt.grid(True)
plt.ylabel('Capa voltage [V]')
plt.xlabel('Bias Voltage [V]')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='right')

plt.tight_layout()
plt.savefig('output/solve_sse_param.png')


