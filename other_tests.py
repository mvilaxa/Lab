import os
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles, Particle, ParticlesSuperset
from amuse.units.optparse import OptionParser

from overflow_fraction import read_file

def max_e(table_name):
    print(table_name.split('.')[0])
    table = pd.read_table(table_name, header=None, names=['filenames'])
    max_e = []
    for f in table['filenames']:
        macc, mdon, a, e, v_fr, v_extra, df = read_file('./data/'+table_name.split('.')[0]+'/'+f)
        max_e.append(df.loc[df['flag impact'] == 0.0]['e p'].max())
        #print(df['e p'].max(), df.loc[df['e p'] == df['e p'].max()]['flag impact'])
        sl = df.loc[(df['flag impact'] == 0.0) & (df['e p'] >= 0.99)]
        if sl.shape[0] > 0:
            print('    '+f+'    {}/{}'.format(sl.shape[0], df.loc[df['flag impact'] == 0.0].shape[0]))
    #print(max_e)

if __name__ == "__main__":
    max_e('e_001_800a_0_80vfr_000_00vexp.dat')