import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from amuse.units import units, constants
from amuse.datamodel import Particles, Particle, ParticlesSuperset

from overflow_fraction import *

def plot_spinup_a(par, df, cons, varname, values, su):
    fig, ax = plt.subplots(figsize = (6,4), dpi=600, layout='constrained')
    
    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']

    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax.set_xlabel('$a$ [AU]', **props)
    ax.set_ylabel(r'$\Delta v_{rot} / v_{crit}$', **props)
    for i in range(len(values)):
        c = next(color)
        ax.plot(par, df.iloc[:, [i+1]], color=c, label=varname+' = {:=04.2f}'.format(values[i]))
    ax.legend(loc=2, handlelength=1, **props)
    ax.text(1.7, 0.5e-5, cons, alpha=0.5, **props)
    

    plt.savefig('./plots/a_dependence_parameter'+su+'.png')
    plt.close()


def plot_spinup_e(ta, ta_lin, tv, tv_lin, acon, vcon, su):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,4), dpi=600, sharey=True)
    
    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    
    ### Left (a) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax1.set_xlabel('$e$', **props)
    ax1.set_ylabel(r'$\Delta v_{rot} / v_{crit}$', **props)
    for a in ta.columns[2:]:
        c = next(color)
        ax1.plot(ta['e'], ta[a], color=c, label = r'$a$ = '+a+' AU')
        ax1.plot(ta_lin['e'], ta_lin[a], color=c, linestyle='dashed')
    ax1.legend(loc=(0.06, 0.765), handlelength=1, **props)
    ax1.text(0.05, 0.15e-5, r'$v_{extra} / v_{per}$ = '+vcon, alpha=0.5, **props)
    
    ### Right (vfr) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax2.set_xlabel('$e$', **props)
    for v in tv.columns[2:]:
        c = next(color)
        ax2.plot(tv['e'], tv[v], color=c, label = r'$v_{extra} / v_{per}$ = '+v)
        ax2.plot(tv_lin['e'], tv_lin[v], color=c, linestyle='dashed')
    ax2.legend(loc=(0.06, 0.675), handlelength=1, **props)
    ax2.text(0.05, 0.15e-5, r'$a$ = '+acon+' AU', alpha=0.5, **props)
    
    plt.subplots_adjust(left=0.09, right=0.99, top=0.95, wspace=0)
    plt.savefig('./plots/e_dependence_2'+su+'.png')
    plt.close()

def convergence(table_list, frac_i, racc, mc, mtr, yn, su):
    fig, axis = plt.subplots(figsize = (4,3), dpi=600, layout='constrained')
    props = {'fontsize': 11}
    styles = ['dotted', 'solid', 'dashed']
    labels = ['n = 100', 'n = 1k', 'n = 5k']
    for i in range(len(table_list)):
        df_store = pd.DataFrame()
        t = table_list[i]
        table = pd.read_table(t, header=None, names=['filenames'])
        L_list, dv_list, par_list, mg_list = [[], [], [], []]
        for f in table['filenames']:
            macc, mdon, a, e, v_fr, v_extra, df = read_file('./data/'+t.split('.')[0]+'/'+f)
            df_frac = add_fraction(df, frac_i, yn)
            T = 2 * np.pi * np.sqrt((a**3) / (constants.G * (macc + mdon)))
            
            dm_l, dm_g, p_t, p_r, p, L_tot = momentum_mtr(df_frac, racc, T, mtr)
            
            dv = L_tot / (racc * (macc + dm_g))
            v_crit = (constants.G * (macc + dm_g) / racc)**0.5
            L_list.append(L_tot.value_in(units.kg * units.m**2 * units.s**(-1)))
            dv_list.append(dv/v_crit)
            mg_list.append(dm_g/dm_l)
            par_list.append(e)
        df_store['spinup'] = dv_list
    #     axis.plot(par_list, dv_list, color='k', linewidth=1, linestyle=styles[i], label=labels[i])
        df_store.to_csv('spinup_convergence_'+labels[i].split(' ')[-1]+'.csv')
    # axis.set_xlabel('e', **props)
    # axis.set_ylabel(r'$\Delta v_{rot} / v_{crit}$', **props)
    
    # axis.legend(loc='best', **props)
    # axis.set_ylim(bottom=-0.01e-5)
    # axis.set_xlim(left=0.5)
    # plt.savefig('./plots/n_spinup_comp'+su+'.png')
    

def L_elements(theta, v, f, L, filename):
    a = float(filename.split('/')[-1].split('mdon_')[-1].split('a_')[0].replace('_', '.'))
    e = float(filename.split('/')[-1].split('a_')[-1].split('e_')[0].replace('_', '.'))

    #L = [i/max(L) for i in L]
    
    fig, axis = plt.subplots(figsize = (4,3), dpi=600, layout='constrained')
    props = {'fontsize': 11}

    axis.plot(theta, v, color='k', linestyle='dotted', label='v')
    #axis.plot(theta, f, color='k', linestyle='dashed', label='f')
    #axis.plot(theta, L, color='k', linestyle='solid', label='L')

    axis.legend(loc='best', **props)
    axis.set_xlim(left=np.pi)
    axis.set_xlabel('true anomaly [rad]')

    axis.set_title(f'a = {a} AU, e = {e}')

    plt.savefig('./plots/momentum_truean_'+filename.split('mdon_')[-1].split('_-')[0]+'.png')

def L_elements_comp(df, parname, su):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (4,6), dpi=600, layout='constrained', sharex=True)
    props = {'fontsize': 11}

    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))

    for i in range(1, df.shape[1], 2):
        print(i)
        c = next(color)
        label = parname + ' = ' + df.columns[i].split('f')[0]

        maxs_top = []
        maxs_bot = []
        maxs_top.append(df.iloc[:, [i]].max().iloc[0])
        maxs_bot.append(df.iloc[:, [i+1]].max().iloc[0])
        ax1.plot(df['theta i [rad]'], df.iloc[:, [i]], label=label, color=c)
        ax2.plot(df['theta i [rad]'], df.iloc[:, [i+1]], label=label, color=c)
    
    ax1.set_ylabel(r'$f( \theta )$', **props)
    ax1.legend(loc=0, **props)
    ax1.set_ylim(bottom=0, top=1.05*max(maxs_top))
    ax1.set_xlim(left=np.pi)

    ax2.set_xlabel('true anomaly [rad]', **props)
    ax2.set_ylabel(r'$v_t$ [km s-1]', **props)
    ax2.set_ylim(bottom=0, top=1.05*max(maxs_bot))

    plt.savefig('./plots/L_elements'+su+'.png')

def velocities(df, parname, su):
    fig, ax = plt.subplots(figsize = (4,4), dpi=600, layout='constrained', sharex=True)
    props = {'fontsize': 11}

    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))

    for i in range(1, df.shape[1], 2):
        print(i)
        c = next(color)
        label = parname + ' = ' + df.columns[i].split('f')[0]

        maxs_top = []
        maxs_bot = []
        maxs_top.append(df.iloc[:, [i]].max().iloc[0])
        maxs_bot.append(df.iloc[:, [i+1]].max().iloc[0])
        ax.plot(df['theta i [rad]'], df.iloc[:, [i+1]], label=label, color=c)
        print(label+'\n')
        print(df.loc[df[df.columns[i+1]] > 0].iloc[:, [i+1]].idxmin())

    ax.set_xlabel('true anomaly [rad]', **props)
    ax.set_ylabel(r'$v_t$ [km s-1]', **props)
    #ax.set_ylim(bottom=400, top=1.05*max(maxs_bot))
    ax.set_ylim(bottom=0, top=1.05*max(maxs_bot))
    #ax.set_xlim(left=5.5)
    ax.set_xlim(left=np.pi)
    ax.legend(loc='best')

    plt.savefig('./plots/impact_velocity'+su+'.png')

def q_plot(df_plot, table_list, racc, mtr, frac_i, yn, su):
    if (df_plot == None) & (table_list != None):
        df_plot = pd.DataFrame()
        flag = 0
        for t in table_list:
            table = pd.read_table(t, header=None, names=['filenames'])
            L_list, dv_list, par_list, q_list = [[], [], [], []]
            for f in table['filenames']:
                macc, mdon, a, e, v_fr, v_extra, df = read_file('./data/'+t.split('.')[0]+'/'+f)
                df_frac = add_fraction(df, frac_i, yn)
                
                T = 2 * np.pi * np.sqrt((a**3) / (constants.G * (macc + mdon)))
                mu = (macc * mdon) / (macc + mdon)
                L_orb = (mu * 2 * np.pi * a**2) / T

                dm_l, dm_g, p_t, p_r, p, L_tot = momentum_mtr(df_frac, racc, T, mtr)
                
                dv = L_tot / (racc * (macc + dm_g))
                v_crit = (constants.G * (macc + dm_g) / racc)**0.5
                L_list.append(L_tot.value_in(units.kg * units.m**2 * units.s**(-1)))
                dv_list.append(dv/v_crit)
                par_list.append(a.value_in(units.au))

            if flag == 0:
                df_plot['a'] = par_list
                f = 1
        
            colname = '{:04.2f}'.format(float(t.split('vexp_')[1].split('q')[0].replace('_', '.')))
            print(t, colname)
            df_plot[colname] = dv_list

    df_plot.to_csv('q_table.csv')
    
    fig, axis = plt.subplots(figsize = (6,4), dpi=600, layout='constrained')
    
    props = {'fontsize': 11}
    
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    for i in range(1, df_plot.shape[1]):
        c = next(color)
        label = r'$q$'+' = {:=04.2f}'.format(float(df_plot.columns[i]))
        
        axis.plot(df_plot.iloc[:, 0], df_plot.iloc[:, i], label=label, color=c)

        axis.set_xlabel(r'$a$ [AU]', **props)
        axis.set_ylabel(r'$\Delta v_{rot} / v_{crit}$', **props)

    node_str = r'$e$'+' = {:=04.2f}, '.format(e)+r'$v_{extra}/v_{per}$'+' = {:=04.2f}'.format(v_fr)
    axis.text(2.2, 0.6e-5, node_str, alpha=0.5, **props)
    axis.legend(loc=2, **props)

    plt.savefig('./plots/q_comparison'+su+'.png')
    plt.close()

if __name__ == "__main__":
    '''
    ta = pd.read_csv('./data/e_a_0.90_table2.csv')
    ta_lin = pd.read_csv('./data/e_a_0.90_table_lin2.csv')
    tv = pd.read_csv('./data/e_vfr_1.80_table2.csv')
    tv_lin = pd.read_csv('./data/e_vfr_1.80_table_lin2.csv')
    con1 = '0.90'
    con2 = '1.80'
    plot_spinup_e(ta, ta_lin, tv, tv_lin, con2, con1, '')
    '''
    #convergence(['e_001_800a_0_90vfr_000_00vexp_100.dat', 'e_001_800a_0_90vfr_000_00vexp_1000.dat', 'e_001_800a_0_90vfr_000_00vexp_5000.dat'], 0.1, 1 | units.RSun, None, 10**(-4) | units.MSun * units.yr**(-1), 'n', '')
    q_plot(None, ['a_00_1000e_0_90vfr_000_00vexp_0_50q.dat', 'a_00_1000e_0_90vfr_000_00vexp_0_75q.dat', 'a_00_1000e_0_90vfr_000_00vexp_1_00q.dat', 'a_00_1000e_0_90vfr_000_00vexp_1_25q.dat', 'a_00_1000e_0_90vfr_000_00vexp_1_50q.dat'], 1 | units.RSun, 10**(-4) | units.MSun * units.yr**(-1), 0.1, 'n', '')