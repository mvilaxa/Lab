import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from amuse.units import units, constants
from amuse.units.optparse import OptionParser
from matplotlib.colors import LinearSegmentedColormap

def equation(macc, mdon, racc, a, r_per, r):
    t1 = (constants.G * mdon) / (r_per - r)**2
    t2 = (constants.G * macc) / (r**2)
    #t3 = (r + (r**2)/racc) * (constants.G * (macc + mdon) / a) / (r * 2 * a)
    t3 = (2 * constants.G * macc) / ((r**2) + ((r**3) / racc))
    eq = t1 - t2 + t3
    return eq

def solve_eq(macc, mdon, racc, a, r_per):
    guess1 = np.linspace(racc.value_in(units.m), r_per.value_in(units.m)-1, 1000) | units.m
    values1 = []
    for i in guess1:
        eq = equation(macc, mdon, racc, a, r_per, i)
        values1.append(np.abs(eq))
    sorted1 = sorted(zip(values1, guess1), key = lambda x: x[0])
    
    guess2 = np.linspace(sorted1[0][1].value_in(units.m), sorted1[1][1].value_in(units.m), 100) |units.m
    values2 = []
    for j in guess2:
        eq = equation(macc, mdon, racc, a, r_per, j)
        values2.append(eq)
        sorted2 = sorted(zip(values2, guess2), key = lambda x: x[0])
    return sorted2[0][1]

def new_eq(macc, mdon, r_per, vi, r):
    t1 = (constants.G * mdon) / (r_per - r)**2
    t2 = (constants.G * macc) / (r**2)
    #t3 = (r + (r**2)/racc) * (constants.G * (macc + mdon) / a) / (r * 2 * a)
    t3 = (vi**2) / r
    eq = t1 - t2 + t3
    return eq

def solve_new_eq(macc, mdon, racc, r_per, v_per):
    guess1 = np.linspace(0, 1, 100, endpoint=False)
    values1 = []
    for i in guess1:
        vi = v_per * (1 - i)
        rL1 = (-racc + (racc**2 + (8 * constants.G * macc * racc / vi**2))**0.5) / 2
        eq = new_eq(macc, mdon, r_per, vi, rL1)
        values1.append(np.abs(eq))
    sorted1 = sorted(zip(values1, guess1), key = lambda x: x[0])
    
    guess2 = np.linspace(sorted1[0][1], sorted1[1][1], 100)
    values2 = []
    for j in guess2:
        vi = v_per * (1 - j)
        rL1 = (-racc + (racc**2 + (8 * constants.G * macc * racc / vi**2))**0.5) / 2
        eq = new_eq(macc, mdon, r_per, vi, rL1)
        values2.append(eq)
        sorted2 = sorted(zip(values2, guess2), key = lambda x: x[0])
    return sorted2[0][1]

def eq_a_max(acc_mass, don_mass, rdon_max, v, sma):
    eq = ( constants.G * ( don_mass / ((rdon_max)**2 ))
           - constants.G * ( acc_mass / (sma - rdon_max)**2 )
           + (v**2/(sma - rdon_max)) )
    
    return eq

def a_max_guess(macc, mdon, vfr, rdon_max):
    guess1 = np.linspace(1.01 * rdon_max.value_in(units.m), 10 * rdon_max.value_in(units.m), 1000) | units.m
    values1 = []
    for i in guess1:
        v = (1 - vfr) * np.sqrt(constants.G * (macc + mdon) * ((2/(i - rdon_max)) - (1/i)))
        eq = eq_a_max(macc, mdon, rdon_max, v, i)
        values1.append(np.abs(eq))
    sorted1 = sorted(zip(values1, guess1), key = lambda x: x[0])

    guess2 = np.linspace(sorted1[0][1].value_in(units.m), sorted1[1][1].value_in(units.m), 100) |units.m
    values2 = []
    for j in guess2:
        v = (1 - vfr) * np.sqrt(constants.G * (macc + mdon) * ((2/(j - rdon_max)) - (1/j)))
        eq = eq_a_max(macc, mdon, rdon_max, v, j)
        values2.append(eq)
        sorted2 = sorted(zip(values2, guess2), key = lambda x: x[0])
    
    return sorted2[0][1]

def L1_eq(acc_mass, don_mass, sma, r, v):
    eq = ( constants.G * ( don_mass / ((sma-r)**2 ))
           - constants.G * ( acc_mass / (r**2) )
           + (v**2/r) )
    return eq

def distance_to_L1(acc_mass, don_mass, sma, v):
    guess1 = np.linspace(1000, sma.value_in(units.m)-1, 1000) | units.m
    values1 = []
    for i in guess1:
        eq = L1_eq(acc_mass, don_mass, sma, i, v)
        values1.append(np.abs(eq))
    sorted1 = sorted(zip(values1, guess1), key = lambda x: x[0])

    guess2 = np.linspace(sorted1[0][1].value_in(units.m), sorted1[1][1].value_in(units.m), 100) |units.m
    values2 = []
    for j in guess2:
        eq = L1_eq(acc_mass, don_mass, sma, j, v)
        values2.append(eq)
        sorted2 = sorted(zip(values2, guess2), key = lambda x: x[0])
    return sorted2[0][1]

def vfr_limit_a(macc, mdon, racc, e):
    data = pd.read_table('1_SeBa_radius_short.data', sep="\t", skiprows=1, header=None, index_col=False,
                         names=['M [MSun]', 'M wd [MSun]', 'R ms [RSun]', 'R hg [RSun]', 'R rgb [RSun]', 'R hb [RSun]', 'R agb [RSun]'])
    r_agb = data.iloc[(data['M [MSun]'] - mdon.value_in(units.MSun)).abs().argsort()[:1]].iloc[0]['R agb [RSun]'] | units.RSun
    '''
    amax = a_max_guess(macc, mdon, v_fr, r_agb)
    amin = 1.01 * (r_agb + racc)
    a_range = np.linspace(amin.value_in(units.au), amax.value_in(units.au), 200) | units.au
    '''
    a_range = np.linspace(1., 3., 200) | units.au
    vfr_list = []
    for a in a_range:
        r_per = a * (1 - e)
        r = solve_eq(macc, mdon, racc, a, r_per)
        #vfr = 1 - np.sqrt((2 * a * macc) / ((macc + mdon) * (r + (r**2)/racc)))
        vfr = 1 - np.sqrt((2 * macc) / ((macc + mdon) * (r + (r**2)/racc) * ((2/r_per) - (1/a))))
        vfr_list.append(vfr)
    
    return a_range, vfr_list

def new_vfr_limit_a(macc, mdon, racc, e):
    data = pd.read_table('1_SeBa_radius_short.data', sep="\t", skiprows=1, header=None, index_col=False,
                         names=['M [MSun]', 'M wd [MSun]', 'R ms [RSun]', 'R hg [RSun]', 'R rgb [RSun]', 'R hb [RSun]', 'R agb [RSun]'])
    r_agb = data.iloc[(data['M [MSun]'] - mdon.value_in(units.MSun)).abs().argsort()[:1]].iloc[0]['R agb [RSun]'] | units.RSun
    
    a_range = np.linspace(1., 3., 200) | units.au
    vfr_list = []
    for a in a_range:
        r_per = a * (1 - e)
        v_per = (constants.G * (macc + mdon) * ((2 / r_per) - (1 / a)))**0.5
        vfr = solve_new_eq(macc, mdon, racc, r_per, v_per)
        vfr_list.append(vfr)
    
    return a_range, vfr_list

def vfr_limit_e(macc, mdon, racc, a):
    data = pd.read_table('1_SeBa_radius_short.data', sep="\t", skiprows=1, header=None, index_col=False,
                         names=['M [MSun]', 'M wd [MSun]', 'R ms [RSun]', 'R hg [RSun]', 'R rgb [RSun]', 'R hb [RSun]', 'R agb [RSun]'])
    r_agb = data.iloc[(data['M [MSun]'] - mdon.value_in(units.MSun)).abs().argsort()[:1]].iloc[0]['R agb [RSun]'] | units.RSun
    
    e_range = np.linspace(0.0, 0.9, 20, endpoint=False)
    vfr_list = []
    rL1_list = []
    for e in e_range:
        r_per = a * (1 - e)
        r = solve_eq(macc, mdon, racc, a, r_per)
        vfr = 1 - np.sqrt((2 * macc) / ((macc + mdon) * (r + (r**2)/racc) * ((2/r_per) - (1/a))))
        vfr_list.append(vfr)
        rL1_list.append(r)
    return e_range, vfr_list, rL1_list

def hut(e):
    t1 = 1 + (15/2)*(e**2) + (45/8)*(e**4) + (5/16)*(e**6)
    t2 = (1 + (3 * e**2) + (3/8)*(e**4)) * (1 + e)**2
    omegan = t1 / t2
    return omegan

def plot_for_e(macc, mdon, racc):
    e_list = [0.0, 0.05, 0.1, 0.3, 0.6]
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    fig, axis = plt.subplots(figsize = (4,5), dpi=350, layout='constrained')
    for e in e_list:
        print(e)
        c = next(color)
        label = 'e = {:.2f}'.format(e)
        a, vfr = vfr_limit_a(macc, mdon, racc, e)
        if e == 0.0:
            axis.plot(a.value_in(units.au), vfr, label=label, marker='*', markersize=7, markevery=(0, 30), linewidth=0.8, color=c)
        elif e == 0.05:
            axis.plot(a.value_in(units.au), vfr, label=label, marker='o', markersize=5, markevery=(10, 30), linewidth=0.8, color=c)
        elif e == 0.1:
            axis.plot(a.value_in(units.au), vfr, label=label, marker='v', markersize=5, markevery=(20, 30), linewidth=0.8, color=c)
        else:
            axis.plot(a.value_in(units.au), vfr, label=label, color=c) #, linestyle=style)
    
    axis.set_xlabel(r'$a$ [AU]')
    axis.set_ylabel(r'min($v_{extra}/v_{per}$)')
    
    axis.legend(loc='best')
    
    plt.savefig('./plots/'+'vfr_limit_vs_a.png')

def plot_quad(macc, mdon, racc):
    e_list = [0.0, 0.05, 0.1, 0.3, 0.6]
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    fig, axis = plt.subplots(figsize = (4,5), dpi=350, layout='constrained')
    for e in e_list:
        print(e)
        c = next(color)
        label = 'e = {:.2f}'.format(e)
        a, vfr = new_vfr_limit_a(macc, mdon, racc, e)
        if e == 0.0:
            axis.plot(a.value_in(units.au), vfr, label=label, marker='*', markersize=7, markevery=(0, 30), linewidth=0.8, color=c)
        elif e == 0.05:
            axis.plot(a.value_in(units.au), vfr, label=label, marker='o', markersize=5, markevery=(10, 30), linewidth=0.8, color=c)
        elif e == 0.1:
            axis.plot(a.value_in(units.au), vfr, label=label, marker='v', markersize=5, markevery=(20, 30), linewidth=0.8, color=c)
        else:
            axis.plot(a.value_in(units.au), vfr, label=label, color=c) #, linestyle=style)
    
    axis.set_xlabel(r'$a$ [AU]')
    axis.set_ylabel(r'min($v_{extra}/v_{per}$)')
    
    axis.legend(loc='best')
    
    plt.savefig('./plots/'+'vfr_limit_vs_a_quad.png')

def plot_hut(macc, mdon, racc):
    a_list = [1.0, 2.0, 3.0]
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    fig, axis = plt.subplots(figsize = (4,5), dpi=350, layout='constrained')
    for a in a_list:
        c = next(color)
        label = 'a = {:.2f} AU'.format(a)
        e, vfr, rL1 = vfr_limit_e(macc, mdon, racc, a | units.au)
        
        rL1 = [x.value_in(units.RSun) for x in rL1] | units.RSun
        rper = (a | units.au) * (1 - e)
        
        my_omega = vfr * rper / (rper - rL1)
        
        axis.plot(e, my_omega, color=c, linestyle='dashed') #, label=label
        
    unity = 1 + (macc / mdon)**0.5
    axis.plot(e, np.ones(len(e)) * unity, color='k', linestyle='dotted') #, label=r'$v_{extra}/v_{per} = 1$'
    axis.text(0.7, 1.85, r'$\frac{v_{extra}}{v_{per}} = 1$', fontsize='medium', color='k')
    
    axis.plot([], [], color='k', linestyle='dashed', label=r'min($\Omega$)')
    axis.text(0.0, 1.59, r'$a = 1$ AU', fontsize='medium', color='firebrick')
    axis.text(0.0, 1.67, r'$a = 2$ AU', fontsize='medium', color='gold')
    axis.text(0.0, 1.775, r'$a = 3$ AU', fontsize='medium', color='limegreen')
    
    hut_omega = hut(e)
    axis.plot(e, hut_omega, color='k', linestyle='solid') #, label='Hut (1981)'
    axis.text(0.7, 0.85, r'$\Omega_{ps}$', fontsize='medium')
    
    axis.set_xlabel(r'$e$')
    axis.set_ylabel(r'$\Omega / n_p$')
    
    axis.legend(loc='best')
    
    plt.savefig('./plots/'+'omega_limit_vs_e.png')

def new_option_parser():
    result = OptionParser()
    result.add_option("--macc", unit=units.MSun,
                      dest="macc", type="float",
                      default = 1.,
                      help="accretor mass")
    result.add_option("--racc", unit=units.RSun,
                      dest="racc", type="float",
                      default = 1.,
                      help="accretor radius")
    result.add_option("--mdon", unit=units.MSun,
                      dest="mdon", type="float",
                      default = 1.2,
                      help="donor mass")
    return result

o, arguments  = new_option_parser().parse_args()

#plot_for_e(o.macc, o.mdon, o.racc)
#plot_hut(o.macc, o.mdon, o.racc)
plot_quad(o.macc, o.mdon, o.racc)
