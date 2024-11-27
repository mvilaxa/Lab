import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from amuse.units import units, constants
from amuse.units.optparse import OptionParser
from matplotlib.colors import LinearSegmentedColormap

def equation(macc, mdon, racc, a, r_per, r):
    t1 = (constants.G * mdon) / (r_per - r)**2
    t2 = (constants.G * macc) / (r**2)
    t3 = (2 * constants.G * macc) / ((r**2) + ((r**3) / racc))
    eq = t1 - t2 + t3
    return eq

def solve_eq(macc, mdon, racc, e, vfr):
    #guess1 = np.linspace(racc.value_in(units.m), (3 | units.au).value_in(units.m), 1000) | units.m
    guess1 = np.linspace(racc.value_in(units.au), 4, 1000) | units.au
    values1 = []
    for i in range(len(guess1)):
        a = ( ((1+e)/(1-e))
            * (guess1[i] + (guess1[i]**2 / racc))
            * (1 - vfr)**2
            * ((macc + mdon) / (2 * macc)) )
        r_per = a * (1 - e)
        eq = equation(macc, mdon, racc, a, r_per, guess1[i])
        #print(eq.value_in(units.m * units.s**-2))
        values1.append(np.abs(eq))
        if (i > 0) & (eq.value_in(units.m * units.s**-2) < 0):
            break
    sorted1 = sorted(zip(values1, guess1), key = lambda x: x[0])
    #print('\n\n')
    guess2 = np.linspace(sorted1[0][1].value_in(units.au), sorted1[1][1].value_in(units.au), 100) |units.au
    values2 = []
    for j in guess2:
        a = ( ((1+e)/(1-e))
            * (j + (j**2 / racc))
            * (1 - vfr)**2
            * ((macc + mdon) / (2 * macc)) )
        r_per = a * (1 - e)
        eq = equation(macc, mdon, racc, a, r_per, j)
        values2.append(np.abs(eq))
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

def peak_a_e(macc, mdon, racc):
    data = pd.read_table('1_SeBa_radius_short.data', sep="\t", skiprows=1, header=None, index_col=False,
                         names=['M [MSun]', 'M wd [MSun]', 'R ms [RSun]', 'R hg [RSun]', 'R rgb [RSun]', 'R hb [RSun]', 'R agb [RSun]'])
    r_agb = data.iloc[(data['M [MSun]'] - mdon.value_in(units.MSun)).abs().argsort()[:1]].iloc[0]['R agb [RSun]'] | units.RSun
    '''
    amax = a_max_guess(macc, mdon, v_fr, r_agb)
    amin = 1.01 * (r_agb + racc)
    a_range = np.linspace(amin.value_in(units.au), amax.value_in(units.au), 200) | units.au
    '''
    vfr = 0.9
    
    e_range = np.linspace(0., 0.95, 200)
    apeak_list = []
    for e in e_range:
        r = solve_eq(macc, mdon, racc, e, vfr)
        apeak = ( ((1+e)/(1-e))
                * (r + (r**2 / racc))
                * (1 - vfr)**2
                * ((macc + mdon) / (2 * macc)) )
        apeak_list.append(apeak.value_in(units.au))
    
    return e_range, apeak_list
    
def peak_a_v(macc, mdon, racc):
    data = pd.read_table('1_SeBa_radius_short.data', sep="\t", skiprows=1, header=None, index_col=False,
                         names=['M [MSun]', 'M wd [MSun]', 'R ms [RSun]', 'R hg [RSun]', 'R rgb [RSun]', 'R hb [RSun]', 'R agb [RSun]'])
    r_agb = data.iloc[(data['M [MSun]'] - mdon.value_in(units.MSun)).abs().argsort()[:1]].iloc[0]['R agb [RSun]'] | units.RSun
    '''
    amax = a_max_guess(macc, mdon, v_fr, r_agb)
    amin = 1.01 * (r_agb + racc)
    a_range = np.linspace(amin.value_in(units.au), amax.value_in(units.au), 200) | units.au
    '''
    e = 0.1
    
    v_range = np.linspace(0.775, 0.95, 200)
    apeak_list = []
    for vfr in v_range:
        r = solve_eq(macc, mdon, racc, e, vfr)
        apeak = ( ((1+e)/(1-e))
                * (r + (r**2 / racc))
                * (1 - vfr)**2
                * ((macc + mdon) / (2 * macc)) )
        apeak_list.append(apeak.value_in(units.au))
    #print(apeak_list)
    return v_range, apeak_list

def plot_a_peak(macc, mdon, racc):
    #colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    #cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    #color = iter(cmap(np.linspace(0, 1, 5)))
    fig, axis = plt.subplots(figsize = (4,5), dpi=350, layout='constrained')
    
    e, peak = peak_a_e(macc, mdon, racc)
    axis.plot(e, peak, color='firebrick')
    
    axis.set_xlabel(r'$e$')
    axis.set_ylabel(r'$a_{peak}$ [AU]')
    
    #axis.legend(loc='best')
    
    plt.savefig('./plots/'+'peak_a_e.png')
    
def plot_a_peak_2(macc, mdon, racc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (5,4), dpi=350, sharey=True)

    props = {'fontsize': 11}

    e, peak1 = peak_a_e(macc, mdon, racc)
    ax1.plot(e, peak1, color='black')
    
    ax1.set_xlim(left=0, right=1.15)

    ax1.set_xlabel(r'$e$', **props)
    ax1.set_ylabel(r'$a_{peak}$ [AU]', **props)

    vfr, peak2 = peak_a_v(macc, mdon, racc)
    ax2.plot(vfr, peak2, color='black')
    
    ax2.set_ylim(bottom=0, top=max(peak2))
    ax2.set_xlim(left=min(vfr), right=1.03)

    ax2.set_xlabel(r'$v_{extra} / v_{per}$', **props)
    
    ax1.text(0.05, max(peak2)*0.9, r'$v_{extra} / v_{per}$ = 0.90', alpha=0.5, **props)
    ax2.text(0.79, max(peak2)*0.9, r'$e$ = 0.10', alpha=0.5, **props)
    
    ax1.vlines(1, 0, max(peak2), linestyle='dashed', color='black')
    ax2.vlines(1, 0, max(peak2), linestyle='dashed', color='black')
    ax1.fill_between([0, 1.15], 1.24, 2.31, color='silver')
    ax2.fill_between([0, 1.03], 1.24, 2.31, color='silver')

    plt.subplots_adjust(wspace=0)
    
    plt.savefig('./plots/'+'peak_a_2.png')

    plt.close()

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

if __name__ == "__main__":
    o, arguments  = new_option_parser().parse_args()

    #plot_a_peak(o.macc, o.mdon, o.racc)
    plot_a_peak_2(o.macc, o.mdon, o.racc)
