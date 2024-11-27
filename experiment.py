import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from amuse.units import units, constants
from amuse.units.optparse import OptionParser

###
def plot_kepler(df_p, df_don, racc, theta_i, macc, mdon):
    fig, axis = plt.subplots(figsize = (7,7), dpi=120)
    
    # Particle orbit
    axis.plot(df_p['r [AU]'] * np.cos(df_p['theta [rad]'] - df_p.iloc[0]['theta [rad]'] + theta_i),
              df_p['r [AU]'] * np.sin(df_p['theta [rad]'] - df_p.iloc[0]['theta [rad]'] + theta_i),
              color='deepskyblue', linewidth=1, label='parcel trajectory', zorder=1)
    
    # L1 orbit
    d_L1 = []
    for i in df_don['r [AU]']:
        d_L1.append(distance_to_L1(macc, mdon, i | units.au).value_in(units.au))
    axis.plot(d_L1 * np.cos(df_don['theta [rad]']),
              d_L1 * np.sin(df_don['theta [rad]']),
              color='k', linewidth=1, linestyle='dashed', label='L1 orbit', zorder=1)
    
    # Donor orbit
    axis.plot(df_don['r [AU]'] * np.cos(df_don['theta [rad]']),
              df_don['r [AU]'] * np.sin(df_don['theta [rad]']),
              color='k', linewidth=1, label='donor orbit', zorder=1)
    
    # Accretor star
    accretor = plt.Circle((0,0), racc.value_in(units.au), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
    accretor_edge = plt.Circle((0,0), racc.value_in(units.au), color='k', fill=False, linewidth=1)
    axis.add_patch(accretor)
    axis.add_patch(accretor_edge)
    
    axis.set_xlabel(r'$r_x$ [AU]')
    axis.set_ylabel(r'$r_y$ [AU]')
    
    axis.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    axis.set_xlim([-1.1*df_don['r [AU]'].max(), 1.1*df_don['r [AU]'].max()])
    axis.set_ylim([-1.1*df_don['r [AU]'].max(), 1.1*df_don['r [AU]'].max()])

    plt.savefig('./plots/aaaa_kepler_plot.png')
    
    ### ZOOM-IN ###
    
    figz, axisz = plt.subplots(figsize = (7,7), dpi=120)
    conv = (1 | units.au).value_in(units.RSun)
    
    # Particle orbit
    axisz.plot(conv * df_p['r [AU]'] * np.cos(df_p['theta [rad]'] - df_p.iloc[0]['theta [rad]'] + theta_i),
               conv * df_p['r [AU]'] * np.sin(df_p['theta [rad]'] - df_p.iloc[0]['theta [rad]'] + theta_i),
               color='deepskyblue', linewidth=1, label='parcel trajectory', zorder=1)
              
    # Accretor star
    accretor = plt.Circle((0,0), racc.value_in(units.RSun), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
    accretor_edge = plt.Circle((0,0), racc.value_in(units.RSun), color='k', fill=False, linewidth=1)
    axisz.add_patch(accretor)
    axisz.add_patch(accretor_edge)
    
    axisz.set_xlabel(r'$r_x$ [$R_{\odot}$]')
    axisz.set_ylabel(r'$r_y$ [$R_{\odot}$]')
    
    axisz.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    axisz.set_xlim([-2*racc.value_in(units.RSun), 2*racc.value_in(units.RSun)])
    axisz.set_ylim([-2*racc.value_in(units.RSun), 2*racc.value_in(units.RSun)])
    
    plt.savefig('./plots/aaaa_zoom_kepler_plot.png')
###

def initial_parameters(a_min, a_max, n):
    a_array = np.linspace(a_min, a_max, n)
    e_array = np.linspace(0, 1, n, endpoint=False)
    data = {'a [AU]': np.ones(n**2), 'e': np.ones(n**2)}
    df = pd.DataFrame(data=data)
    i = 0
    for a in a_array:
        for e in e_array:
            df.iloc[i] = [a, e]
            i += 1
    return df

def orbit_data(a, e, mtot, T, tau, n):
    '''
    Returns trajectory, velocity and it's angle as tangent to the orbit for a set of initial orbital parameters
    :a: Orbit's semi-major axis
    :e: Orbit's eccentricity
    :mtot: System's total mass (m1 + m2)
    :T: Orbital period
    :tau: 
    :n: Number of datapoints in orbit
    '''
    times = np.linspace(0.0, (T/2).value_in(units.day), int(n/2))
    
    true_an = []
    for t in times:
        dif = 1.0
        M = 2*np.pi*(t - tau)/(T.value_in(units.day))
        E0 = M
        while dif > 1e-10:
            E = M + e * np.sin(E0)
            dif = E - E0
            E0 = E
        theta = 2 * np.arctan(np.sqrt((1 + e)/(1 - e)) * np.tan(E/2))
        if theta >= 0:
            true_an.append(theta)
        else:
            true_an.append((2*np.pi)+theta)
    bottom_half = [2 * np.pi - i for i in true_an[::-1]]
    true_an = true_an + bottom_half
    r = (a * (1 - e**2)) / (1 + e*np.cos(true_an))
    v_2 = constants.G * mtot * ((2 / r) - (1 / a))
    
    data = {'theta [rad]': true_an,
            'r [AU]': r.value_in(units.au),
            'v [km s-1]': np.sqrt(v_2.value_in(units.km**2 * units.s**(-2))),
            'angle [rad]': np.ones(n)}
    df = pd.DataFrame(data=data)
    
    # Angle of velocity (tangent to the orbit)
    for i in df.index.tolist():
        # Angle between the lines conecting the point and each foci
        aux_ang = np.arcsin(np.sin(df.iloc[i]['theta [rad]']) * 2 * (a.value_in(units.au) * e) / (2 * a.value_in(units.au) - df.iloc[i]['r [AU]']))
        df.iloc[i]['angle [rad]'] = (aux_ang + np.pi) / 2
    return df

def L1_eq(acc_mass, don_mass, sma, r, v):
    eq = ( constants.G * ( don_mass / ((sma-r)**2 ))
           - constants.G * ( acc_mass / (r**2) )
           + (v**2/r) )
    return eq

# Computes the distance from the accretor's center to the L1 point of the system    
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

def get_new_orbital_elements(v_i, v_angle, r, macc):
    a = ((2 / r) - ((v_i**2) / (constants.G * macc)))**(-1)
    c = 0.5 * np.sqrt(r**2 + (2*a - r)**2 - 2*r*(2*a -r)*np.cos(np.pi - 2*v_angle))
    e = c / a
    theta_i = np.pi - np.arcsin(((2*a - r) / (2*c)) * np.sin(np.pi - 2*v_angle))
    return a, e, theta_i

def impact_kepler(df, racc):
    #print('ALL:\n', df, len(df))
    #print('INSIDE STAR:', df[df['r [AU]'] < racc.value_in(units.au)], len(df[df['r [AU]'] < racc.value_in(units.au)]))
    #print(len(df[df['r [AU]'] < racc.value_in(units.au)]))
    if len(df[df['r [AU]'] < racc.value_in(units.au)]) == 0:
        ang = 0
        v = 0
        pre_impact = df
        flag = 0
    else:
        impact_idx = df[df['r [AU]'] < racc.value_in(units.au)].index[0]
        pre_impact = df.iloc[:impact_idx+1]
        
        #print('PRE-IMPACT\n', pre_impact, len(pre_impact))
        
        x = (pre_impact['r [AU]'].iloc[-1] * np.cos(pre_impact['theta [rad]'].iloc[-1]) - pre_impact['r [AU]'].iloc[-2] * np.cos(pre_impact['theta [rad]'].iloc[-2]))
        y = (pre_impact['r [AU]'].iloc[-1] * np.sin(pre_impact['theta [rad]'].iloc[-1]) - pre_impact['r [AU]'].iloc[-2] * np.sin(pre_impact['theta [rad]'].iloc[-2]))
        
        ang = (pre_impact.iloc[-1]['angle [rad]'] + pre_impact.iloc[-2]['angle [rad]']) / 2
        v = (pre_impact['v [km s-1]'].iloc[-1] + pre_impact['v [km s-1]'].iloc[-2]) / 2
        flag = 1
    
    return pre_impact, ang, v | units.km / units.s, flag

def impact_new(ap, ep, racc, macc):
    '''
    Finds angle and velocity of the parcel at impact with the accretor's surface
    :ap: Semi-major axis of parcel's orbit
    :ep: Eccentricity of parcel's orbit
    :racc: Radius of accretor star
    :macc: Mass of accretor star
    '''
    peri = ap * (1 - ep)
    if peri.value_in(units.RSun) <= racc.value_in(units.RSun):
        # True anomaly (in the parcel's orbit) where impact occurs (r = racc)
        #theta_imp = np.arccos(((ap * (1 - ep**2) / racc) - 1) / ep)
        theta_imp = 2*np.pi - np.arccos(((ap * (1 - ep**2) / racc) - 1) / ep)
        
        # Angle between the lines from the point in the orbit to each foci
        # aux_ang = np.arcsin((np.sin(np.pi - theta_imp) * 2 * ap * ep) / (2 * ap - racc))
        aux_ang = np.arccos((4*ap**2 - 4*ap*racc + 2*racc**2 - 4*(ap*ep)**2)/(4*ap*racc - 2*racc**2))
        ang_imp = (np.pi - aux_ang) / 2
        v_imp = np.sqrt(constants.G * macc * ((2 / racc) - (1 / ap)))
        flag = 1
    else:
        ang_imp = 0
        v_imp = 0 | units.km * units.s**-1
        flag = 0
    
    return ang_imp, v_imp, flag
    

def get_particle_data(macc, mdon, racc, r_accdon, v_orb, ang_orb, true_an_i, v_extra, v_exp, ndat):
    #d_L1 = distance_to_L1(macc, mdon, r_accdon)
    #print('r_L1 = ', d_L1.value_in(units.au), ' AU')
    v_sqr = v_orb**2 + v_extra**2 - 2 * v_orb * np.sqrt(v_extra**2) * np.cos((np.pi / 2) - ang_orb)
    if abs(v_sqr.value_in(units.km**2 * units.s**-2)) <= 1e-10:
        v_sqr = 0 | units.km**2 * units.s**-2
    else:
        v_sqr = v_sqr
    
    v_fs = v_sqr**0.5 # !!! Run a=3, e=0.1 and print
    
    if v_fs.value_in(units.km * units.s**(-1)) == 0.:
        ang_fs = 0.
    else:
        #Addition or substraction given by the sign on v_extra
        ang_fs = ang_orb + np.arcsin(np.sin((np.pi / 2) - ang_orb) * v_extra / v_fs)
        
    if v_exp is None:
        v_i = v_fs
        ang_v = ang_fs
    else:
        v_i = np.sqrt(v_fs**2 + v_exp**2 - 2 * v_fs * np.sqrt(v_exp**2) * np.cos(np.pi - ang_fs))
        #Expansion is always pointed towards the accretor (therefore the - sign)
        ang_v = ang_fs - np.arcsin(np.sin(np.pi - ang_fs) * v_exp / v_i)
    
    d_L1 = distance_to_L1(macc, mdon, r_accdon, v_i)
    
    if v_i.value_in(units.km * units.s**(-1)) == 0.:    # Straight line
        ang_v = 0.
        # Orbital elements of particle orbit
        a_p, e_p, theta_i = get_new_orbital_elements(v_i, ang_v, d_L1, macc)
        e_p = np.inf
        theta_i = np.pi
        flag = 1.0
        v_impact = np.sqrt(constants.G * macc * ((2 / racc) - (1 / a_p)))
        ang_impact = 0.
        
    else:
        # Orbital elements of particle orbit
        a_p, e_p, theta_i = get_new_orbital_elements(v_i, ang_v, d_L1, macc)
        
        if a_p.value_in(units.m) < 0 or e_p > 1:
            ang_impact = 0
            v_impact = 0 | units.km * units.s**(-1)
            flag = 0
        else:
            '''
            T_p = 2 * np.pi * np.sqrt((a_p**3) / (constants.G * macc))
            E = 2 * np.arctan(np.sqrt((1 - e_p)/(1 + e_p)) * np.tan(theta_i / 2))
            tau = - (T_p/(2*np.pi)) * (E - e_p * np.sin(E))
            
            df = orbit_data(a_p, e_p, macc, T_p, tau.value_in(units.day), 2*ndat)
            pre_impact, ang_impact, v_impact, flag = impact_kepler(df, racc)
            '''
            ang_impact, v_impact, flag = impact_new(a_p, e_p, racc, macc)
            
    return d_L1, v_i, ang_v, a_p, e_p, theta_i, flag, v_impact, ang_impact

def ang_velocity(macc, mdon, r):
    return np.sqrt(constants.G * (macc + mdon) / r**3)

def get_table_for_system(macc, mdon, racc, a, e, v_fr, v_rot, v_exp, n, dirname):
    # Creating Filename
    macc_str = '{:=05.2f}macc'.format(macc.value_in(units.MSun)).replace('.', '_')
    mdon_str = '_{:=05.2f}mdon'.format(mdon.value_in(units.MSun)).replace('.', '_')
    a_str = '_{:=09.4f}a'.format(a.value_in(units.au)).replace('.', '_')
    e_str = '_{:=07.4f}e'.format(e).replace('.', '_')
    vfr_string = '_{:=07.4f}vfr'.format(v_fr).replace('.', '_')
    if v_rot.value_in(units.km * units.s**(-1)) < 0:
        v_rot_str = '_{:=07.2f}rot'.format(v_rot.value_in(units.km * units.s**(-1))).replace('.', '_')
    elif v_rot.value_in(units.km * units.s**(-1)) > 0:
        v_rot_str = '_+{:=06.2f}rot'.format(v_rot.value_in(units.km * units.s**(-1))).replace('.', '_')
    elif v_rot.value_in(units.km * units.s**(-1)) == 0:
        v_rot_str = '_+000_00rot'
    
    filename = macc_str + mdon_str + a_str + e_str + vfr_string + v_rot_str
    
    # Prepare dataframe to store data
    data = {'theta i [rad]': np.zeros(n),    ###
            'r orb [AU]': np.zeros(n),
            'v orb [km s-1]': np.zeros(n),
            'ang orb [rad]': np.zeros(n),
            'r L1 [AU]': np.zeros(n),        ###
            'v i [km s-1]': np.zeros(n),
            'ang i [rad]': np.zeros(n),
            'a p [AU]' : np.zeros(n),
            'e p' : np.zeros(n),
            'theta p i [rad]' : np.zeros(n),
            'flag impact': np.zeros(n),      ###
            'v imp [km s-1]': np.zeros(n),
            'ang imp [rad]': np.zeros(n)}
    df = pd.DataFrame(data = data)
    
    T = 2 * np.pi * np.sqrt((a**3) / (constants.G * (macc + mdon)))
    
    # Get orbit data
    #orbit_df = orbit_data(a, e, macc + mdon, 0, n)    # dataframe with theta, r, v and angle
    orbit_df = orbit_data(a, e, macc + mdon, T, 0, n)    # dataframe with theta, r, v and angle
    
    #print(orbit_df['v [km s-1]'].iloc[0])
    
    for i in orbit_df.index.to_list():
        #print('AAA\n', orbit_df.iloc[i])
        r_L1, v_i, ang_v, a_p, e_p, theta_p_i, flag, v_impact, ang_impact = get_particle_data(macc, mdon, racc,
                                                                                  orbit_df.iloc[i]['r [AU]'] | units.au,
                                                                                  orbit_df.iloc[i]['v [km s-1]'] | units.km * units.s**(-1),
                                                                                  orbit_df.iloc[i]['angle [rad]'],
                                                                                  orbit_df.iloc[i]['theta [rad]'],
                                                                                  v_rot, v_exp, n)
    
        df.iloc[i] = [orbit_df.iloc[i]['theta [rad]'], orbit_df.iloc[i]['r [AU]'], orbit_df.iloc[i]['v [km s-1]'], orbit_df.iloc[i]['angle [rad]'],
                      r_L1.value_in(units.au), v_i.value_in(units.km * units.s**(-1)), ang_v, a_p.value_in(units.au), e_p, theta_p_i,
                      flag, v_impact.value_in(units.km * units.s**(-1)), ang_impact]
    
    df.to_csv(dirname+filename+'.csv')
    
    return filename

def vel_limit(macc, mdon, racc, r, sma):
    v_orbit = np.sqrt(constants.G * (macc + mdon) * ((2/r) - (1/sma)))
    d_L1 = distance_to_L1(macc, mdon, sma, v_orbit)
    v = np.sqrt((2 * constants.G * macc)/((d_L1**2 / racc) + d_L1))
    return v, v_orbit

# From Eggleton 1983
def roche_radius(mdon, macc):
    q = mdon / macc
    q_23 = q**(2/3)
    q_13 = q**(1/3)
    r_L = (0.49 * q_23) / (0.6 * q_23 + np.log(1 + q_13))
    return r_L

def separation_limit(mdon, macc, rdon_max):
    r_l = roche_radius(mdon, macc)
    r_max = rdon_max / r_l
    return r_max

def eq_a_max(acc_mass, don_mass, rdon_max, v, sma):
    eq = ( constants.G * ( don_mass / ((rdon_max)**2 ))
           - constants.G * ( acc_mass / (sma - rdon_max)**2 )
           + (v**2/(sma - rdon_max)) )
    
    return eq

def a_max_guess(macc, mdon, vfr, rdon_max):
    
    guess1 = np.linspace(1.01 * rdon_max.value_in(units.m), 10 * rdon_max.value_in(units.m), 1000) | units.m
    values1 = []
    for i in guess1:
        #print(i, rdon_max)
        v = (1 - vfr) * np.sqrt(constants.G * (macc + mdon) * ((2/(i - rdon_max)) - (1/i)))
        #print(v)
        eq = eq_a_max(macc, mdon, rdon_max, v, i)
        #print(eq)
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

def many_systems_a(macc, mdon, racc, a_min, a_max, e, v_fr, v_exp, n_sys, n_dat):
    e_str = '_{:=07.4f}e'.format(e).replace('.', '_')
    vfr_str = '_{:=04.2f}vfr'.format(v_fr).replace('.', '_')
    vexp_str = '_{:=06.2f}vexp'.format(v_exp.value_in(units.km*units.s**-1)).replace('.', '_')
    dirname = './data/a'+e_str+vfr_str+vexp_str+'/'
    if not os.path.exists(dirname): 
        os.makedirs(dirname)
    
    #print(type(a_max))
    
    if a_max.value_in(units.au) == 0:
        '''
        If there is no stated value for max a, compute the maximum separation possible for the largest possible radius of a donor
        in its AGB stage. Data provided by Alonso.
        '''
        data = pd.read_table('1_SeBa_radius_short.data', sep="\t", skiprows=1, header=None, index_col=False,
                         names=['M [MSun]', 'M wd [MSun]', 'R ms [RSun]', 'R hg [RSun]', 'R rgb [RSun]', 'R hb [RSun]', 'R agb [RSun]'])
        r_agb = data.iloc[(data['M [MSun]'] - mdon.value_in(units.MSun)).abs().argsort()[:1]].iloc[0]['R agb [RSun]'] | units.RSun
        amax = a_max_guess(macc, mdon, v_fr, r_agb)
        if a_min.value_in(units.au) == 0:
            amin = 1.01 * (r_agb + racc)
            a_array = np.linspace(amin.value_in(units.au), amax.value_in(units.au), n_sys, endpoint=False)
        else:
            a_array = np.linspace(a_min.value_in(units.au), amax.value_in(units.au), n_sys, endpoint=False)
    else:
        a_array = np.linspace(a_min.value_in(units.au), a_max.value_in(units.au), n_sys, endpoint=False)
    
    i = 1
    f = open('a'+e_str+vfr_str+vexp_str+'.dat', 'x')
    for a in a_array:
        peri = (a | units.au) * (1 - e)
        v = -1 * vel_limit(macc, mdon, racc, peri, a | units.au)[1] * v_fr    #Donor rotation velocity as -90% v_orb at periastron
        print('Working on system number {} ...'.format(i))
        filename = get_table_for_system(macc, mdon, racc, a | units.au, e, v_fr, v, v_exp, n_dat, dirname)
        f.write(filename+'\n')
        i += 1
    f.close()

def many_systems_e(macc, mdon, racc, a, e_min, e_max, v_fr, v_exp, n_sys, n_dat):
    a_str = '_{:=07.3f}a'.format(a.value_in(units.au)).replace('.', '_')
    vfr_str = '_{:=04.2f}vfr'.format(v_fr).replace('.', '_')
    vexp_str = '_{:=06.2f}vexp'.format(v_exp.value_in(units.km*units.s**-1)).replace('.', '_')
    dirname = './data/e'+a_str+vfr_str+vexp_str+'/'
    if not os.path.exists(dirname): 
        os.makedirs(dirname)
    
    e_array = np.linspace(e_min, e_max, n_sys, endpoint=False)
    
    i = 1
    f = open('e'+a_str+vfr_str+vexp_str+'.dat', 'x')
    for e in e_array:
        print('Working on system number {} ...'.format(i))
        peri = a * (1 - e)
        v = -1 * vel_limit(macc, mdon, racc, peri, a)[1] * v_fr
        filename = get_table_for_system(macc, mdon, racc, a, e, v_fr, v, v_exp, n_dat, dirname)
        f.write(filename+'\n')
        i += 1
    f.close()

def many_systems_v(macc, mdon, racc, a, e, vfr_min, vfr_max, v_exp, n_sys, n_dat):
    a_str = '_{:=07.3f}a'.format(a.value_in(units.au)).replace('.', '_')
    e_str = '_{:=07.4f}e'.format(e).replace('.', '_')
    vexp_str = '_{:=06.2f}vexp'.format(v_exp.value_in(units.km*units.s**-1)).replace('.', '_')
    dirname = './data/vfr'+a_str+e_str+vexp_str+'/'
    if not os.path.exists(dirname): 
        os.makedirs(dirname)
    
    vfr_array = np.linspace(vfr_min, vfr_max, n_sys, endpoint=True)
    
    i = 1
    f = open('vfr'+a_str+e_str+vexp_str+'.dat', 'x')
    for v_fr in vfr_array:
        peri = a * (1 - e)
        v = -1 * vel_limit(macc, mdon, racc, peri, a)[1] * v_fr
        print('Working on system number {} ...'.format(i))
        filename = get_table_for_system(macc, mdon, racc, a, e, v_fr, v, v_exp, n_dat, dirname)
        f.write(filename+'\n')
        i += 1
    f.close()

def system_evol(macc_i, mdon_i, racc_i, a_i, e_i, vfr_i, vexp_i, n_dat, evol_a, evol_r):
    '''
    Evolves a system over time until at least one of three conditions is met:
        1. Donor star looses its envelope
        2. Accretor reaches critical rotation speed
        3. 
    This function takes the following input
    :macc_i:    Initial mass of the accretor
    :mdon_i:    Initial mass of the donor
    :racc_i:    Initial radius of the accretor
    :a_i:       Initial semi-major axis of the system
    :e_i:       Initial eccentricity of the system
    :vfr_i:     Initial donor rotation to periastron velocity ratio
    :vexp_i:    Initial donor's expansion velocity
    :n_dat:     Number of datapoints on an orbit
    :evol_a:    Semi-major axis evolution flag. If True, a is updated with each mass gain
    :evol_r:    Accretor's radius evolution flag. If True, racc is updated with each mass gain
    '''

    filename = get_table_for_system(macc_i, mdon_i, racc_i, a_i, e_i, vfr_i, vexp_i, n_dat, dirname)
    
    macc = macc_i
    mdon = mdon_i
    a = a_i
    e = e_i
    vfr = vfr_i
    vexp = vexp_i
    T = 2 * np.pi * np.sqrt((a**3) / (constants.G * (macc + mdon)))

    if evol_a == False:
        while:
            
            macc +=
            mdon +=
            if evol_r == False:
                racc = racc_i
            elif evol_r == True:
                racc = macc**0.8

            get_table_for_system(macc, mdon, racc, a, e)

    elif evol_a == True:
        while:
            T = 2 * np.pi * np.sqrt((a**3) / (constants.G * (macc + mdon)))
            mu = (macc * mdon) / (macc + mdon)
            L_orb = (mu * 2 * np.pi * a**2) / T
            dm_don =
            dm_acc =
            muf = ((macc+dm_g) * (mdon-dm_l)) / (macc + mdon + dm_g - dm_l)
            af = (L_orb**2 / (constants.G * (macc + mdon))) * muf**-2


def new_option_parser():
    result = OptionParser()
    result.add_option("--mode",
                      dest="mode", type="str",
                      default = 's',
                      help="run a single system (s), vary a (a), e (e) or v_fr (v)")
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
    result.add_option("-a", unit=units.au,
                      dest="a", type="float", 
                      default = 1.,
                      help="semi-major axis")
    result.add_option("--amin", unit=units.au,
                      dest="amin", type="float", 
                      default = 0.,
                      help="semi-major axis")
    result.add_option("--amax", unit=units.au,
                      dest="amax", type="float", 
                      default = 0.,
                      help="semi-major axis")
    result.add_option("-e", unit=None,
                      dest="e", type="float", 
                      default = 0.,
                      help="eccentricity")
    result.add_option("--emin", unit=None,
                      dest="emin", type="float", 
                      default = 0.,
                      help="eccentricity")
    result.add_option("--emax", unit=None,
                      dest="emax", type="float", 
                      default = 1.,
                      help="eccentricity")
    result.add_option("--vfr",
                      dest="v_fr", type="float", 
                      default = 0.9,
                      help="fraction of orb velocity as additional tangential velocity")
    result.add_option("--vfrmin",
                      dest="v_fr_min", type="float", 
                      default = 0.,
                      help="minimum value of v_fr")
    result.add_option("--vfrmax",
                      dest="v_fr_max", type="float", 
                      default = 1.,
                      help="maximum value of v_fr")
    result.add_option("--vtan", unit=units.km * units.s**-1,    #Not being used currently
                      dest="v_tan", type="float", 
                      default = 0.0,
                      help="additional tangential velocity")
    result.add_option("--vrad", unit=units.km * units.s**-1,
                      dest="v_rad", type="float", 
                      default = 0.0,
                      help="additional radial velocity (expansion of donor is positive)")
    result.add_option("--ndat",
                      dest="n_dat", type="int", 
                      default = 400,
                      help="number of datapoints in single orbit")
    result.add_option("--nsys",
                      dest="n_sys", type="int", 
                      default = 100,
                      help="number of systems to be generated (modes a and e)")
    return result

o, arguments  = new_option_parser().parse_args()

#print(np.linspace(o.emin, o.emax, o.n_sys, endpoint=False)[61])


if not os.path.exists('./data/'):
    os.makedirs('./data/')

if o.mode == 's':
    peri = o.a * (1 - o.e)
    v = -1 * vel_limit(o.macc, o.mdon, o.racc, peri, o.a)[1] * o.v_fr
    filename = get_table_for_system(o.macc, o.mdon, o.racc, o.a, o.e, o.v_fr, v, o.v_rad, o.n_dat, './data/')
elif o.mode == 'a':
    many_systems_a(o.macc, o.mdon, o.racc, o.amin, o.amax, o.e, o.v_fr, o.v_rad, o.n_sys, o.n_dat)
elif o.mode == 'e':
    many_systems_e(o.macc, o.mdon, o.racc, o.a, o.emin, o.emax, o.v_fr, o.v_rad, o.n_sys, o.n_dat)
elif o.mode == 'v':
    many_systems_v(o.macc, o.mdon, o.racc, o.a, o.e, o.v_fr_min, o.v_fr_max, o.v_rad, o.n_sys, o.n_dat)
else:
    print('Invalid mode. Please try again :/')
