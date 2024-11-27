import os
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

from amuse.community.mesa_r2208.interface import MESA

# Units
radius_unit = units.RSun
length_unit = units.au
time_unit = units.day
mass_unit = units.MSun

def mesa_data(M, R):
    stellar = MESA(version='2208')
    star = stellar.particles.add_particle(Particle(mass=M))
    print('Evolving star...')
    while (star.radius <= R):
        stellar.evolve_model(keep_synchronous=False)
    
    r = stellar.particles[0].get_radius_profile()
    rhos = stellar.particles[0].get_density_profile()
    P = stellar.particles[0].get_pressure_profile()
    stellar.stop()
    
    data = {'r': r.value_in(radius_unit), 'rho': rhos.value_in(units.g * units.cm**(-3)), 'P' : P.value_in(units.barye)}
    df = pd.DataFrame(data=data)
        
    return df

def pressure_intersect(df, rho, v):
    r = df['r']
    P = df['P']
    
    plt.plot(r, np.log10(P))
    plt.hlines(np.log10((rho * v**2).value_in(units.barye)), min(r), max(r))
    
    plt.savefig('./plots/pressure_gradient_intersect.png')

def density_profile(m, r):
    df = mesa_data(m, r)
    
    plt.plot(df['r'], np.log10(df['rho']))
    plt.ylabel(r'$\rho$ [g cm-3]')
    plt.xlabel(r'r [$R_{\odot}$]')
    
    plt.savefig('./plots/density_gradient.png')

def from_pars(parname, con1, con2, frac_i, T, plotname):
    frac_str = '_{:=04.2f}OF'.format(frac_i).replace('.', '_')
    if parname == 'a':
        print('Plotting '+plotname+' vs a')
        axname = r'$a$ [AU]'
        con1_str = '_{:=07.4f}e'.format(con1).replace('.', '_')
        con2_str = '_{:=05.2f}vfr'.format(con2).replace('.', '_')
        title = (r'$T$ = '+'{:=05.1f} d'.format(T.value_in(time_unit))
                 +r', $e =$'+'{:=.2f}'.format(con1)
                 +r', $v_{extra}/v_{per} =$'+'{:=.2f}'.format(con2)
                 +r', $f =$'+'{:=04.2f}'.format(frac_i))
    elif parname == 'e':
        print('Plotting '+plotname+' vs e')
        axname = r'$e$'
        con1_str = '_{:=07.2f}a'.format(con1.value_in(length_unit)).replace('.', '_')
        con2_str = '_{:=05.2f}vfr'.format(con2).replace('.', '_')
        title = (r'$T$ = '+'{:=05.1f} d'.format(T.value_in(time_unit))
                 +r', $a =$'+'{:=.2f} AU'.format(con1.value_in(length_unit))
                 +r', $v_{extra}/v_{per} =$'+'{:=.2f}'.format(con2)
                 +r', $f =$'+'{:=04.2f}'.format(frac_i))
    elif parname == 'vfr':
        print('Plotting '+plotname+' vs v')
        axname = r'$v_{extra}/v_{per}$'
        con1_str = '_{:=07.2f}a'.format(con1.value_in(length_unit)).replace('.', '_')
        con2_str = '_{:=07.4f}e'.format(con2).replace('.', '_')
        title = (r'$T$ = '+'{:=05.1f} d'.format(T.value_in(time_unit))
                 +r', $a =$'+'{:=.2f} AU'.format(con1.value_in(length_unit))
                 +r', $e =$'+'{:=.2f}'.format(con2)
                 +r', $f =$'+'{:=04.2f}'.format(frac_i))
    
    filename = parname+con1_str+con2_str+frac_str
    return axname, title, filename


def orbit_data(a, e, mtot, T, tau, n):
    '''
    Returns trajectory, velocity and it's angle as tangent to the orbit for a set of initial orbital parameters
    '''
    times = np.linspace(0.0, T.value_in(units.day), n)
    
    true_an = []
    for t in times:
        dif = 1.0
        M = 2*np.pi*(t - tau.value_in(units.day))/(T.value_in(units.day))
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
    r = (a * (1 - e**2)) / (1 + e*np.cos(true_an))
    v_2 = constants.G * mtot * ((2 / r) - (1 / a))
    
    data = {'theta [rad]': true_an,
            'r [AU]': r.value_in(units.au),
            'v [km s-1]': np.sqrt(v_2.value_in(units.km**2 * units.s**(-2))),
            'angle [rad]': np.ones(n)}
    df = pd.DataFrame(data=data)

    # Angle of velocity (tangent to the orbit)
    for i in df.index.tolist():
        aux_ang = np.arcsin(np.sin(df.iloc[i]['theta [rad]']) * 2 * (a.value_in(units.au) * e) / (2 * a.value_in(units.au) - df.iloc[i]['r [AU]']))
        df['angle [rad]'][i] = (aux_ang + np.pi) / 2
    
    return df

def orbit_lin(a, e, mtot, T, tau, n):
    '''
    Returns trajectory, velocity and it's angle as tangent to the orbit for a set of initial orbital parameters
    '''
    true_an = np.linspace(0, np.pi * 2, n)
    r = (a * (1 - e**2)) / (1 + e*np.cos(true_an))
    v_2 = constants.G * mtot * ((2 / r) - (1 / a))

    data = {'theta [rad]': true_an,
            'r [AU]': r.value_in(units.au),
            'v [km s-1]': np.sqrt(v_2.value_in(units.km**2 * units.s**(-2))),
            'angle [rad]': np.ones(n)}
    df = pd.DataFrame(data=data)

    # Angle of velocity (tangent to the orbit)
    for i in df.index.tolist():
        aux_ang = np.arcsin(np.sin(df.iloc[i]['theta [rad]']) * 2 * (a.value_in(units.au) * e) / (2 * a.value_in(units.au) - df.iloc[i]['r [AU]']))
        df.loc[i, 'angle [rad]'] = (aux_ang + np.pi) / 2
    return df


def orbit_example(filename, racc, ta, n, su):

    macc = float(((filename.split('/')[-1]).split('macc')[0]).replace('_', '.')) | mass_unit
    a = float((((filename.split('/')[-1]).split('mdon_')[-1]).split('a')[0]).replace('_', '.')) | length_unit
    e = float((((filename.split('/')[-1]).split('a_')[-1]).split('e')[0]).replace('_', '.'))
    vfr = float((((filename.split('/')[-1]).split('e_')[-1]).split('vfr')[0]).replace('_', '.'))
    df = pd.read_csv(filename+'.csv')
    
    # Get L1 orbit
    x_L1 = df['r L1 [AU]'] * np.cos(df['theta i [rad]'])
    y_L1 = df['r L1 [AU]'] * np.sin(df['theta i [rad]'])
    
    # Get orbit for particle at periapsis
    a_p = df.iloc[ta]['a p [AU]'] | length_unit
    e_p = df.iloc[ta]['e p']
    theta_i = df.iloc[ta]['theta p i [rad]']
    T_p = 2 * np.pi * np.sqrt((a_p**3) / (constants.G * macc))
    E = 2 * np.arctan(np.sqrt((1 - e_p)/(1 + e_p)) * np.tan(theta_i / 2))
    tau = - (T_p/(2*np.pi)) * (E - e_p * np.sin(E))
    particle = orbit_data(a_p, e_p, macc, T_p, tau, n)
    x_p = particle['r [AU]'] * np.cos(particle['theta [rad]'] - particle['theta [rad]'].iloc[0] + df.iloc[ta]['theta i [rad]'])
    y_p = particle['r [AU]'] * np.sin(particle['theta [rad]'] - particle['theta [rad]'].iloc[0] + df.iloc[ta]['theta i [rad]'])
    
    
    ################
    ### Plotting ###
    ################

    fig, ax = plt.subplots(figsize = (6,4), dpi=350)
    
    text = r'$a$ = ' + f'{a.value_in(length_unit)} AU, ' + r'$e$ = ' + f'{e}, ' + r'$v_{extra}/v_{per}$ = '+f'{vfr}'
    ax.set_title(text)

    # Plot L1 orbit
    ax.plot(x_L1, y_L1, linestyle='dashed', linewidth=1, color='k', label='L1 orbit')
    
    # Plot trajecroty
    if df.iloc[ta]['flag impact'] == 1:
        # Get index of impact
        idx_impact = particle.loc[(x_p**2 + y_p**2) <= (racc.value_in(length_unit)**2)].index[0]

        ax.plot(x_p[:idx_impact+1], y_p[:idx_impact+1], linestyle='solid', linewidth=1.5, color='deepskyblue', label='parcel trajectory')
        ax.plot(x_p[idx_impact:], y_p[idx_impact:], linestyle='dashed', linewidth=1.5, color='deepskyblue', alpha=0.5)
    else:
        ax.plot(x_p, y_p, linestyle='dashed', linewidth=1.5, color='crimson')
    
    # Accretor star
    accretor = plt.Circle((0,0), racc.value_in(length_unit), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
    ax.add_patch(accretor)
    
    # Normal view
    ax.set_xlim(-5 * racc.value_in(length_unit), 1.1 * max(x_p))
    ax.set_ylim(1.5 * min(y_p), 6.5 * max(y_p))
    
    ax.set_xlabel(r'$X$ [AU]')
    ax.set_ylabel(r'$Y$ [AU]')

    # Zoom-in plot
    zoom = max(x_p) / (3 * 2 * 1.75 * racc.value_in(length_unit))
    print(zoom)

    axzoom = zoomed_inset_axes(ax, zoom, loc=1)

    if df.iloc[ta]['flag impact'] == 1:
        axzoom.plot(x_p[:idx_impact+1], y_p[:idx_impact+1], linestyle='solid', linewidth=1.5, color='deepskyblue', label='parcel trajectory')
    else:
        axzoom.plot(x_p, y_p, linestyle='dashed', linewidth=1.5, color='crimson')
    
    axzoom.set_xlim(-1.75 * racc.value_in(length_unit), 1.75 * racc.value_in(length_unit))
    axzoom.set_ylim(-1.75 * racc.value_in(length_unit), 1.75 * racc.value_in(length_unit))
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    accretor = plt.Circle((0,0), racc.value_in(length_unit), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
    axzoom.add_patch(accretor)
    
    mark_inset(ax, axzoom, loc1=2, loc2=4)
    plt.draw()

    ax.set_aspect('equal')
    
    plt.savefig('./plots/'+filename.split('/')[-1]+'_orbit_example'+su+'.png')
    #plt.savefig('./plots/'+filename.split('/')[-1]+'_orbit_example_zoom.png')

    plt.close()

def orbits_example(filename, racc, n, div, su):

    df = pd.read_csv(filename+'.csv')
    
    macc = float(((filename.split('/')[-1]).split('macc')[0]).replace('_', '.')) | mass_unit
    df = pd.read_csv(filename+'.csv')
    
    # Get L1 orbit
    x_L1 = df['r L1 [AU]'] * np.cos(df['theta i [rad]'])
    y_L1 = df['r L1 [AU]'] * np.sin(df['theta i [rad]'])
    
    # Get indexes of orbits to plot
    idx_list = df.index.to_list()[::int(np.floor(len(df.index)/div))]
    
    ################
    ### Plotting ###
    ################

    fig, ax = plt.subplots(dpi=300, layout='constrained') #(figsize = (5.4,3.5), dpi=300, layout='constrained')
    
    # Plot L1 orbit
    ax.plot(x_L1, y_L1, linestyle='dashed', linewidth=1, color='k', label='L1 orbit')
    
    if 'new flag' in df.columns:
        flag_impact = df['flag impact'] + df['new flag']
        cut = 2
        ap = '_n'
    else:
        flag_impact = df['flag impact']
        cut = 1
        ap = '_y'
    
    for i in idx_list:
        
        if df.iloc[i]['fraction'] <= 0:
            ax.scatter(x_L1[i], y_L1[i], marker='x', c='crimson', s=50)
        else:
            # Get orbit of a parcel
            a_p = df.iloc[i]['a p [AU]'] | length_unit
            e_p = df.iloc[i]['e p']
            theta_i = df.iloc[i]['theta p i [rad]']
            T_p = 2 * np.pi * np.sqrt((a_p**3) / (constants.G * macc))
            E = 2 * np.arctan(np.sqrt((1 - e_p)/(1 + e_p)) * np.tan(theta_i / 2))
            tau = - (T_p/(2*np.pi)) * (E - e_p * np.sin(E))
            #particle = orbit_data(a_p, e_p, macc, T_p, tau, n)
            particle = orbit_lin(a_p, e_p, macc, T_p, tau, n)

            ta = df['theta i [rad]'].iloc[i]
            ang_t = df['ang imp [rad]'].iloc[i]
            # # This is ok!!!
            # theta_imp = 2*np.pi - np.arccos(((a_p * (1 - e_p**2) / racc) - 1) / e_p)
            # # This is wrong !!!!!!!!
            # #aux_ang = np.arcsin(np.sin(theta_imp - np.pi) * (2 * a_p * e_p) / (2 * a_p - racc))
            # aux_ang = np.arccos((4*a_p**2 - 4*a_p*racc + 2*racc**2 - 4*(a_p*e_p)**2)/(4*a_p*racc - 2*racc**2))
            # ang_imp = (np.pi - aux_ang) / 2

            # print(f'{i} a = {a_p}\n    e = {e_p}\n    2a-r = {(2 * a_p - racc).value_in(units.au)} au\n    2c = {(2 * a_p * e_p).value_in(units.au)}\n    true anomaly impact: {np.rad2deg(theta_imp)}\n    aux ang: {np.rad2deg(aux_ang)}\n    impact angle: {np.rad2deg(ang_imp)}')

            # theta_pi is inverted in my tables. This is a quick fix.
            theta_patch = theta_i + 2 * (-1 * theta_i + np.pi)

            idx_drop = particle.loc[(particle['theta [rad]'] - theta_patch).abs().argsort()[:1]].index[0]
            ang_corr = df.loc[i, 'theta i [rad]'] - theta_i
            
            x_p = particle['r [AU]'] * np.cos(particle['theta [rad]'] - particle['theta [rad]'].iloc[0] + ang_corr)
            y_p = particle['r [AU]'] * np.sin(particle['theta [rad]'] - particle['theta [rad]'].iloc[0] + ang_corr)

            if cut == 2:
                if df.iloc[i]['new flag'] == 0:
                    ax.scatter(x_L1[i], y_L1[i], marker='x', c='crimson', s=50)
                else:
                    if df.iloc[i]['flag impact'] == 0:
                        ax.plot(x_p, y_p, linewidth=1, alpha=0.5, color='crimson')
                    else:
                        # Get index of impact
                        idx_impact = particle.loc[(particle['r [AU]'] - racc.value_in(length_unit)).abs().argsort()[:2]].index.max() #[0]
                        
                        ax.plot(x_p[idx_drop:idx_impact], y_p[idx_drop:idx_impact], linestyle='solid', linewidth=1.5, color='deepskyblue')
                        ax.plot(x_p, y_p, linestyle='dashed', linewidth=1.5, color='deepskyblue', alpha=0.5)

            # if flag_impact[i] < cut:
            #     ax.plot(x_p, y_p, linewidth=1, alpha=0.5, color='crimson')
            # else:
            #     # Get index of impact
            #     idx_impact = particle.loc[particle['r [AU]'] <= racc.value_in(length_unit)].index[0]
            #     ax.plot(x_p[:idx_impact], y_p[:idx_impact], linestyle='solid', linewidth=1.5, color='deepskyblue')
            #     ax.plot(x_p[idx_impact:], y_p[idx_impact:], linestyle='dashed', linewidth=1.5, color='deepskyblue', alpha=0.5)

    
    # Accretor star
    accretor = plt.Circle((0,0), racc.value_in(length_unit), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
    ax.add_patch(accretor)
    
    # ax.set_xlim(1.2 * min(x_L1), 1.2 * max(x_L1))
    # ax.set_ylim(1.2 * min(y_L1), 1.2 * max(y_L1))
        
    ax.set_xlabel(r'$X$ [AU]')
    ax.set_ylabel(r'$Y$ [AU]')
    

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
    
    ax.set_aspect('equal')
    
    plt.savefig('./plots/'+filename.split('/')[-1]+'_orbits_example'+ap+su+'.png')

    ax.set_xlim(-1.2 * racc.value_in(length_unit), 1.2 * racc.value_in(length_unit))
    ax.set_ylim(-1.2 * racc.value_in(length_unit), 1.2 * racc.value_in(length_unit))

    ax.scatter([0], [0], c='black')

    plt.savefig('./plots/'+filename.split('/')[-1]+'_orbits_example'+ap+su+'_zoom.png')


def orbits_stream(filename, racc, T, tf, idx):

    df = pd.read_csv(filename+'.csv', index_col=0)
    
    macc = float(((filename.split('/')[-1]).split('macc')[0]).replace('_', '.')) | mass_unit
    mdon = float((((filename.split('/')[-1]).split('macc_')[-1]).split('mdon')[0]).replace('_', '.')) | mass_unit
    a = float((((filename.split('/')[-1]).split('mdon_')[-1]).split('a')[0]).replace('_', '.')) | length_unit
    e = float((((filename.split('/')[-1]).split('a_')[-1]).split('e')[0]).replace('_', '.'))
    v_rot = - float((((filename.split('/')[-1]).split('vfr_-')[-1]).split('rot')[0]).replace('_', '.'))
    
    print(T.value_in(time_unit))
    #n = np.floor(T.value_in(time_unit) / dt.value_in(time_unit)).astype('int')
    n = 150
    dt = T.value_in(units.s) / n
    idx_list = []
    
    n_final = int(np.ceil(tf.value_in(units.s) / dt))
    theta = 0
    for i in range(0, n_final+1):
        closest_idx = df.iloc[(df['theta i [rad]'] - theta).abs().argsort()[:1]].index[0]
        idx_list.append(closest_idx)
        
        v_i = df.iloc[closest_idx]['v i [km s-1]'] + (v_rot)
        r_i = df.iloc[closest_idx]['r orb [AU]']
        ang_i = df.iloc[closest_idx]['ang i [rad]']
        
        vang = np.sqrt(constants.G * (macc + mdon) * ((2 / (r_i | length_unit)) - (1 / a))) / (r_i | length_unit)
        theta += vang.value_in(units.s**(-1)) * dt
        
    idx_list = np.unique(np.array(idx_list))[:]
    #print(df.iloc[idx_list])
    
    #fig, axis = plt.subplots(figsize = (9,5), dpi=120, layout='constrained')
    fig, axis = plt.subplots(dpi=300, layout='constrained')
    
    # Get L1 orbit
    x_L1 = df['r L1 [AU]'] * np.cos(df['theta i [rad]'])
    y_L1 = df['r L1 [AU]'] * np.sin(df['theta i [rad]'])
    
    axis.plot(x_L1, y_L1, linestyle='dashed', linewidth=1, color='k', label='L1 orbit')
    
    hhh = 0
    count = 0
    for j in idx_list:
        p_orbit = orbit_data(df.iloc[j]['a p [AU]'] | length_unit, df.iloc[j]['e p'], macc, df.iloc[j]['theta p i [rad]'], 400)
        times = [0]
        t = 0
        
        for k in p_orbit.index.to_list():
            times.append(t)
            vang = np.sqrt(constants.G * (macc) * ((2 / (p_orbit.iloc[k]['r [AU]'] | length_unit)) - (1 / (df.iloc[j]['a p [AU]'] | length_unit)))) / (p_orbit.iloc[k]['r [AU]'] | length_unit)
            t += ((p_orbit.iloc[1]['theta [rad]'] - p_orbit.iloc[0]['theta [rad]']) / vang).value_in(time_unit)
            
        p_orbit['time [day]'] = times[1::] + (count * (dt | units.s).value_in(time_unit))
        count += 1
        #print(p_orbit['time [day]'])
        
        if df.iloc[j]['flag impact'] == 1:
            p_plot = p_orbit.loc[p_orbit['time [day]'] <= tf.value_in(time_unit)].iloc[:(p_orbit[p_orbit['r [AU]'] < racc.value_in(length_unit)].index[0]) + 1]
            
            x_t = p_plot['r [AU]'] * np.cos(p_plot['theta [rad]'] - p_orbit['theta [rad]'].iloc[0] + df.iloc[j]['theta i [rad]'])
            y_t = p_plot['r [AU]'] * np.sin(p_plot['theta [rad]'] - p_orbit['theta [rad]'].iloc[0] + df.iloc[j]['theta i [rad]'])
            
            if j == 0:
                axis.plot(x_t, y_t, linewidth = 0.5, alpha=0.1, color='mediumslateblue', label='particle trajectory')
            else:
                axis.plot(x_t, y_t, linewidth = 0.5, alpha=0.1, color='mediumslateblue')
        else:
            p_plot = p_orbit.loc[p_orbit['time [day]'] <= tf.value_in(time_unit)]
            
            x_t = p_plot['r [AU]'] * np.cos(p_plot['theta [rad]'] - p_orbit['theta [rad]'].iloc[0] + df.iloc[j]['theta i [rad]'])
            y_t = p_plot['r [AU]'] * np.sin(p_plot['theta [rad]'] - p_orbit['theta [rad]'].iloc[0] + df.iloc[j]['theta i [rad]'])
            
            if hhh == 0:
                #axis.plot(x_t, y_t, linewidth = 0.5, linestyle='dotted', alpha=0.5, color='crimson') #, label='particle trajectory (no impact)')
                hhh = 1
            #else:
                #axis.plot(x_t, y_t, linestyle='dotted', alpha=0.5, color='crimson')
        if len(x_t.index.to_list()) >= 1:
            if j == 0:
                axis.scatter(x_t.iloc[max(x_t.index.to_list())], y_t.iloc[max(x_t.index.to_list())], s=5, c='deepskyblue', label='parcel of mass')
            else:
                axis.scatter(x_t.iloc[max(x_t.index.to_list())], y_t.iloc[max(x_t.index.to_list())], s=5, c='deepskyblue')
        
    # Accretor star
    accretor = plt.Circle((0,0), racc.value_in(length_unit), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
    axis.add_patch(accretor)
    
    axis.set_xlim(1.2 * min(x_L1), 1.2 * max(x_L1))
    axis.set_ylim(1.2 * min(y_L1), 1.2 * max(y_L1))
    
    axis.set_xlabel(r'$r_x$ [AU]')
    axis.set_ylabel(r'$r_y$ [AU]')
    
    axis.set_title('t = {:=04} day'.format(tf.value_in(time_unit)))
    
    #axis.legend(loc='lower center', bbox_to_anchor=(0.5, -0.225), ncol=5)
    
    axis.set_aspect('equal')
    
    plt.savefig('./plots/stream/'+filename.split('/')[-1]+'_{:04}_stream.png'.format(idx))
    plt.close()
    
    return axis

def stream_animation(filename, racc, T, fps=15):

    # Read file and get parameters from its name
    df = pd.read_csv(filename+'.csv', index_col=0)
    macc = float(((filename.split('/')[-1]).split('macc')[0]).replace('_', '.')) | mass_unit
        
    print(T.value_in(time_unit))

    # Number of frames and timestamps
    df = df.iloc[::5]   # Only use some of the data points
    n_frames = len(df)
    timestamps = np.linspace(0, T.value_in(time_unit), n_frames)

    

    ###################
    ### First frame ###
    ###################
    fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
    
    # Get L1 orbit
    x_L1 = df['r L1 [AU]'] * np.cos(df['theta i [rad]'])
    y_L1 = df['r L1 [AU]'] * np.sin(df['theta i [rad]'])
    ax.plot(x_L1, y_L1, linestyle='dashed', linewidth=1, color='k', label='L1 orbit')

    def plot_trajs(i):
        print(f'Frame {i} of {n_frames}')
        dt = timestamps[1] | time_unit
        t = timestamps[i]
        print(f'    Plotting {len(df.iloc[:i+1])} trajectories')

        for j in range(len(df.iloc[:i+1])):
            if df.iloc[j]['fraction'] > 0.0:
                a_p = df.iloc[j]['a p [AU]'] | length_unit
                e_p = df.iloc[j]['e p']
                theta_p_i = df.iloc[j]['theta p i [rad]']
                
                T_p = 2 * np.pi * np.sqrt((a_p**3) / (constants.G * macc))
                E = 2 * np.arctan(np.sqrt((1 - e_p)/(1 + e_p)) * np.tan(theta_p_i / 2))
                tau = - (T_p/(2*np.pi)) * (E - e_p * np.sin(E))
                n_p = np.arange(0, T_p.value_in(time_unit), dt.value_in(time_unit))
                #p_orbit = orbit_data(a_p, e_p, macc, T_p, tau, len(n_p))
                
                if (df.iloc[j]['flag impact'] == 1) and (df.iloc[j]['new flag'] == 1):
                    p_orbit = orbit_data(a_p, e_p, macc, T_p, tau, len(n_p))
                    # Trajectories stop at accretor
                    x = (p_orbit['r [AU]'] * np.cos(p_orbit['theta [rad]'] - p_orbit.iloc[0]['theta [rad]'] + df.iloc[j]['theta i [rad]']))
                    y = (p_orbit['r [AU]'] * np.sin(p_orbit['theta [rad]'] - p_orbit.iloc[0]['theta [rad]'] + df.iloc[j]['theta i [rad]']))
                    
                    #print(p_orbit.loc[p_orbit['r [AU]'] <= racc.value_in(length_unit)])
                    #print(p_orbit['r [AU]'])
                    #print(n_p)
                    idx_impact = p_orbit.loc[p_orbit['r [AU]'] <= racc.value_in(length_unit)].index[0]
                    
                    # Update maximum reach of the trajectories
                    x = x[:idx_impact]
                    y = y[:idx_impact]

                    # Plot trajectories up to current time
                    x = x[:i-j+1]
                    y = y[:i-j+1]
                    ax.plot(x, y, linewidth=0.5, alpha=0.3, color='mediumslateblue')
                elif (df.iloc[j]['flag impact'] == 1) and (df.iloc[j]['new flag'] == 0):
                    ax.scatter(x_L1.iloc[i], y_L1.iloc[i], s=10, c='crimson', marker='x')
                else:
                    p_orbit = orbit_data(a_p, e_p, macc, T_p, tau, len(n_p))
                    # Make trajectory stop at current time
                    x = (p_orbit['r [AU]'] * np.cos(p_orbit['theta [rad]'] - p_orbit.iloc[0]['theta [rad]'] + df.iloc[j]['theta i [rad]']))[:i-j+1]
                    y = (p_orbit['r [AU]'] * np.sin(p_orbit['theta [rad]'] - p_orbit.iloc[0]['theta [rad]'] + df.iloc[j]['theta i [rad]']))[:i-j+1]

                    ax.plot(x, y, linewidth=0.5, alpha=0.3, color='crimson', linestyle='dashed')
                
                if len(x) > 1:
                    ax.scatter(x.iloc[-1], y.iloc[-1], s=5, c='deepskyblue')
                else:
                    ax.scatter(x, y, s=5, c='deepskyblue')
            else:
                ax.scatter(x_L1.iloc[i], y_L1.iloc[i], s=10, c='crimson', marker='x')


    # Starting parcel at periastron
    trajectories = plot_trajs(i=0)

    # Accretor star
    accretor = plt.Circle((0,0), racc.value_in(length_unit), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
    ax.add_patch(accretor)

    ax.set_xlim(1.2 * min(x_L1), 1.2 * max(x_L1))
    ax.set_ylim(1.2 * min(y_L1), 1.2 * max(y_L1))
    
    ax.set_xlabel(r'$r_x$ [AU]')
    ax.set_ylabel(r'$r_y$ [AU]')
    
    ####################
    ### Update frame ### 
    ####################

    def update(frame):

        #trajectories.set(i=frame)
        trajectories = plot_trajs(i=frame)

        ax.set_title('t = {:=04} day'.format(timestamps[1]*frame))
    
    ax.set_aspect('equal')
    
    #################
    ### Animation ###        
    #################
    
    # Create
    video = animation.FuncAnimation(fig=fig, func=update, frames=n_frames)

    # Save
    video.save('./plots/'+filename.split('/')[-1]+'_stream_animation.mp4', writer=animation.FFMpegWriter(fps=fps))

    plt.close()


def plot_larim(filename, racc, tf_list):
    
    fig, axis = plt.subplots(2, 2, figsize = (7,3), dpi=300, layout='constrained', sharex=True, sharey=True)
    
    text_list = ['a)', 'b)', 'c)', 'd)']
    
    df = pd.read_csv(filename+'.csv', index_col=0)
    
    macc = float(((filename.split('/')[-1]).split('macc')[0]).replace('_', '.')) | mass_unit
    mdon = float((((filename.split('/')[-1]).split('macc_')[-1]).split('mdon')[0]).replace('_', '.')) | mass_unit
    a = float((((filename.split('/')[-1]).split('mdon_')[-1]).split('a')[0]).replace('_', '.')) | length_unit
    e = float((((filename.split('/')[-1]).split('a_')[-1]).split('e')[0]).replace('_', '.'))
    v_rot = - float((((filename.split('/')[-1]).split('e_-')[-1]).split('rot')[0]).replace('_', '.'))
    
    period = 2 * np.pi * np.sqrt(a**3 / (constants.G * (macc + mdon)))
    print(period.value_in(time_unit))
    n = 100
    dt = period.value_in(units.s) / n
    
    for idx, ax in enumerate(fig.axes):
        idx_list = []
        n_final = int(np.ceil(tf_list[idx].value_in(units.s) / dt))
        theta = 0
        for i in range(0, n_final+1):
            closest_idx = df.iloc[(df['theta i [rad]'] - theta).abs().argsort()[:1]].index[0]
            idx_list.append(closest_idx)
            
            v_i = df.iloc[closest_idx]['v i [km s-1]'] + (v_rot)
            r_i = df.iloc[closest_idx]['r orb [AU]']
            ang_i = df.iloc[closest_idx]['ang i [rad]']
            
            vang = np.sqrt(constants.G * (macc + mdon) * ((2 / (r_i | length_unit)) - (1 / a))) / (r_i | length_unit)
            theta += vang.value_in(units.s**(-1)) * dt
            
        idx_list = np.unique(np.array(idx_list))[:]
        
        # Get L1 orbit
        x_L1 = df['r L1 [AU]'] * np.cos(df['theta i [rad]'])
        y_L1 = df['r L1 [AU]'] * np.sin(df['theta i [rad]'])
        
        ax.plot(x_L1, y_L1, linestyle='dashed', linewidth=1, color='k', label='L1 orbit')
    
        hhh = 0
        count = 0
        for j in idx_list:
            p_orbit = orbit_data(df.iloc[j]['a p [AU]'] | length_unit, df.iloc[j]['e p'], macc, df.iloc[j]['theta p i [rad]'], 400)
            times = [0]
            t = 0
            
            for k in p_orbit.index.to_list():
                times.append(t)
                vang = np.sqrt(constants.G * (macc) * ((2 / (p_orbit.iloc[k]['r [AU]'] | length_unit)) - (1 / (df.iloc[j]['a p [AU]'] | length_unit)))) / (p_orbit.iloc[k]['r [AU]'] | length_unit)
                t += ((p_orbit.iloc[1]['theta [rad]'] - p_orbit.iloc[0]['theta [rad]']) / vang).value_in(time_unit)
                
            p_orbit['time [day]'] = times[1::] + (count * (dt | units.s).value_in(time_unit))
            count += 1
            #print(p_orbit['time [day]'])
            
            if df.iloc[j]['flag impact'] == 1:
                p_plot = p_orbit.loc[p_orbit['time [day]'] <= tf_list[idx].value_in(time_unit)].iloc[:(p_orbit[p_orbit['r [AU]'] < racc.value_in(length_unit)].index[0]) + 1]
                
                x_t = p_plot['r [AU]'] * np.cos(p_plot['theta [rad]'] - p_orbit['theta [rad]'].iloc[0] + df.iloc[j]['theta i [rad]'])
                y_t = p_plot['r [AU]'] * np.sin(p_plot['theta [rad]'] - p_orbit['theta [rad]'].iloc[0] + df.iloc[j]['theta i [rad]'])
                
                if j == 0:
                    ax.plot(x_t, y_t, linewidth = 0.5, alpha=0.5, color='deepskyblue', label='particle trajectory')
                else:
                    ax.plot(x_t, y_t, linewidth = 0.5, alpha=0.5, color='deepskyblue')
            else:
                p_plot = p_orbit.loc[p_orbit['time [day]'] <= tf_list[idx].value_in(time_unit)]
                
                x_t = p_plot['r [AU]'] * np.cos(p_plot['theta [rad]'] - p_orbit['theta [rad]'].iloc[0] + df.iloc[j]['theta i [rad]'])
                y_t = p_plot['r [AU]'] * np.sin(p_plot['theta [rad]'] - p_orbit['theta [rad]'].iloc[0] + df.iloc[j]['theta i [rad]'])
                
                if hhh == 0:
                    ax.plot(x_t, y_t, linewidth = 0.5, linestyle='dotted', alpha=0.5, color='crimson') #, label='particle trajectory (no impact)')
                    hhh = 1
                else:
                    ax.plot(x_t, y_t, linestyle='dotted', alpha=0.5, color='crimson')
            if len(x_t.index.to_list()) >= 1:
                if j == 0:
                    ax.scatter(x_t.iloc[max(x_t.index.to_list())], y_t.iloc[max(x_t.index.to_list())], s=4, c='deepskyblue', label='parcel of mass')
                else:
                    ax.scatter(x_t.iloc[max(x_t.index.to_list())], y_t.iloc[max(x_t.index.to_list())], s=4, c='deepskyblue')
            
        # Accretor star
        accretor = plt.Circle((0,0), racc.value_in(length_unit), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
        ax.add_patch(accretor)
        
        ax.set_xlim(1.2 * min(x_L1), 1.2 * max(x_L1))
        ax.set_ylim(1.2 * min(y_L1), 1.2 * max(y_L1))
        
        ax.text(1.1 * min(x_L1), 0.8 * max(y_L1), text_list[idx])
    
    #fig.subplots_adjust(hspace=0)
    
    axis[1][0].set_xlabel(r'$r_x$ [AU]')
    axis[1][1].set_xlabel(r'$r_x$ [AU]')
    
    axis[0][0].set_ylabel(r'$r_y$ [AU]')
    axis[1][0].set_ylabel(r'$r_y$ [AU]')
    
    plt.savefig('./plots/equisde.pdf')


def spinup(par, dv, parname, con1, con2, frac_i, T, su):
    fig, axis = plt.subplots(figsize = (5,4), dpi=350, layout='constrained')
    
    axis.plot(par, dv, color='forestgreen')
    
    if parname in ['a', 'e', 'vfr']:
        axname, title, filename = from_pars(parname, con1, con2, frac_i, T, 'spin-up')
    else:
        print('No available plot :c')
    
    axis.set_title(title)
    axis.set_xlabel(axname)
    axis.set_ylabel(r'$\Delta v_{rot}$ [km s-1]')
    
    plt.savefig('./plots/'+filename+'_spinup'+su+'.png')
    
    return filename

def momentum_vs_par(par, p, p_t, p_r, parname, con1, con2, frac_i, T, su):
    fig, axis = plt.subplots(figsize = (5,4), dpi=350, layout='constrained')
    
    axis.plot(par, np.array(p)**2, color='deepskyblue', label='total')
    axis.plot(par, np.array(p_t)**2, color='yellowgreen', label='tangential')
    axis.plot(par, np.array(p_r)**2, color='tomato', label='radial')
    
    if parname in ['a', 'e', 'vfr']:
        axname, title, filename = from_pars(parname, con1, con2, frac_i, T, 'p')
    else:
        print('No available plot :c')
    
    axis.set_title(title)
    axis.set_xlabel(axname)
    axis.set_ylabel(r'$p^{2}$ [kg2 m2 s-2]')
    
    axis.legend(loc='best')
    
    plt.savefig('./plots/'+filename+'_momentum'+su+'.png')

def conservativeness_plot(par, mg, ml, parname, con1, con2, frac_i, T, su):
    fig, axis = plt.subplots(figsize = (5,4), dpi=350, layout='constrained')
    
    axis.plot(par, mg, color='limegreen', label='direct impact on accretor')
    axis.plot(par, ml, color='red', label='no impact on accretor')
    axis.plot(par, mg+ml, color='black', label='transfered by donor')
    
    if parname in ['a', 'e', 'vfr']:
        axname, title, filename = from_pars(parname, con1, con2, frac_i, T, 'mass conservativeness')
    else:
        print('No available plot :c')
    
    axis.set_title(title)
    axis.set_xlabel(axname)
    axis.set_ylabel(r'mass / transfered mass')
    
    axis.legend(loc='best')
    axis.set_ylim(bottom=-0.05)
    
    plt.savefig('./plots/'+filename+'_conservativeness'+su+'.png')

def spinup_comp(par, df, parname, varname, con1, con2, frac_i, T, su):
    fig, axis = plt.subplots(figsize = (4,3), dpi=600, layout='constrained')
    
    props = {'fontsize': 11}
    
    '''
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    '''
    colors = ['gold', 'limegreen']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 2)))
    
    for i in range(1, df.shape[1]):
        c = next(color)
        if varname == 'a':
            label = 'a = {:.2f} [AU]'.format(float(df.columns[i]))
            if parname == 'e':
                node_str = r'$v_{extra}/v_{per}$'+' = {:=04.2f}'.format(con2)
                axis.text(0.05, 2e-5, node_str, alpha=0.5, **props)
            elif parname == 'vfr':
                node_str = r'$e$'+' = {:=04.2f}'.format(con1)
                axis.text(0.8, 3e-5, node_str, alpha=0.5, **props)
        elif varname == 'e':
            label = 'e = {:.2f}'.format(float(df.columns[i]))
            if parname == 'a':
                node_str = r'$v_{extra}/v_{per}$'+' = {:=04.2f}'.format(con2)
                axis.text(1.225, 2.25e-5, node_str, alpha=0.5, **props)
            elif parname == 'vfr':
                node_str = r'$a$'+' = {:=04.2f} AU'.format(con1.value_in(length_unit))
                axis.text(0.8, 0.3e-5, node_str, alpha=0.5, **props)
        elif varname == 'vfr':
            label = '$v_{extra}/v_{per}$'+' = {:.2f}'.format(float(df.columns[i]))
            if parname == 'a':
                node_str = r'$e$'+' = {:=04.2f}'.format(con1)
                axis.text(2.1, 0.1e-5, node_str, alpha=0.5, **props)
            elif parname == 'e':
                node_str = r'$a$'+' = {:=04.2f} AU'.format(con1.value_in(length_unit))
                axis.text(0.01, 2e-5, node_str, alpha=0.5, **props)
        
        axis.plot(par, df.iloc[:, [i]], label=label, color=c)#, linestyle='dashed')
    
    if parname in ['a', 'e', 'vfr']:
        axname, title, filename = from_pars(parname, con1, con2, frac_i, T, 'spin-up comparison')
    else:
        print('No available plot :c')
    
    #axis.set_title(title)
    axis.set_xlabel(axname, **props)
    axis.set_ylabel(r'$\Delta v_{rot} / v_{crit}$', **props)
    
    #axis.hlines(1, min(par), max(par), linestyle='dashed', color='black')
    
    axis.legend(loc='best', **props)
    axis.set_ylim(bottom=-0.01e-5)
    
    plt.savefig('./plots/'+filename+'_spinup_comp'+su+'.png')

def split_table(df, param):
    '''
    Returns indexes of intervals where
    '''
    param_col = df[param]
    thing1 = param_col - np.roll(param_col, 1)
    if np.abs(np.abs(thing1.iloc[0]) - (2 * np.pi)) < 0.01:
        thing1.iloc[0] = 0.0
    idxs = df.loc[np.abs(thing1) > (1.1 * thing1.iloc[1])].index

    new_idxs = []
    if len(idxs) == 0:
        print('No split in domain')
        if (df.index.to_list()[0] == 0) & (df.index.to_list()[1] != 0):
            new_idxs.append(df.index.to_list()[1:]+[0])
        else:
            new_idxs.append(df.index.to_list())
    elif len(idxs) == 1:
        new_idxs.append(df.loc[df.index >= idxs.to_list()[0]].index.to_list() + df.loc[df.index < idxs.to_list()[0]].index.to_list())
    else:
        for i in range(1, len(idxs)+1):
            if i == len(idxs):
                new_idxs.append(df.loc[df.index >= idxs.to_list()[-1]].index.to_list() + df.loc[df.index < idxs.to_list()[0]].index.to_list())
            else:
                new_idxs.append(df.loc[(df.index >= idxs.to_list()[i-1]) & (df.index < idxs.to_list()[i])].index.to_list())

    return new_idxs

def plot_conditions(filename, racc, su):
    '''
    Plots the domain in the L1 orbit where each of the conditions is met.
    :filename: Name of file to be analized
    :racc: Radius of the accretor star
    :su: Optional sufix for the plot's name
    '''

    a = float((((filename.split('/')[-1]).split('mdon_')[-1]).split('a')[0]).replace('_', '.')) | length_unit
    e = float((((filename.split('/')[-1]).split('a_')[-1]).split('e')[0]).replace('_', '.'))
    vfr = float((((filename.split('/')[-1]).split('e_')[-1]).split('vfr')[0]).replace('_', '.'))
    
    df = pd.read_csv(filename+'.csv')

    # Get L1 orbit
    x_L1 = df['r L1 [AU]'] * np.cos(df['theta i [rad]'])
    y_L1 = df['r L1 [AU]'] * np.sin(df['theta i [rad]'])

    ################
    ### Plotting ###
    ################

    props = {'fontsize': 11}

    fig, ax = plt.subplots(dpi=300, figsize=(6, 4)) #(figsize = (5.4,3.5), dpi=300, layout='constrained')
    
    # Set axis limits
    ax.set_ylim(min(y_L1[::2])*1.3, max(y_L1[::2])*1.3)
    ax.set_xlim(min(x_L1[::2])-(max(x_L1[::2])/3), max(x_L1[::2])+(max(x_L1[::2])/3))
    ax.set_aspect('equal')

    # Axes names
    ax.set_xlabel(r'$X$ [AU]', **props)
    ax.set_ylabel(r'$Y$ [AU]', **props)

    # Plot accretor star
    accretor = plt.Circle((0,0), racc.value_in(length_unit), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
    ax.add_patch(accretor)

    # Plot L1 orbit
    ax.plot(x_L1, y_L1, linestyle='dashed', linewidth=1, color='k', label='L1 orbit')

    # Write parameters in title
    text = r'$a$ = ' + f'{a.value_in(length_unit)} AU, ' + r'$e$ = ' + f'{e}, ' + r'$v_{extra}/v_{per}$ = '+f'{vfr}'
    ax.set_title(text)

    # Plot 1st condition (overflow)
    cond_1 = df.loc[df['fraction']>0]
    idxs_1 = split_table(cond_1, 'theta i [rad]')

    for ipack in idxs_1:
        r_L1, theta_i = [[], []]
        for i in ipack:
            row = cond_1.loc[cond_1.index == i]
            r_L1.append(row['r L1 [AU]'])
            theta_i.append(row['theta i [rad]'])
        x = r_L1 * np.cos(theta_i)
        y = r_L1 * np.sin(theta_i)

        ax.plot(x, y, color='deepskyblue', alpha=0.33, linewidth=20, solid_capstyle='round')

    # Save plot with only 1st condition
    #plt.savefig('./plots/'+filename.split('/')[-1]+'_conditions_1'+su+'.png')

    # Plot 2nd condition (trajectory)
    cond_2 = df.loc[df['flag impact']==1]

    if len(cond_2) == 0:
        ax.plot([], [])
    else:
        idxs_2 = split_table(cond_2, 'theta i [rad]')

        for ipack in idxs_2:
            r_L1, theta_i = [[], []]
            for i in ipack:
                row = cond_2.loc[cond_2.index == i]
                r_L1.append(row['r L1 [AU]'])
                theta_i.append(row['theta i [rad]'])
            x = r_L1 * np.cos(theta_i)
            y = r_L1 * np.sin(theta_i)

            ax.plot(x, y, color='limegreen', alpha=0.33, linewidth=20, solid_capstyle='round')

    # Save plot with 1st and 2nd conditions
    #plt.savefig('./plots/'+filename.split('/')[-1]+'_conditions_12'+su+'.png')
    
    inter_top = df.iloc[:int(len(df)/2)].loc[(df['fraction']>0) & (df['flag impact']==1)]
    inter_bot = df.iloc[int(len(df)/2):].loc[(df['fraction']>0) & (df['flag impact']==1)]

    # Plot 3rd condition (angle) if applied
    if 'new flag' in df.columns:
        cond_3 = df.loc[df['new flag']==1]
        idxs_3 = split_table(cond_3, 'theta i [rad]')
        
        for ipack in idxs_3:
            r_L1, theta_i = [[], []]
            for i in ipack:
                row = cond_3.loc[cond_3.index == i]
                r_L1.append(row['r L1 [AU]'])
                theta_i.append(row['theta i [rad]'])
            x = r_L1 * np.cos(theta_i)
            y = r_L1 * np.sin(theta_i)

            ax.plot(x, y, color='darkviolet', alpha=0.33, linewidth=20, solid_capstyle='round')

        # Save plot with 1st, 2nd and 3rd conditions
        #plt.savefig('./plots/'+filename.split('/')[-1]+'_conditions_123'+su+'.png')

        inter_top = inter_top.loc[inter_top['new flag']==1]
        inter_bot = inter_bot.loc[inter_bot['new flag']==1]

    # Plot conditions intersection

    x_intertop = inter_top['r L1 [AU]'] * np.cos(inter_top['theta i [rad]'])
    y_intertop = inter_top['r L1 [AU]'] * np.sin(inter_top['theta i [rad]'])

    x_interbot = inter_bot['r L1 [AU]'] * np.cos(inter_bot['theta i [rad]'])
    y_interbot = inter_bot['r L1 [AU]'] * np.sin(inter_bot['theta i [rad]'])

    ax.plot(x_intertop, y_intertop, color='k', linewidth=17, solid_capstyle='round')
    ax.plot(x_interbot, y_interbot, color='k', linewidth=17, solid_capstyle='round')
    
    # Save plot with all three conditions and their intersection
    plt.savefig('./plots/'+filename.split('/')[-1]+'_conditions'+su+'.png')

    plt.close()

def conditions_svg(filename, racc, su):
    a = float((((filename.split('/')[-1]).split('mdon_')[-1]).split('a')[0]).replace('_', '.')) | length_unit
    e = float((((filename.split('/')[-1]).split('a_')[-1]).split('e')[0]).replace('_', '.'))
    vfr = float((((filename.split('/')[-1]).split('e_')[-1]).split('vfr')[0]).replace('_', '.'))
    
    df = pd.read_csv(filename+'.csv')

    # Get L1 orbit
    x_L1 = df['r L1 [AU]'] * np.cos(df['theta i [rad]'])
    y_L1 = df['r L1 [AU]'] * np.sin(df['theta i [rad]'])

    # Get donor orbit
    x_orb = df['r orb [AU]'] * np.cos(df['theta i [rad]'])
    y_orb = df['r orb [AU]'] * np.sin(df['theta i [rad]'])

    ################
    ### Plotting ###
    ################

    props = {'fontsize': 11}

    fig, ax = plt.subplots(dpi=300, figsize=(6, 4)) #(figsize = (5.4,3.5), dpi=300, layout='constrained')
    
    # Set axis limits
    ax.set_ylim(min(y_orb[::2])*1.3, max(y_orb[::2])*1.3)
    ax.set_xlim(min(x_orb[::2])-(max(x_orb[::2])/3), max(x_orb[::2])+(max(x_orb[::2])/3))
    ax.set_aspect('equal')

    # Axes names
    #ax.set_xlabel(r'$X$ [AU]', **props)
    #ax.set_ylabel(r'$Y$ [AU]', **props)

    # Plot accretor star
    accretor = plt.Circle((0,0), 10*racc.value_in(length_unit), color='hotpink', label='accretor star', zorder=3)
    ax.add_patch(accretor)

    # Plot L1 orbit
    ax.plot(x_orb, y_orb, linestyle='dotted', linewidth=1, color='k', label='L1 orbit')

    # Write parameters in title
    #text = r'$a$ = ' + f'{a.value_in(length_unit)} AU, ' + r'$e$ = ' + f'{e}, ' + r'$v_{extra}/v_{per}$ = '+f'{vfr}'
    #ax.set_title(text)

    # Plot 1st condition (overflow)
    cond_1 = df.loc[df['fraction']>0]
    idxs_1 = split_table(cond_1, 'theta i [rad]')

    for ipack in idxs_1:
        r_orb, theta_i = [[], []]
        for i in ipack:
            row = cond_1.loc[cond_1.index == i]
            r_orb.append(row['r orb [AU]'])
            theta_i.append(row['theta i [rad]'])
        x = r_orb * np.cos(theta_i)
        y = r_orb * np.sin(theta_i)

        ax.plot(x, y, color='deepskyblue', alpha=0.50, linewidth=30, solid_capstyle='round')

    # Save plot with only 1st condition
    #plt.savefig('./plots/'+filename.split('/')[-1]+'_conditions_1'+su+'.png')

    # Plot 2nd condition (trajectory)
    cond_2 = df.loc[df['flag impact']==1]

    if len(cond_2) == 0:
        ax.plot([], [])
    else:
        idxs_2 = split_table(cond_2, 'theta i [rad]')

        for ipack in idxs_2:
            r_orb, theta_i = [[], []]
            for i in ipack:
                row = cond_2.loc[cond_2.index == i]
                r_orb.append(row['r orb [AU]'])
                theta_i.append(row['theta i [rad]'])
            x = r_orb * np.cos(theta_i)
            y = r_orb * np.sin(theta_i)

            ax.plot(x, y, color='limegreen', alpha=0.50, linewidth=20, solid_capstyle='round')

    # Save plot with 1st and 2nd conditions
    #plt.savefig('./plots/'+filename.split('/')[-1]+'_conditions_12'+su+'.png')
    
    inter_top = df.iloc[:int(len(df)/2)].loc[(df['fraction']>0) & (df['flag impact']==1)]
    inter_bot = df.iloc[int(len(df)/2):].loc[(df['fraction']>0) & (df['flag impact']==1)]

    # Plot 3rd condition (angle) if applied
    if 'new flag' in df.columns:
        cond_3 = df.loc[df['new flag']==1]
        idxs_3 = split_table(cond_3, 'theta i [rad]')
        
        for ipack in idxs_3:
            r_orb, theta_i = [[], []]
            for i in ipack:
                row = cond_3.loc[cond_3.index == i]
                r_orb.append(row['r orb [AU]'])
                theta_i.append(row['theta i [rad]'])
            x = r_orb * np.cos(theta_i)
            y = r_orb * np.sin(theta_i)

            ax.plot(x, y, color='darkviolet', alpha=0.50, linewidth=10, solid_capstyle='round')

        inter_top = inter_top.loc[inter_top['new flag']==1]
        inter_bot = inter_bot.loc[inter_bot['new flag']==1]

    # Plot conditions intersection

    x_intertop = inter_top['r orb [AU]'] * np.cos(inter_top['theta i [rad]'])
    y_intertop = inter_top['r orb [AU]'] * np.sin(inter_top['theta i [rad]'])

    x_interbot = inter_bot['r orb [AU]'] * np.cos(inter_bot['theta i [rad]'])
    y_interbot = inter_bot['r orb [AU]'] * np.sin(inter_bot['theta i [rad]'])

    ax.plot(x_intertop, y_intertop, color='k', linewidth=5, solid_capstyle='round')
    ax.plot(x_interbot, y_interbot, color='k', linewidth=5, solid_capstyle='round')
    
    ax.axis('off')

    # Save plot with all three conditions and their intersection
    plt.savefig('./plots/'+filename.split('/')[-1]+'_conditions'+su+'.svg')

    plt.close()

def accretion_space(df, filename, racc):
    a = float((((filename.split('/')[-1]).split('mdon_')[-1]).split('a')[0]).replace('_', '.'))
    e = float((((filename.split('/')[-1]).split('a_')[-1]).split('e')[0]).replace('_', '.'))
    
    fig, axis = plt.subplots(figsize = (7,5), dpi=120, layout='constrained')
    
    # Get L1 orbit
    x_L1 = df['r L1 [AU]'] * np.cos(df['theta i [rad]'])
    y_L1 = df['r L1 [AU]'] * np.sin(df['theta i [rad]'])
    
    axis.plot(x_L1, y_L1, linestyle='dashed', linewidth=1, color='k', label='L1 orbit')
    
    # Accretor star
    accretor = plt.Circle((0,0), racc.value_in(length_unit), color='hotpink', alpha=0.75, label='accretor star', zorder=3)
    axis.add_patch(accretor)
    
    # RLOF
    x_RLOF = x_L1.loc[df['fraction'] > 0]
    y_RLOF = y_L1.loc[df['fraction'] > 0]
    
    cut_RLOF = df.index.to_numpy()[np.where((df.loc[df['fraction'] > 0].index.to_numpy() - np.roll(df.loc[df['fraction'] > 0].index.to_numpy(), 1)) > 1)]
    
    if len(cut_RLOF) > 0:
        axis.plot(x_RLOF.iloc[:cut_RLOF[0]], y_RLOF.iloc[:cut_RLOF[0]], linewidth=10, color='deepskyblue', alpha=0.5)
        axis.plot(x_RLOF.iloc[cut_RLOF[0]:], y_RLOF.iloc[cut_RLOF[0]:], linewidth=10, color='deepskyblue', alpha=0.5)
    else:
        axis.plot(x_RLOF, y_RLOF, linewidth=10, color='deepskyblue', alpha=0.5)
    
    # Velociy
    x_vel = x_L1.loc[df['flag impact'] == 1.0]
    y_vel = y_L1.loc[df['flag impact'] == 1.0]
    
    cut_vel = df.index.to_numpy()[np.where((df.loc[df['flag impact'] == 1.0].index.to_numpy() - np.roll(df.loc[df['flag impact'] == 1.0].index.to_numpy(), 1)) > 1)]
    
    if len(cut_vel) > 0:
        axis.plot(x_vel.iloc[:cut_vel[0]], y_vel.iloc[:cut_vel[0]], linewidth=10, color='forestgreen', alpha=0.4)
        axis.plot(x_vel.iloc[cut_vel[0]:], y_vel.iloc[cut_vel[0]:], linewidth=10, color='forestgreen', alpha=0.4)
    else:
        axis.plot(x_vel, y_vel, linewidth=10, color='forestgreen', alpha=0.4)
    
    axis.set_title(r'$e = $'+'{:=05.3f}'.format(e))
    axis.set_aspect('equal')
    
    axis.set_xlabel(r'$r_x$ [AU]')
    axis.set_ylabel(r'$r_y$ [AU]')
    
    axis.set_xlim(left=-1.1 * a, right=0.75 * a)
    axis.set_ylim(bottom=-0.75 * a, top=0.75 * a)
    
    plt.savefig('./plots/'+filename+'_accretion_space.png')

def spinup_vs_gm(par_list, L_list, dm_list, dml_list, parname, con1, con2, frac_i, period, macc, racc, mdon, su):
    table = pd.read_table('1_SeBa_radius_short.data', sep="\t", skiprows=1, header=None, index_col=False,
                         names=['M [MSun]', 'M wd [MSun]', 'R ms [RSun]', 'R hg [RSun]', 'R rgb [RSun]', 'R hb [RSun]', 'R agb [RSun]'])
    m_env = mdon - (table.iloc[(table['M [MSun]'] - mdon.value_in(mass_unit)).abs().argsort()[:1]].iloc[0]['M wd [MSun]'] | mass_unit)
    
    L_arr = np.array(L_list)
    dm_arr = np.array(dm_list)
    dml_arr = np.array(dml_list)
    '''
    L_arr = L_arr[np.where(L_arr != 0)]
    dm_arr = dm_arr[np.where(dm_arr != 0)]
    dml_arr = dml_arr[np.where(dml_arr != 0)]
    '''
    par_val = {'a': [1.3, 1.8, 2.2], 'e': [0.0, 0.05, 0.1, 0.3, 0.6], 'vfr':[0.80, 0.85, 0.90, 0.95]}
    
    fig, axis = plt.subplots(figsize = (4,4), dpi=600)
    
    props = {'fontsize': 11}
    
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    axis.set_xlabel(r'$dm / m_{acc, i}$', **props)
    axis.set_ylabel(r'$v_{rot}/v_{crit}$', **props)
    
    if parname in ['a', 'e', 'vfr']:
        axname, title, filename = from_pars(parname, con1, con2, frac_i, 0 | time_unit, '')
    else:
        print('No available plot :c')
    
    par_sr = pd.Series(par_list)
    aprox = [par_sr[(par_sr - v).abs().argsort()[:1]].iloc[0] for v in par_val[parname]]
    
    idx_list = []
    for ap in aprox:
        for idx, v in enumerate(par_list):
            if v == ap:
                idx_list.append(idx)
    
    dm_max = []
    for i in idx_list:
        L_evol, v_evol, T_list, v40_list, mg = [[], [], [], [], []]
        L = 0 | units.kg * units.m**2 * units.s**(-1)
        v = 0
        T = 0
        v40 = (40 | units.km * units.s**-1) / (constants.G * macc / racc)**0.5
        mass = macc
        ml = 0 | mass_unit
        while ml.value_in(mass_unit) < m_env.value_in(mass_unit):
            #print('i = {}, T = {}, v/v_crit = {}, mass left = {}'.format(i, T, v, (m_env - mass + macc).value_in(mass_unit)))
            
            L_evol.append(L.value_in(units.kg * units.m**2 * units.s**(-1)))
            v_evol.append(v)
            T_list.append(T)
            v40_list.append(v40)
            mg.append((mass - macc)/macc)
            
            L += L_arr[i] | units.kg * units.m**2 * units.s**(-1)
            mass += dm_arr[i] | units.kg
            ml += dml_arr[i] | units.kg
            v_crit = (constants.G * mass / racc)**0.5
            v = (L / (racc * mass)) / v_crit
            v40 = (40 | units.km * units.s**-1) / v_crit
            #print(v_crit, L)
            T += 1
            
        c = next(color)

        if parname == 'a':
            label = 'a = {:.2f} AU, t = {} periods'.format(float(par_list[i]), T)
            ncols = 1
            text = r'$e = 0.10$, $v_{extra}/v_{per} = 0.90$'
            fc = 0.34
            linestyle = 'solid'
        elif parname == 'e':
            label = 'e = {:.2f}'.format(float(par_list[i]))
            ncols = 2
            text = r'$a = 1.80$ AU, $v_{extra}/v_{per} = 0.90$'
            fc = 0.3
            linestyle = 'solid'
        elif parname == 'vfr':
            label = '$v_{extra}/v_{per}$'+' = {:.2f}'.format(float(par_list[i]))
            ncols = 1
            text = r'$a = 1.80$ AU, $e = 0.10$'
            fc = 0.45
            print(par_list[i])
            if np.abs(par_list[i]-0.95) < 0.01:
                linestyle = 'dotted'
            elif np.abs(par_list[i]-0.80) < 0.01:
                linestyle = 'solid'
                axis.scatter(mg, v_evol, color=c, zorder=3, s=1.5)
            else:
                linestyle = 'solid'

        axis.plot(mg, v_evol, color=c, label=label, linestyle=linestyle) #, alpha=0.3)
        axis.plot(mg, v40_list, color='k', linestyle='dashed')
        
        dm_max.append(max(mg))
        
    axis.plot([], [], color='k', linestyle='dashed', label='40 km s-1')
    axis.legend(loc='center left', ncol=ncols, **props)
    #print(mg, v_evol)
    axis.text(max(dm_max)*fc, 0.0, text, alpha=0.5, **props)
    
    plt.subplots_adjust(left=0.16, right=0.99, top=0.99, bottom=0.12)
    plt.savefig('./plots/'+filename+'_spin_mg'+su+'.png')
    
    print(T)
    
    plt.close()

def spinup_per_period(par_list, L_list, dm_list, dml_list, parname, con1, con2, frac_i, period, macc, racc, mdon, su):
    table = pd.read_table('1_SeBa_radius_short.data', sep="\t", skiprows=1, header=None, index_col=False,
                         names=['M [MSun]', 'M wd [MSun]', 'R ms [RSun]', 'R hg [RSun]', 'R rgb [RSun]', 'R hb [RSun]', 'R agb [RSun]'])
    m_env = mdon - (table.iloc[(table['M [MSun]'] - mdon.value_in(mass_unit)).abs().argsort()[:1]].iloc[0]['M wd [MSun]'] | mass_unit)
    
    L_arr = np.array(L_list)
    dm_arr = np.array(dm_list)
    dml_arr = np.array(dml_list)
    
    L_arr = L_arr[np.where(L_arr != 0)]
    dm_arr = dm_arr[np.where(dm_arr != 0)]
    dml_arr = dml_arr[np.where(dml_arr != 0)]
    
    par_val = {'a': [1.3, 1.8, 2.2], 'e': [0.0, 0.05, 0.1, 0.3, 0.6], 'vfr':[0.80, 0.85, 0.90, 0.95]}
    
    ### PERIOD VS SPINUP PLOT & GAINED MASS VS SPINUP ###
    fig, (axis, axis2) = plt.subplots(1, 2, figsize = (6,4), dpi=600, sharey=True)
    
    props = {'fontsize': 11}
    
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    max_T = []
    
    axis.set_xlabel('time [periods]', **props)
    axis.set_ylabel(r'$v_{rot}/v_{crit}$', **props)
    
    axis2.set_xlabel(r'$dm / m_{acc}$', **props)
    
    if parname in ['a', 'e', 'vfr']:
        axname, title, filename = from_pars(parname, con1, con2, frac_i, 0|units.day, '')
        if parname == 'a':
            node_str = r'$e$'+' = {:=04.2f}, '.format(con1)+r'$v_{extra}/v_{per}$'+' = {:=04.2f}'.format(con2)
            axis2.text(0.05, 0.055, node_str, alpha=0.5, **props)
        elif parname == 'e':
            node_str = r'$a$'+' AU = {:=04.2f}, '.format(con1)+r'$v_{extra}/v_{per}$'+' = {:=04.2f}'.format(con2)
            axis2.text(0.1, 0.05, node_str, alpha=0.5, **props)
        elif parname == 'vfr':
            node_str = r'$a$'+' AU = {:=04.2f}, '.format(con1)+r'$e$'+' = {:=04.2f}'.format(con2)
            axis2.text(0.1, 0.05, node_str, alpha=0.5, **props)
    else:
        print('No available plot :c')
    
    T_max = []
    par_sr = pd.Series(par_list)
    aprox = [par_sr[(par_sr - v).abs().argsort()[:1]].iloc[0] for v in par_val[parname]]
    
    idx_list = []
    for ap in aprox:
        for idx, v in enumerate(par_list):
            if v == ap:
                idx_list.append(idx)
    
    for i in idx_list:
        L_evol, v_evol, T_list, v40_list, mg = [[], [], [], [], []]
        L = 0 | units.kg * units.m**2 * units.s**(-1)
        v = 0
        T = 0
        v40 = (40 | units.km * units.s**-1) / (constants.G * macc / racc)**0.5
        mass = macc
        ml = 0 | mass_unit
        #while v <= 1:
        #while mass.value_in(mass_unit) < (macc + m_env).value_in(mass_unit) and v <= 1:
        while ml.value_in(mass_unit) < m_env.value_in(mass_unit):
            #print('i = {}, T = {}, v/v_crit = {}, mass left = {}'.format(i, T, v, (m_env - mass + macc).value_in(mass_unit)))
            L_evol.append(L.value_in(units.kg * units.m**2 * units.s**(-1)))
            v_evol.append(v)
            T_list.append(T)
            v40_list.append(v40)
            mg.append((mass - macc)/macc)
            
            try:
                L += L_arr[i] | units.kg * units.m**2 * units.s**(-1)
            except:
                break    
            mass += dm_arr[i] | units.kg
            ml += dml_arr[i] | units.kg
            v_crit = (constants.G * mass / racc)**0.5
            v = (L / (racc * mass)) / v_crit
            #new_racc = (mass.value_in(units.MSun))**0.8 | units.RSun
            #v_crit = (constants.G * mass / new_racc)**0.5
            #v = (L / (new_racc * mass)) / v_crit
            v40 = (40 | units.km * units.s**-1) / v_crit
            
            T += 1
            
        max_T.append(T)
        c = next(color)
        
        if parname == 'a':
            a = float(par_list[i]) | units.au
            P = 2 * np.pi * (a**3 / (constants.G * (macc + mdon)))**0.5
            label = 'a = {:.2f} AU, P = {:.2f} d'.format(a.value_in(units.au), P.value_in(units.day))
        elif parname == 'e':
            label = 'e = {:.2f}'.format(float(par_list[i]))
        elif parname == 'vfr':
            label = '$v_{extra}/v_{per}$'+' = {:.2f}'.format(float(par_list[i]))
        T_max.append(max(T_list))
        
        axis.plot(T_list, v_evol, color=c, label=label)
        axis.plot(T_list, v40_list, color=c, linestyle='dashed')
        
        if np.abs(float(par_list[i]) - 2.20) < 0.1:
            top = 0.0375
            node = [3.25e5, 0.04]
        elif np.abs(float(par_list[i]) - 1.80) < 0.1:
            top = v_evol[-1] + 0.0025
            node = [T_list[-1] - 0.25e5, v_evol[-1] + 0.005]
        elif np.abs(float(par_list[i]) - 1.30) < 0.1:
            top = v_evol[-1]
            node = [4.25e5, v_evol[-1] + 0.0025]

        
        print(float(par_list[i]))

        axis.vlines(T_list[-1], 0, top, color=c, linestyle='dotted')
        axis.text(node[0], node[1], 't = {:.2f} Myr'.format(T*P.value_in(units.Myr)), color=c)

        axis2.plot(mg, v_evol, color=c, label=label)
    
    axis2.plot([], [], linestyle='dashed', color='k', label='40 km s-1')
    
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis2.legend(loc='upper center', ncol=1, **props)
    
    
    axis2.vlines

    plt.subplots_adjust(left=0.11, right=0.99, top=0.95, wspace=0)
    plt.savefig('./plots/'+filename+'_spin_evol'+su+'.png')
    plt.close()

def envmass_vs_time(dml_list):
    table = pd.read_table('1_SeBa_radius_short.data', sep="\t", skiprows=1, header=None, index_col=False,
                         names=['M [MSun]', 'M wd [MSun]', 'R ms [RSun]', 'R hg [RSun]', 'R rgb [RSun]', 'R hb [RSun]', 'R agb [RSun]'])
    print(dml_list)
    '''
    for:
        m_env = mdon - (table.iloc[(table['M [MSun]'] - mdon.value_in(mass_unit)).abs().argsort()[:1]].iloc[0]['M wd [MSun]'] | mass_unit)
        t = 0
        while m_env > 0:
            m_env - 
    '''

def angmom_vs_truean(filename, L, L_m, theta, yn):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (4,6), dpi=300, layout='constrained', sharex=True)
    
    a = float(filename.split('/')[-1].split('mdon_')[-1].split('a_')[0].replace('_', '.'))
    e = float(filename.split('/')[-1].split('a_')[-1].split('e_')[0].replace('_', '.'))
    v = float(filename.split('/')[-1].split('e_')[-1].split('vfr_')[0].replace('_', '.'))
    ax1.set_title(f'a = {a}, e = {e},'+r' $v_{extra}/v_{per}$ = '+f'{v}')

    ax1.plot(theta, L.value_in(units.kg * units.m**2 * units.s**(-1)), color='blueviolet')
    ax1.set_ylabel(r'$d L$ [kg m2 s-1]')
    
    ax2.plot(theta, L_m.value_in(units.m**2 * units.s**(-1)), color='blueviolet')
    ax2.set_xlabel('true anomaly [rad]')
    ax2.set_ylabel(r'$d L$ / $d m$ [m2 s-1]')
    
    plt.savefig('./plots/'+filename+'_L_truean_'+yn+'.png')

def momentum_comp(df, parname, cons1, cons2, yn, su):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (4,6), dpi=600, layout='constrained', sharex=True)
    
    props = {'fontsize': 11}

    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    if parname == 'vfr':
        parname = r'$v_{extra}/v ̣{per}$'

    maxs_top = []
    maxs_bot = []
    for i in range(1, df.shape[1], 2):
        print(i)
        c = next(color)
        label = parname + ' = ' + df.columns[i].split('L')[0]

        maxs_top.append(df.iloc[:, [i]].max().iloc[0])
        maxs_bot.append(df.iloc[:, [i+1]].max().iloc[0])
        ax1.plot(df['theta i [rad]'], df.iloc[:, [i]], label=label, color=c)
        ax2.plot(df['theta i [rad]'], df.iloc[:, [i+1]], label=label, color=c)
    
    ax1.set_title(f'{cons1}, {cons2}', **props)
    ax1.set_ylabel(r'$d L$ [kg m2 s-1]', **props)
    ax1.legend(loc=0, **props)
    ax1.set_ylim(bottom=0, top=1.05*max(maxs_top))

    ax2.set_xlabel('true anomaly [rad]', **props)
    ax2.set_ylabel(r'$d L$ / $d m$ [m2 s-1]', **props)
    ax2.set_ylim(bottom=0, top=1.05*max(maxs_bot))

    if yn == 'n':
        ax1.set_xlim(left=np.pi)

    plt.savefig('./plots/momentum_truean_'+yn+su+'.png')

def plot_cons_a(te, tv, econ, vcon, su):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,4), dpi=600, sharey=True)
    
    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    
    ### Left (e) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax1.set_xlabel('$a$ [AU]', **props)
    ax1.set_ylabel(r'$| dm_{acc} / dm_{don} |$', **props)
    for e in te.columns[2:]:
        c = next(color)
        ax1.plot(te['a'], te[e], color=c, label = r'$e$ = '+e)
    ax1.legend(loc='center left', handlelength=1, **props)
    ax1.text(te['a'].iloc[0], 0.925, r'$v_{extra} / v_{per}$ = '+vcon, alpha=0.5, **props)
    
    ### Right (vfr) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax2.set_xlabel('$a$ [AU]', **props)
    for v in tv.columns[2:]:
        c = next(color)
        ax2.plot(tv['a'], tv[v], color=c, label = r'$v_{extra} / v_{per}$ = '+v)
    ax2.legend(loc='center left', handlelength=1, **props)
    ax2.text(tv['a'].iloc[0], 0.925, r'$e$ = '+econ, alpha=0.5, **props)
    
    plt.subplots_adjust(left=0.09, right=0.99, top=0.95, wspace=0)
    plt.savefig('./plots/a_dependence_cons'+su+'.png')
    plt.close()

def plot_cons_e(ta, tv, acon, vcon, su):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,4), dpi=600, sharey=True)
    
    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    
    ### Left (a) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax1.set_xlabel('$e$', **props)
    ax1.set_ylabel(r'$| dm_{acc} / dm_{don} |$', **props)
    for a in ta.columns[2:]:
        c = next(color)
        if a == '1.80':
            lstyle = 'dotted'
        else:
            lstyle = 'solid'
        ax1.plot(ta['e'], ta[a], color=c, label = r'$a$ = '+a+' AU', linestyle=lstyle)
    ax1.legend(loc='center right', handlelength=1, **props)
    ax1.text(0.05, 0.05, r'$v_{extra} / v_{per}$ = '+vcon, alpha=0.5, **props)
    
    ### Right (vfr) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax2.set_xlabel('$e$', **props)
    for v in tv.columns[2:]:
        c = next(color)
        if v == '0.95':
            lstyle = 'dotted'
        else:
            lstyle = 'solid'
        ax2.plot(tv['e'], tv[v], color=c, label = r'$v_{extra} / v_{per}$ = '+v, linestyle=lstyle)
    ax2.legend(loc='center left', handlelength=1, **props)
    ax2.text(0.05, 0.05, r'$a$ = '+acon+' AU', alpha=0.5, **props)
    
    plt.subplots_adjust(left=0.09, right=0.99, top=0.95, wspace=0)
    plt.savefig('./plots/e_dependence_cons'+su+'.png')
    plt.close()

def plot_cons_v(ta, te, acon, econ, su):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,4), dpi=350, sharey=True)
    
    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    
    ### Left (a) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax1.set_xlabel('$v_{extra} / v_{per}$', **props)
    ax1.set_ylabel(r'$| dm_{acc} / dm_{don} |$', **props)
    for a in ta.columns[2:]:
        c = next(color)
        ax1.plot(ta['vfr'], ta[a], color=c, label = r'$a$ = '+a+' AU')
    ax1.legend(loc=4, handlelength=1, **props)
    ax1.text(ta['vfr'].iloc[0], 0.95, r'$e$ = '+econ, alpha=0.5, **props)
    ax1.set_xlim(right=1.015)
    
    ### Right (e) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax2.set_xlabel('$v_{extra} / v_{per}$', **props)
    for e in te.columns[2:]:
        c = next(color)
        ax2.plot(te['vfr'], te[e], color=c, label = r'$e$ = '+e)
    ax2.legend(loc=4, handlelength=1, **props)
    ax2.text(te['vfr'].iloc[0]-0.005, 0.95, r'$a$ = '+acon+' AU', alpha=0.5, **props)
    ax2.set_xlim(left=0.785)
    
    plt.subplots_adjust(left=0.09, right=0.99, top=0.95, wspace=0)
    plt.savefig('./plots/vfr_dependence_cons'+su+'.png')
    plt.close()
    
def plot_spinup_a(te, tv, econ, vcon, su):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,4), dpi=600, sharey=True)
    
    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    
    ### Left (e) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax1.set_xlabel('$a$ [AU]', **props)
    ax1.set_ylabel(r'$\Delta v_{rot} / v_{crit}$', **props)
    for e in te.columns[2:]:
        c = next(color)
        ax1.plot(te['a'], te[e], color=c, label = r'$e$ = '+e)
    ax1.legend(loc=2, handlelength=1, **props)
    ax1.text(te['a'].iloc[0], 0, r'$v_{extra} / v_{per}$ = '+vcon, alpha=0.5, **props)
    
    ### Right (vfr) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax2.set_xlabel('$a$ [AU]', **props)
    for v in tv.columns[2:]:
        c = next(color)
        ax2.plot(tv['a'], tv[v], color=c, label = r'$v_{extra} / v_{per}$ = '+v)
    ax2.legend(loc=1, handlelength=1, **props)
    ax2.text(tv['a'].iloc[0], 2.35e-7, r'$e$ = '+econ, alpha=0.5, **props)
    
    plt.subplots_adjust(left=0.09, right=0.99, top=0.95, wspace=0)
    plt.savefig('./plots/a_dependence'+su+'.png')
    plt.close()

def plot_spinup_e(ta, tv, acon, vcon, su):
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
    ax1.legend(loc=1, handlelength=1, **props)
    ax1.text(0.05, 0.25e-7, r'$v_{extra} / v_{per}$ = '+vcon, alpha=0.5, **props)
    ax1.set_ylim(top=2.75e-7)

    ### Right (vfr) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax2.set_xlabel('$e$', **props)
    for v in tv.columns[2:]:
        c = next(color)
        ax2.plot(tv['e'], tv[v], color=c, label = r'$v_{extra} / v_{per}$ = '+v)
    ax2.legend(loc=(0.06, 0.675), handlelength=1, **props)
    ax2.text(0.05, 0.25e-7, r'$a$ = '+acon+' AU', alpha=0.5, **props)
    
    plt.subplots_adjust(left=0.09, right=0.99, top=0.95, wspace=0)
    plt.savefig('./plots/e_dependence'+su+'.png')
    plt.close()
    
def plot_spinup_v(ta, te, acon, econ, su):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,4), dpi=350, sharey=True)
    
    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    
    ### Left (a) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax1.set_xlabel('$v_{extra} / v_{per}$', **props)
    ax1.set_ylabel(r'$\Delta v_{rot} / v_{crit}$', **props)
    for a in ta.columns[2:]:
        c = next(color)
        ax1.plot(ta['vfr'], ta[a], color=c, label = r'$a$ = '+a+' AU')
    ax1.legend(loc=2, handlelength=1, **props)
    ax1.text(0.95, 2.25e-7, r'$e$ = '+econ, alpha=0.5, **props)
    ax1.set_xlim(right=1.015)
    
    ### Right (e) panel ###
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    ax2.set_xlabel('$v_{extra} / v_{per}$', **props)
    for e in te.columns[2:]:
        c = next(color)
        ax2.plot(te['vfr'], te[e], color=c, label = r'$e$ = '+e)
    ax2.legend(loc=2, handlelength=1, **props)
    ax2.text(0.925, 2.25e-7, r'$a$ = '+acon+' AU', alpha=0.5, **props)
    ax2.set_xlim(left=0.785)
    
    plt.subplots_adjust(left=0.09, right=0.99, top=0.95, wspace=0)
    plt.savefig('./plots/vfr_dependence'+su+'.png')
    plt.close()

def plot_momentum_a(par, df, parname, varname, con1, con2, frac_i, T, su):
    fig, axis = plt.subplots(figsize = (6,4), dpi=600, layout='constrained')
    
    props = {'fontsize': 11}

    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))

    if parname == 'a':
        axname, title, filename = from_pars(parname, con1, con2, frac_i, T, 'spin-up comparison')
        if varname == 'e':
            node_str = r'$v_{extra}/v_{per}$'+' = {:=04.2f}'.format(con2)
            #axis.text(1.225, -0.00015, node_str, alpha=0.5, **props)
            axis.text(1.525, 1.75e-6, node_str, alpha=0.5, **props)
        elif varname == 'vfr':
            node_str = r'$e$'+' = {:=04.2f}'.format(con1)
            axis.text(2.1, 0.1e-6, node_str, alpha=0.5, **props)
    else:
        print('No available plot :c')

    for i in range(1, df.shape[1]):
        c = next(color)
        if varname == 'e':
                label = 'e = {:.2f}'.format(float(df.columns[i]))
                if float(df.columns[i]) == 0.60:
                    linestyle = 'dashed'
                else:
                    linestyle = 'solid'
        elif varname == 'vfr':
            label = '$v_{extra}/v_{per}$'+' = {:.2f}'.format(float(df.columns[i]))
        #axis.plot(par, df.iloc[:, [i]], label=label, color=c, linestyle=linestyle)
        #print(label, '\n', df.iloc[:, i], '\n', np.array(par), '\n', df.iloc[:, i]/np.array(par))
        axis.plot(par, df.iloc[:, i]/np.array(par), label=label, color=c, linestyle=linestyle)
    
    axis.set_xlabel(r'$a_i$ [AU]', **props)
    #axis.set_ylabel(r'$\Delta a$ [AU]', **props)
    axis.set_ylabel(r'$\Delta a / a_i$', **props)
    
    axis.hlines(0, min(par), max(par), linestyle='dotted', color='black')
    
    axis.legend(loc='best', **props)
    axis.set_xlim(left=min(par), right=max(par))
    
    axis.text(1.275, 0.05e-6, 'expansion', **props)
    axis.text(1.275, -0.15e-6, 'shrinkage', **props)

    axis.text(1.4, -0.65e-6, 'conservative', alpha=0.5, rotation=-7, **props)
    axis.text(1.95, 0.75e-6, 'non-conservative', alpha=0.5, rotation=15, **props)

    # X's marking the change in regime (selected manually)
    axis.scatter(par[114], df.iloc[114, 1]/np.array(par[114]), color='firebrick', marker='x', s=75)
    axis.scatter(par[115], df.iloc[115, 2]/np.array(par[115]), color='gold', marker='x', s=75)
    axis.scatter(par[117], df.iloc[117, 3]/np.array(par[117]), color='limegreen', marker='x', s=75)
    axis.scatter(par[148], df.iloc[148, 4]/np.array(par[148]), color='royalblue', marker='x', s=75)

    plt.savefig('./plots/'+filename+'_da'+su+'.png')
    plt.close()


if __name__ == "__main__":
    print('ola')

#density_profile(1.2 | mass_unit, 263.648425528 | radius_unit)
#orbit_example('./a_00_0000e/01_00macc_01_20mdon_0002_50a_00_0000e_-025_16rot', 1 | radius_unit, 400)
#orbits_example('./v/01_00macc_01_20mdon_0001_00a_00_30e_-056_48rot', 1 | radius_unit, 400, 10)
#v_vs_pars('./a.dat', './e.dat', './v.dat')

#orbits_example('./data/e_001_000a_0_90vfr_000_00vexp/01_00macc_01_20mdon_0001_00a_00_1080e_00_90vfr_-044_32rot', 1 | radius_unit, 400, 10)

#plot_larim('./v/01_00macc_01_20mdon_0001_00a_00_30e_-056_48rot', 1 | radius_unit, [20 | time_unit, 40 | time_unit, 60 | time_unit, 80 | time_unit])

df = pd.read_csv('./a_e_0.90_da_table.csv', index_col=0)
plot_momentum_a(df['a'], df, 'a', 'e', 0.6, 0.90, 0.1, 0 | units.day, '_ai')