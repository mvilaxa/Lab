import os
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles, Particle, ParticlesSuperset
from amuse.units.optparse import OptionParser

from plotting import * #accretion_space, spinup_per_period, angmom_vs_truean, momentum_vs_par, spinup, conservativeness_plot, orbits_example, spinup_comp, orbits_stream, plot_spinup_a, plot_spinup_e, plot_spinup_v, spinup_vs_gm, plot_cons_a, plot_cons_e, plot_cons_v, stream_animation
import special_plots as sp

def read_file(filename):
    
    macc = float(((filename.split('/')[-1]).split('macc')[0]).replace('_', '.')) | units.MSun
    mdon = float((((filename.split('/')[-1]).split('macc_')[-1]).split('mdon')[0]).replace('_', '.')) | units.MSun
    a = float((((filename.split('/')[-1]).split('mdon_')[-1]).split('a')[0]).replace('_', '.')) | units.au
    e = float((((filename.split('/')[-1]).split('a_')[-1]).split('e')[0]).replace('_', '.'))
    v_fr = float((((filename.split('/')[-1]).split('e_')[-1]).split('vfr')[0]).replace('_', '.'))
    v_extra = float((((filename.split('/')[-1]).split('vfr_')[-1]).split('rot')[0]).replace('_', '.')) | units.km * units.s**(-1)
    df = pd.read_csv(filename+'.csv')
    
    return macc, mdon, a, e, v_fr, v_extra, df


def add_fraction(df, frac_i, yn):
    r_don = (df.iloc[0]['r orb [AU]'] - df.iloc[0]['r L1 [AU]']) / (1 - frac_i)
    frac_list, newflag_list = [[], []]
    if yn == 'n':
        for i in df.index.to_list():    ### 1st condition ###
            frac_list.append(1 - (df.iloc[i]['r orb [AU]'] - df.iloc[i]['r L1 [AU]']) / (r_don))
            if df.iloc[i]['ang i [rad]'] > df.iloc[i]['ang orb [rad]']:    ### 3rd condition ###
                newflag_list.append(0)
            else:
                newflag_list.append(1)
        df['new flag'] = newflag_list
    else:
        for i in df.index.to_list():
            frac_list.append(1 - (df.iloc[i]['r orb [AU]'] - df.iloc[i]['r L1 [AU]']) / (r_don))
    
    frac_list = [0 if x < 0 else x for x in frac_list]
    df['fraction'] = frac_list
    
    return df

def momentum_contribution_mc(df, racc, mc):
    impact = (df.loc[df['fraction'] > 0]).loc[df['flag impact'] == 1.0]
    print(len(impact.index))
    if len(impact.index) > 0:
        v_t = impact['v imp [km s-1]'].astype('float') * np.sin(impact['ang imp [rad]'].astype('float'))
        L = racc.value_in(units.km) * v_t * impact['fraction'] * mc.value_in(units.kg)
        L_tot = L.cumsum().iloc[-1] | units.kg * units.km**2 * units.s**(-1)
        
        dm = impact['fraction'].cumsum().iloc[-1] * mc
    else:
        L_tot = 0 | units.kg * units.km**2 * units.s**(-1)
        dm = 0 | units.kg
    
    return L_tot, dm

def momentum_contribution_truean(filename, df, racc, T, mtr, yn):
    impact = df
    idxs1 = df.loc[df['fraction'] <= 0].index
    idxs2 = df.loc[df['flag impact'] == 0.0].index
    impact.loc[idxs1, 'v imp [km s-1]'] = np.zeros(len(idxs1))
    impact.loc[idxs2, 'v imp [km s-1]'] = np.zeros(len(idxs2))
    
    if 'new flag' in df.columns:
        idxs = impact.loc[impact['new flag'] == 0].index
        impact.loc[idxs, 'v imp [km s-1]'] = np.zeros(len(idxs))
        dm = mtr * T * 0.5
    else:
        dm = mtr * T
    
    if len(impact.index) > 0:
        m = (dm.value_in(units.kg) / impact['fraction'].cumsum().iloc[-1]) * impact['fraction']
        
        # Temporary solution for error in impact angle
        a_p = impact['a p [AU]']
        e_p = impact['e p']
        theta_imp = 2*np.pi - np.arccos(((a_p * (1 - e_p**2) / racc.value_in(units.au)) - 1) / e_p)
        aux_ang = np.arccos((4*a_p**2 - 4*a_p*racc.value_in(units.au) + 2*racc.value_in(units.au)**2 - 4*(a_p*e_p)**2)/(4*a_p*racc.value_in(units.au) - 2*racc.value_in(units.au)**2))
        ang_imp = (np.pi - aux_ang) / 2

        v_t = impact['v imp [km s-1]'].astype('float') * np.sin(ang_imp)

        L_m = racc.value_in(units.km) * v_t * impact['fraction']
        L = L_m * m
    else:
        L_m = 0
        L = 0
        dm = 0 | units.kg
    
    L_m = (L_m).values.tolist()
    L = (L).values.tolist()
    #angmom_vs_truean(filename, L | units.kg * units.km**2 * units.s**(-1) , L_m | units.km**2 * units.s**(-1), impact['theta i [rad]'], yn)
    #L_elements(impact['theta i [rad]'], v_t, impact['fraction'], L_m, filename)
    return impact['theta i [rad]'], L, L_m, impact['fraction'], v_t

def kin_energies(df, racc, T, mtr):
    impact = (df.loc[df['fraction'] > 0]).loc[df['flag impact'] == 1.0]
    if len(impact.index) > 0:
        dm = mtr * T
        m = (dm.value_in(units.kg) / impact['fraction'].cumsum().iloc[-1]) * impact['fraction']
        v_t = impact['v imp [km s-1]'].astype('float') * np.sin(impact['ang imp [rad]'].astype('float'))
        v_r = impact['v imp [km s-1]'].astype('float') * np.cos(impact['ang imp [rad]'].astype('float'))
        
        E_t = 0.5 * (m * v_t**2)    #In kg km2 s-2
        E_r = 0.5 * (m * v_r**2)
        E_tot = 0.5 * (m * impact['v imp [km s-1]'].astype('float')**2)
        E_sum = E_t + E_r
        E_p = np.sqrt(E_t**2 + E_r**2)
    else:
        E_t = 0
        E_r = 0
        E_tot = 0
        E_sum = 0
        E_p = 0
        
    print((E_tot-E_sum).cumsum().iloc[-1])
    print((E_tot-E_p).cumsum().iloc[-1])

def momentum_mtr(df, racc, T, mtr):
    dm_lost = mtr * T    #Total mass the donor loses in a single period
    df = df.iloc[1::]
    #print(df)
    impact0 = df.loc[(df['fraction'] > 0.0) & (df['flag impact'] == 1.0)]    #Takes all instances where
                                                                            #both conditions are met
    if len(impact0.index) > 0:
        if 'new flag' in df.columns:
            impact = impact0.loc[impact0['new flag'] == 1]
            frac_sum = df.loc[df['new flag'] == 1]['fraction'].cumsum().iloc[-1]
        else:
            impact = impact0
            frac_sum = df['fraction'].cumsum().iloc[-1]
        
        if len(impact.index) > 0:
            m = (dm_lost.value_in(units.kg) / frac_sum) * impact['fraction']
            dm_gained = m.cumsum().iloc[-1] | units.kg

            # Temporary solution for error in impact angle
            a_p = impact['a p [AU]']
            e_p = impact['e p']
            theta_imp = 2*np.pi - np.arccos(((a_p * (1 - e_p**2) / racc.value_in(units.au)) - 1) / e_p)
            aux_ang = np.arccos((4*a_p**2 - 4*a_p*racc.value_in(units.au) + 2*racc.value_in(units.au)**2 - 4*(a_p*e_p)**2)/(4*a_p*racc.value_in(units.au) - 2*racc.value_in(units.au)**2))
            ang_imp = (np.pi - aux_ang) / 2

            v_t = impact['v imp [km s-1]'].astype('float') * np.sin(ang_imp)
            v_r = impact['v imp [km s-1]'].astype('float') * np.cos(ang_imp)
            
            #print(len(m), len(v_t), impact0['ang orb [rad]'] - impact0['ang i [rad]'])
            
            pt = m * v_t
            pr = m * v_r
            p = m * impact['v imp [km s-1]'].astype('float')
            L = racc.value_in(units.km) * v_t * impact['fraction'] * m
            
            p_t = pt.cumsum().iloc[-1] | units.kg * units.km * units.s**(-1)
            p_r = pr.cumsum().iloc[-1] | units.kg * units.km * units.s**(-1)
            p_total = p.cumsum().iloc[-1] | units.kg * units.km * units.s**(-1)
            L_tot = L.cumsum().iloc[-1] | units.kg * units.km**2 * units.s**(-1)
        else:
            dm_gained = 0 | units.kg
            p_t = 0 | units.kg * units.km * units.s**(-1)
            p_r = 0 | units.kg * units.km * units.s**(-1)
            p_total = 0 | units.kg * units.km * units.s**(-1)
            L_tot = 0 | units.kg * units.km**2 * units.s**(-1)
    
    else:
        dm_gained = 0 | units.kg
        p_t = 0 | units.kg * units.km * units.s**(-1)
        p_r = 0 | units.kg * units.km * units.s**(-1)
        p_total = 0 | units.kg * units.km * units.s**(-1)
        L_tot = 0 | units.kg * units.km**2 * units.s**(-1)
    
    return dm_lost, dm_gained, p_t, p_r, p_total, L_tot

def conservativeness(df):
    if 'new flag' in df.columns:
        impact = ((df.loc[df['fraction'] > 0]).loc[df['flag impact'] == 1.0]).loc[df['new flag'] == 1]
        nonimpact = (df.loc[df['fraction'] > 0]).loc[(df['flag impact'] == 0.0) | (df['new flag'] == 0)]
    else:
        impact = (df.loc[df['fraction'] > 0]).loc[df['flag impact'] == 1.0]
        nonimpact = (df.loc[df['fraction'] > 0]).loc[df['flag impact'] == 0.0]

    if len(impact.index) == 0:
        mg_fracsum = 0
    else:
        mg_fracsum = impact['fraction'].cumsum().iloc[-1]
        
    if len(nonimpact.index) == 0:
        ml_fracsum = 0
    else:
        ml_fracsum = nonimpact['fraction'].cumsum().iloc[-1]
    
    mtot_fracsum = mg_fracsum + ml_fracsum
    return mg_fracsum/mtot_fracsum, ml_fracsum/mtot_fracsum

def comparison_momentum(f_list, dirname, parname, frac_i, racc, mtr, yn, su):
    df_mom = pd.DataFrame()
    df_other = pd.DataFrame()
    flag = 0
    for f in f_list:
        macc, mdon, a, e, v_fr, v_extra, df = read_file(os.path.join(dirname, f))
        
        if parname == 'a':
            colname = str(a)
            cons1 = f'e = {e}'
            cons2 = r'$v_{extra}/v_{per}$ = '+f'{v_fr}'
        elif parname == 'e':
            colname = str(e)
            cons1 = f'a = {a} AU'
            cons2 = r'$v_{extra}/v_{per}$ = '+f'{v_fr}'
        elif parname == 'vfr':
            colname = str(v_fr)
            cons1 = f'a = {a} AU'
            cons2 = f'e = {e}'

        df_frac = add_fraction(df, frac_i, yn)
        T = 2 * np.pi * np.sqrt((a**3) / (constants.G * (macc + mdon)))

        truean, L, L_m, fr, v = momentum_contribution_truean(f, df_frac, racc, T, mtr, yn)

        if flag == 0:
            df_mom['theta i [rad]'] = truean
            df_other['theta i [rad]'] = truean
            flag = 1
        
        df_mom[colname+'L'] = L
        df_mom[colname+'Lm'] = L_m

        df_other[colname+'f'] = fr
        df_other[colname+'v'] = v
    
    #momentum_comp(df_mom, parname, cons1, cons2, yn, su)
    #L_elements_comp(df_other, parname, su)
    sp.velocities(df_other, parname, su)

def fraction(datname, frac_i, racc, mc, mtr, yn, su):
    parname = datname.split('_')[0]
    table = pd.read_table(datname, header=None, names=['filenames'])
    L_list, dmg_list, dml_list, dv_list, par_list, p_list, pt_list, pr_list , mg_list, ml_list = [[], [], [], [], [], [], [], [], [], []]
    for f in table['filenames']:
        macc, mdon, a, e, v_fr, v_extra, df = read_file('./data/'+datname.split('.')[0]+'/'+f)
        df_frac = add_fraction(df, frac_i, yn)
        T = 2 * np.pi * np.sqrt((a**3) / (constants.G * (macc + mdon)))
        
        dm_l, dm_g, p_t, p_r, p, L_tot = momentum_mtr(df_frac, racc, T, mtr)
        mg, ml = conservativeness(df_frac)
        
        
        dv = L_tot / (racc * (macc + dm_g))
        v_crit = (constants.G * (macc + dm_g) / racc)**0.5
        L_list.append(L_tot.value_in(units.kg * units.m**2 * units.s**(-1)))
        dmg_list.append(dm_g.value_in(units.kg))
        dml_list.append(dm_l.value_in(units.kg))
        dv_list.append(dv/v_crit)
        
        p_list.append(p.value_in(units.kg * units.m * units.s**(-1)))
        pt_list.append(p_t.value_in(units.kg * units.m * units.s**(-1)))
        pr_list.append(p_r.value_in(units.kg * units.m * units.s**(-1)))
        
        mg_list.append(mg)
        ml_list.append(ml)
        
        if parname == 'a':
            par_list.append(a.value_in(units.au))
            con1 = e
            con2 = v_fr
        elif parname == 'e':
            par_list.append(e)
            con1= a
            con2 = v_fr
        elif parname == 'vfr':
            par_list.append(v_fr)
            con1 = a
            con2 = e
    
        #print(parname+' = ', par_list[-1], ', spinup = ', dv/v_crit)
    
    #spinup(par_list, dv_list, parname, con1, con2, frac_i, T, '_'+su)
    #momentum_vs_par(par_list, p_list, pt_list, pr_list, parname, con1, con2, frac_i, T, '_'+su)
    #conservativeness_plot(par_list, np.array(mg_list), np.array(ml_list), parname, con1, con2, frac_i, T, '_'+su)
    spinup_per_period(par_list, L_list, dmg_list, dml_list, parname, con1, con2, frac_i, T, macc, racc, mdon, su)
    #spinup_vs_gm(par_list, L_list, dmg_list, dml_list, parname, con1, con2, frac_i, T, macc, racc, mdon, su)
    #envmass_vs_time(dml_list)

def comparison(table_list, frac_i, racc, mc, mtr, yn, su):
    parname = table_list[0].split('_')[0]
    
    # Identify the varying parameter between tables
    name1 = table_list[0].split(parname+'_')[-1]
    name2 = table_list[1].split(parname+'_')[-1]
    if parname == 'a':
        e1 = name1.split('e_')[0]
        e2 = name2.split('e_')[0]
        if e1 != e2:
            varname = 'e'
            str_range = [0, 6]
            conname = 'vfr'
        elif e1 == e2:
            varname = 'vfr'
            str_range = [9, 12]
            conname = 'e'
    elif parname == 'e':
        a1 = name1.split('a_')[0]
        a2 = name2.split('a_')[0]
        if a1 != a2:
            varname = 'a'
            str_range = [0, 6]
            conname = 'vfr'
        elif a1 == a2:
            varname = 'vfr'
            str_range = [9, 12]
            conname = 'a'
    elif parname == 'vfr':
        a1 = name1.split('a_')[0]
        a2 = name2.split('a_')[0]
        if a1 != a2:
            varname = 'a'
            str_range = [0, 6]
            conname = 'e'
        elif a1 == a2:
            varname = 'e'
            str_range = [9, 15]
            conname = 'a'
    
    df_plotting = pd.DataFrame()
    df_cons = pd.DataFrame()
    df_transf = pd.DataFrame()
    flag = 0
    for t in table_list:
        table = pd.read_table(t, header=None, names=['filenames'])
        L_list, dv_list, par_list, mg_list, da_list = [[], [], [], [], []]
        for f in table['filenames']:
            macc, mdon, a, e, v_fr, v_extra, df = read_file('./data/'+t.split('.')[0]+'/'+f)
            df_frac = add_fraction(df, frac_i, yn)
            
            T = 2 * np.pi * np.sqrt((a**3) / (constants.G * (macc + mdon)))
            mu = (macc * mdon) / (macc + mdon)
            L_orb = (mu * 2 * np.pi * a**2) / T

            dm_l, dm_g, p_t, p_r, p, L_tot = momentum_mtr(df_frac, racc, T, mtr)
            
            muf = ((macc+dm_g) * (mdon-dm_l)) / (macc + mdon + dm_g - dm_l)
            af = (L_orb**2 / (constants.G * (macc + mdon))) * muf**-2

            dv = L_tot / (racc * (macc + dm_g))
            v_crit = (constants.G * (macc + dm_g) / racc)**0.5
            L_list.append(L_tot.value_in(units.kg * units.m**2 * units.s**(-1)))
            dv_list.append(dv/v_crit)
            mg_list.append(dm_g/dm_l)
            da_list.append((af - a).value_in(units.au))
            if parname == 'a':
                par_list.append(a.value_in(units.au))
                con1 = e
                con2 = v_fr
            elif parname == 'e':
                par_list.append(e)
                con1 = a
                con2 = v_fr
            elif parname == 'vfr':
                par_list.append(v_fr)
                con1 = a
                con2 = e
        if flag == 0:
            df_plotting[parname] = par_list
            df_cons[parname] = par_list
            df_transf[parname] = par_list
            f = 1
        
        colname = '{:04.2f}'.format(float(t.split(parname+'_')[1][str_range[0]:str_range[1]+1].replace('_', '.')))
        print(t, colname)
        #colname = t
        df_plotting[colname] = dv_list
        df_cons[colname] = mg_list
        df_transf[colname] = da_list
    
    if conname == 'a':
        constant = a.value_in(units.au)
    
    elif conname == 'e':
        constant = e
    
    elif conname == 'vfr':
        constant = v_fr
    
    #df_plotting.to_csv(parname+'_'+varname+'_{:04.2f}_table.csv'.format(constant))
    #df_cons.to_csv(parname+'_'+varname+'_{:04.2f}_cons_table.csv'.format(constant))
    df_transf.to_csv(parname+'_'+varname+'_{:04.2f}_da_table.csv'.format(constant))
    
    #spinup_comp(par_list, df_plotting, parname, varname, con1, con2, frac_i, T, '_'+su)
    #sp.plot_spinup_a(par_list, df_plotting, r'$e$ = 0.10, $v_{extra}/v_{per}$ = 0.90', r'$q$', [0.25, 0.50, 0.75, 1.00], su)
    #plot_momentum_a(par_list, df_transf, parname, varname, con1, con2, frac_i, T, su)

def new_option_parser():
    result = OptionParser()
    result.add_option("--fname",
                      dest="fname", type="str",
                      default = None,
                      help="name of file to read")
    result.add_option("-t",
                      dest="tname", type="str",
                      default = None,
                      help="name of table with filenames to read")
    result.add_option("-m",
                      dest="mtrexp", type="float",
                      default = -4.,
                      help="power of mass transfer rate")
    result.add_option("-f",
                      dest="f", type="float",
                      default = 0.1,
                      help="overfill fraction")
    result.add_option("--racc", unit=units.RSun,
                      dest="racc", type="float",
                      default = 1.,
                      help="accretor radius")
    result.add_option("--yn",
                      dest="yn", type="string",
                      default = 'y',
                      help="to include (y) or not to include (n) orbits beyond L1")
    result.add_option("--su",
                      dest="su", type="string",
                      default = '',
                      help="sufix for plots")
    result.add_option("--comp",
                      dest="comp", type="string",
                      default = None,
                      help="csv files to compare (use a comma for separation)")
    result.add_option("--stream",
                      dest="stream", type="string",
                      default = 'n',
                      help="whether to plot (y) or not (n) stream snapshots")
    result.add_option("--join",
                      dest="t12", type="string",
                      default = None,
                      help="two tables to be plotted together (separated by a comma)")
    return result

if __name__ == "__main__":

    o, arguments  = new_option_parser().parse_args()

    mtr = 10**o.mtrexp | units.MSun * units.yr**(-1)
    
    if o.fname != None:
        macc, mdon, a, e, v_fr, v_extra, df = read_file('./'+o.fname)
        df_frac = add_fraction(df, o.f, o.yn)
        new_dir = './data/'+o.fname.split('/')[-1]+'_{:=05.2f}f'.format(o.f)
        df_frac.to_csv(new_dir+'.csv')
        
        #orbit_example(new_dir, o.racc, 0, 10000, o.su)
        #orbits_example(new_dir, o.racc, 1000, 50, o.su)
        #plot_conditions(new_dir, o.racc, o.su)
        conditions_svg(new_dir, o.racc, o.su)

        T = 2 * np.pi * np.sqrt((a**3) / (constants.G * (macc + mdon)))
        #momentum_mtr(df_frac, o.racc, T, mtr)
        #momentum_contribution_truean(o.fname.split('/')[-1], df_frac, o.racc, T, mtr, o.yn)
        if o.stream == 'y':
            stream_animation(new_dir, o.racc, T)
            
            #for i in range(df_frac.index[-1]):
            for i, t in enumerate(range(1, int(T.value_in(units.day)), 1)):
                orbits_stream(new_dir, o.racc, T, t | units.day, i)
            

    if o.tname != None:
        if o.tname.split('_')[0] in ['a', 'e', 'vfr']:
            fraction(o.tname, o.f, o.racc, None, mtr, o.yn, o.su)

    if o.comp != None:
        comparison(o.comp.split(','), o.f, o.racc, None, mtr, o.yn, o.su)

    if o.t12 != None:
        tname1, tname2 = o.t12.split(',')
        t1 = pd.read_csv(tname1)
        t2 = pd.read_csv(tname2)
        con1 = tname1.split('_')[2]
        con2 = tname2.split('_')[2]
        
        if 'cons' in o.t12:
            if tname1[0] == 'a':
                plot_cons_a(t1, t2, con2, con1, o.su)
            elif tname1[0] == 'e':
                plot_cons_e(t1, t2, con2, con1, o.su)
            elif tname1[0] == 'v':
                plot_cons_v(t1, t2, con2, con1, o.su)
        
        else:
            if tname1[0] == 'a':
                plot_spinup_a(t1, t2, con2, con1, o.su)
            elif tname1[0] == 'e':
                plot_spinup_e(t1, t2, con2, con1, o.su)
            elif tname1[0] == 'v':
                plot_spinup_v(t1, t2, con2, con1, o.su)
    