#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:43:39 2017

%reset -f
%clear
%pylab
%load_ext autoreload
%autoreload 2

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

from numpy.linalg import norm

import seaborn as sns

#rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
#      'font.sans-serif': 'Helvetica'}
rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Arial'}
sns.set('notebook', 'ticks', font='Helvetica',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_forces/{}.pdf'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}


# %%

from matplotlib.ticker import FuncFormatter

def _formatter_degree(x, pos):
    """Add a degree symbol.
    """

    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x) + u'\u00B0'
#    if np.abs(x) > 0 and np.abs(x) < 1:
#        return val_str.replace("0", "", 1)
#    else:
#        return val_str
    return val_str

degree_formatter = FuncFormatter(_formatter_degree)


# %% Downsample a trial, then try fitting a spline

def ret_fnames(snake=None, trial=None):

    from glob import glob

    if snake is None:
        snake = '*'
    if trial is None:
        trial = '*'

    fn_trial = '{0}_{1}.npz'.format(trial, snake)
    fn_proc = '../Data/Processed Qualisys output/'
    fn_search = fn_proc + fn_trial

    return sorted(glob(fn_search))


def trial_info(fname):
    trial_id = fname.split('/')[-1][:3]
    snake_id = fname.split('/')[-1][4:6]

    return int(snake_id), int(trial_id)


fnames = ret_fnames()
ntrials = len(fnames)

# average start in X and Y of com as the new
X0, Y0 = np.zeros(ntrials), np.zeros(ntrials)
for i, fname in enumerate(fnames):
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)
    X, Y, Z = d['Ro_I'].T / 1000  # m
    X0[i] = X[0]
    Y0[i] = Y[0]
Xo = X0.mean()
Yo = Y0.mean()


# masses for labels
snake_ids = [81, 91, 95, 88, 90, 86, 94]
masses = []
masses_std = []
for snake_id in snake_ids:
    fnames = ret_fnames(snake_id)
    mm = []
    for fname in fnames:
        d = np.load(fname)
        mm.append(float(d['mass']))
    mm = np.array(mm)
    masses.append(mm.mean())
    masses_std.append(mm.std())

masses = np.array(masses)
masses_std = np.array(masses_std)


# indices to use (determined from the ball drop experiment)
start = 8
stop = -10

g = 9.81

# C. paradisi to analyze
snake_ids = [81, 91, 95, 88, 90, 86, 94]
nsnakes = len(snake_ids)
masses = np.zeros(nsnakes)

# interpolation grid for forces
Zi = np.linspace(8.5, 0, 200)

# colors for plots
colors = sns.color_palette('husl', nsnakes)


# %% Yaw and mu variabtion

# indices to use (determined from the ball drop experiment)
start = 8
stop = -10

snake_ids = [81, 91, 95, 88, 90, 86, 94]
nsnakes = len(snake_ids)



fig, axs = plt.subplots(7, 3, figsize=(9, 12), sharex=True, sharey=True)

sns.despine()

ax = axs[0, 0]
ax.invert_xaxis()
ax.set_xlim(8.5, 0)

for ax in axs.flatten():
    ax.axhline(0, color='gray', lw=1)


for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    ntrials = len(fn_names)
    colors_trial_id = colors[row]

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times'][start:stop]
        Z = d['Ro_S'][start:stop, 2] / 1000


        yaw = np.rad2deg(d['yaw'][start:stop])
        mus = np.rad2deg(d['mus'][start:stop]).cumsum()

        yaw -= yaw[0]

        axs[row, 0].plot(Z, yaw, c=colors_trial_id)
        axs[row, 1].plot(Z, mus, c=colors_trial_id)
        axs[row, 2].plot(Z, yaw - mus, c=colors_trial_id)


# %% Interpolate and errors in dictionaries

# indices to use (determined from the ball drop experiment)
start = 8
stop = -10

g = 9.81

# C. paradisi to analyze
snake_ids = [81, 91, 95, 88, 90, 86, 94]
nsnakes = len(snake_ids)
masses = np.zeros(nsnakes)

# interpolation grid for forces
Zi = np.linspace(8.5, 0, 200)
#zr = np.zeros((nsnakes, len(ZZ)))
#Fa_avg, Fa_std = zr.copy(), zr.copy()
#ddRo_avg, ddRo_std = zr.copy(), zr.copy()


# %% Minimze ||ddRo - Fl + Fd|| by changing an average boost to Fd and Fl

from scipy.optimize import minimize

def to_min(boosts, args):
    boost_Fl, boost_Fd = boosts
    Fl, Fd, ddRo = args

    Fa_b = (boost_Fl * Fl + boost_Fd * Fd).sum(axis=0)

    error = np.linalg.norm(ddRo - Fa_b)
    return error


Sn, In = {}, {}
for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    ntrials = len(fn_names)
    colors_trial_id = colors[row]
    nheight = len(Zi)

    ddRo_all = np.zeros((ntrials, nheight, 3))
    Fa_all = np.zeros((ntrials, nheight, 3))
    Fl_all = np.zeros((ntrials, nheight, 3))
    Fd_all = np.zeros((ntrials, nheight, 3))
    Fa_all_p = np.zeros((ntrials, nheight, 3))
    Fl_all_p = np.zeros((ntrials, nheight, 3))
    Fd_all_p = np.zeros((ntrials, nheight, 3))
    boost_all = np.zeros((ntrials, nheight, 3))
    trial_ids = np.zeros(ntrials, dtype=np.int)

    Fa_pererr = np.zeros((ntrials, nheight))
    Fl_pererr = np.zeros((ntrials, nheight))
    Fd_pererr = np.zeros((ntrials, nheight))


    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        trial_ids[i] = trial_id
        d = np.load(fname)

        times = d['times'][start:stop]
        Z = d['Ro_S'][start:stop, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * g

        masses[row] = float(d['mass'])

        # forces and accelerations in STRAIGHTENED frame
        ddRo = d['ddRo_S'][start:stop] / 1000
        Fl = d['Fl_S'][start:stop]
        Fd = d['Fd_S'][start:stop]
        Fa = Fl + Fd
        Fa_non, Fl_non, Fd_non = Fa / mg, Fl / mg, Fd / mg
        ddRo_non = ddRo / g
        ddRo_non[:, 2] += 1  # move gravity onto acceleration

        boost_Fl, boost_Fd = 1.7, .65
        boosts0 = (boost_Fl, boost_Fd)
        ntime, nbody = Fa_non.shape[0], Fa_non.shape[1]

        boost = np.zeros((ntime, 3))
        for ii in np.arange(ntime):
            args = (Fl_non[ii], Fd_non[ii], ddRo_non[ii])
            res = minimize(to_min, boosts0, args=(args,))
            bl, bd = res.x
            bld = bl / bd
            boost[ii] = bl, bd, bld

        Fa_non = Fa_non.sum(axis=1)
        Fl_non = Fl_non.sum(axis=1)
        Fd_non = Fd_non.sum(axis=1)

        # apply the boosts
        Fl_non_b = (Fl_non.T * boost[:, 0]).T
        Fd_non_b = (Fd_non.T * boost[:, 1]).T
        Fa_non_b = Fl_non_b + Fd_non_b

        # interpolate
        for k in np.arange(3):
            ddRo_all[i, :, k] = interp1d(Z, ddRo_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)

            Fa_all[i, :, k] = interp1d(Z, Fa_non[:, k], bounds_error=False,
                                       fill_value=np.nan)(Zi)
            Fl_all[i, :, k] = interp1d(Z, Fl_non[:, k], bounds_error=False,
                                       fill_value=np.nan)(Zi)
            Fd_all[i, :, k] = interp1d(Z, Fd_non[:, k], bounds_error=False,
                                       fill_value=np.nan)(Zi)

            # boosted
            Fa_all_p[i, :, k] = interp1d(Z, Fa_non_b[:, k], bounds_error=False,
                                       fill_value=np.nan)(Zi)
            Fl_all_p[i, :, k] = interp1d(Z, Fl_non_b[:, k], bounds_error=False,
                                       fill_value=np.nan)(Zi)
            Fd_all_p[i, :, k] = interp1d(Z, Fd_non_b[:, k], bounds_error=False,
                                       fill_value=np.nan)(Zi)

        for k in np.arange(3):
            boost_all[i, :, k] = interp1d(Z, boost[:, k], bounds_error=False,
                                           fill_value=np.nan)(Zi)

        # percent error for forces
        Fa_norm = norm(Fa_all[i], axis=1)
        Fl_norm = norm(Fl_all[i], axis=1)
        Fd_norm = norm(Fd_all[i], axis=1)
        Fa_norm_p = norm(Fa_all_p[i], axis=1)
        Fl_norm_p = norm(Fl_all_p[i], axis=1)
        Fd_norm_p = norm(Fd_all_p[i], axis=1)

        # (exp - theor) / theor * 100
        Fa_pererr[i] = (Fa_norm_p - Fa_norm) / Fa_norm_p * 100
        Fl_pererr[i] = (Fl_norm_p - Fl_norm) / Fl_norm_p * 100
        Fd_pererr[i] = (Fd_norm_p - Fd_norm) / Fd_norm_p * 100


    # absolute errors
    err = ddRo_all - Fa_all
    err_p = ddRo_all - Fa_all_p

    Fa_avg = np.nanmean(Fa_all, axis=0)
    Fl_avg = np.nanmean(Fl_all, axis=0)
    Fd_avg = np.nanmean(Fd_all, axis=0)

    Fa_std = np.nanstd(Fa_all, axis=0)
    Fl_std = np.nanstd(Fl_all, axis=0)
    Fd_std = np.nanstd(Fd_all, axis=0)

    Fa_avg_p = np.nanmean(Fa_all_p, axis=0)
    Fl_avg_p = np.nanmean(Fl_all_p, axis=0)
    Fd_avg_p = np.nanmean(Fd_all_p, axis=0)

    Fa_std_p = np.nanstd(Fa_all_p, axis=0)
    Fl_std_p = np.nanstd(Fl_all_p, axis=0)
    Fd_std_p = np.nanstd(Fd_all_p, axis=0)

    ddRo_avg = np.nanmean(ddRo_all, axis=0)
    ddRo_std = np.nanstd(ddRo_all, axis=0)

    err_avg = np.nanmean(err, axis=0)
    err_std = np.nanstd(err, axis=0)

    err_avg_p = np.nanmean(err_p, axis=0)
    err_std_p = np.nanstd(err_p, axis=0)

    boost_avg = np.nanmean(boost_all, axis=0)
    boost_std = np.nanstd(boost_all, axis=0)

    Fa_pererr_avg = np.nanmean(Fa_pererr, axis=0)
    Fl_pererr_avg = np.nanmean(Fl_pererr, axis=0)
    Fd_pererr_avg = np.nanmean(Fd_pererr, axis=0)
    Fa_pererr_std = np.nanstd(Fa_pererr, axis=0)
    Fl_pererr_std = np.nanstd(Fl_pererr, axis=0)
    Fd_pererr_std = np.nanstd(Fd_pererr, axis=0)


    Si = {}
    Si['Fa'] = Fa_all
    Si['Fl'] = Fl_all
    Si['Fd'] = Fd_all

    Si['Fa_p'] = Fa_all_p
    Si['Fl_p'] = Fl_all_p
    Si['Fd_p'] = Fd_all_p

    Si['ddRo'] = ddRo_all

    Si['err'] = err
    Si['err_p'] = err_p

    Si['boost'] = boost_all

    Si['Fa_avg'] = Fa_avg
    Si['Fl_avg'] = Fl_avg
    Si['Fd_avg'] = Fd_avg

    Si['Fa_std'] = Fa_std
    Si['Fl_std'] = Fl_std
    Si['Fd_std'] = Fd_std

    Si['Fa_avg_p'] = Fa_avg_p
    Si['Fl_avg_p'] = Fl_avg_p
    Si['Fd_avg_p'] = Fd_avg_p

    Si['Fa_std_p'] = Fa_std_p
    Si['Fl_std_p'] = Fl_std_p
    Si['Fd_std_p'] = Fd_std_p

    Si['ddRo_std'] = ddRo_std
    Si['ddRo_avg'] = ddRo_avg

    Si['err_avg'] = err_avg
    Si['err_std'] = err_std

    Si['err_avg_p'] = err_avg_p
    Si['err_std_p'] = err_std_p

    Si['boost_avg'] = boost_avg
    Si['boost_std'] = boost_std

    Si['trial_ids'] = trial_ids

    Si['Fa_pererr'] = Fa_pererr
    Si['Fl_pererr'] = Fl_pererr
    Si['Fd_pererr'] = Fd_pererr

    Si['Fa_pererr_avg'] = Fa_pererr_avg
    Si['Fl_pererr_avg'] = Fl_pererr_avg
    Si['Fd_pererr_avg'] = Fd_pererr_avg

    Si['Fa_pererr_std'] = Fa_pererr_std
    Si['Fl_pererr_std'] = Fl_pererr_std
    Si['Fd_pererr_std'] = Fd_pererr_std

    Sn[snake_id] = Si


# %% Time series of boosts to check the forces_dist code

snake_id = 95
idx = 2

assert(Sn[snake_id]['trial_ids'][idx] == 618)

boosts = Sn[snake_id]['boost'][idx]


fig, ax = plt.subplots()
ax.plot(boosts)
sns.despine()


# %% Plot 9 x 2 force and acceleration fill

#snake_ids_i = [81, 91, 95]

fig, axs = plt.subplots(9, 2, figsize=(6, 12))

sns.despine()

# X
for ax in axs[:3].flatten():
    ax.invert_xaxis()
    ax.set_ylim(-.75, .75)
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

# Y
for ax in axs[3:6].flatten():
    ax.invert_xaxis()
    ax.set_ylim(-.2, .8)
    ax.set_xlim(8.5, 0)
    ax.set_yticks([0, .75])
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

# Z
for ax in axs[6:].flatten():
    ax.invert_xaxis()
    # ax.set_ylim(-1.1, 1)
    ax.set_ylim(-.1, 2)
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    # ax.axhline(0, color='gray', linestyle='--')
    ax.axhline(1, color='gray', linestyle='--')

for i in np.arange(9):
    for j in np.arange(2):
        if i < 8:
            axs[i, j].set_xticklabels([])
        if j > 0:
            axs[i, j].set_yticklabels([])

axs[0, 0].set_title('Aerodynamic forces', fontsize='x-small')
axs[0, 1].set_title('Acceleration + gravity', fontsize='x-small')
axs[-1, 0].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 1].set_xlabel('Height (m)', fontsize='x-small')

for i in np.arange(3):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    Si = Sn[snake_id]
    # Si = S[snake_id]
    ntrials = Si['Fa'].shape[0]

    label = 'Snake {}, {:.1f}g, n={}'.format(snake_id, masses[i], ntrials)
    axs[i + 0, 0].text(8, .45, label, fontsize='x-small', color=colors_trial_id)

    # fills
    Fa_l, Fa_h = Si['Fa_avg'] - Si['Fa_std'], Si['Fa_avg'] + Si['Fa_std']
    Fa_lp, Fa_hp = Si['Fa_avg_p'] - Si['Fa_std_p'], Si['Fa_avg_p'] + Si['Fa_std_p']
    ddRo_l, ddRo_h = Si['ddRo_avg'] - Si['ddRo_std'], Si['ddRo_avg'] + Si['ddRo_std']

    Fa_l_x, Fa_l_y, Fa_l_z = Fa_l.T
    Fa_h_x, Fa_h_y, Fa_h_z = Fa_h.T
    Fa_lp_x, Fa_lp_y, Fa_lp_z = Fa_lp.T
    Fa_hp_x, Fa_hp_y, Fa_hp_z = Fa_hp.T

    ddRo_l_x, ddRo_l_y, ddRo_l_z = ddRo_l.T
    ddRo_h_x, ddRo_h_y, ddRo_h_z = ddRo_h.T

    axs[i + 0, 0].fill_between(Zi, Fa_l_x, Fa_h_x, color=colors_trial_id, alpha=.5)
    axs[i + 3, 0].fill_between(Zi, Fa_l_y, Fa_h_y, color=colors_trial_id, alpha=.5)
    axs[i + 6, 0].fill_between(Zi, Fa_l_z, Fa_h_z, color=colors_trial_id, alpha=.5)

    axs[i + 0, 0].fill_between(Zi, Fa_lp_x, Fa_hp_x, color=colors_trial_id)
    axs[i + 3, 0].fill_between(Zi, Fa_lp_y, Fa_hp_y, color=colors_trial_id)
    axs[i + 6, 0].fill_between(Zi, Fa_lp_z, Fa_hp_z, color=colors_trial_id)

    axs[i + 0, 1].fill_between(Zi, ddRo_l_x, ddRo_h_x, color=colors_trial_id)
    axs[i + 3, 1].fill_between(Zi, ddRo_l_y, ddRo_h_y, color=colors_trial_id)
    axs[i + 6, 1].fill_between(Zi, ddRo_l_z, ddRo_h_z, color=colors_trial_id)


    # averages
    axs[i + 0, 0].plot(Zi, Si['Fa_avg'][:, 0], 'gray', lw=2, alpha=.5)
    axs[i + 3, 0].plot(Zi, Si['Fa_avg'][:, 1], 'gray', lw=2, alpha=.5)
    axs[i + 6, 0].plot(Zi, Si['Fa_avg'][:, 2], 'gray', lw=2, alpha=.5)

    axs[i + 0, 0].plot(Zi, Si['Fa_avg_p'][:, 0], 'k', lw=2)
    axs[i + 3, 0].plot(Zi, Si['Fa_avg_p'][:, 1], 'k', lw=2)
    axs[i + 6, 0].plot(Zi, Si['Fa_avg_p'][:, 2], 'k', lw=2)

    axs[i + 0, 1].plot(Zi, Si['ddRo_avg'][:, 0], 'k', lw=2)
    axs[i + 3, 1].plot(Zi, Si['ddRo_avg'][:, 1], 'k', lw=2)
    axs[i + 6, 1].plot(Zi, Si['ddRo_avg'][:, 2], 'k', lw=2)

#fig.savefig(FIG.format('boosts F=ma 81_91_95 9x2'), **FIGOPT)


# %% Error time series - fill

#fig, axs = plt.subplots(3, 3, sharex=True, sharey=True,
#                        figsize=(8, 8))
fig, axs = plt.subplots(7, 3, sharex=True, sharey=True,
                        figsize=(8, 12))

sns.despine()

for ax in axs.flatten():
    ax.invert_xaxis()
    ax.set_ylim(-.75, .75)
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

axs[-1, 0].set_ylabel('Abs. error', fontsize='x-small')
axs[-1, 0].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 1].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 2].set_xlabel('Height (m)', fontsize='x-small')

axs[0, 0].set_title('Lateral', fontsize='x-small')
axs[0, 1].set_title('Fore-aft', fontsize='x-small')
axs[0, 2].set_title('Vertical', fontsize='x-small')

#for i in np.arange(3):
for i in np.arange(7):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]

    # Si = S[snake_id]
    Si = Sn[snake_id]
    ntrials = Si['Fa'].shape[0]

    errl, errh = Si['err_avg'] - Si['err_std'], Si['err_avg'] + Si['err_std']
    xl, xh = errl[:, 0], errh[:, 0]
    yl, yh = errl[:, 1], errh[:, 1]
    zl, zh = errl[:, 2], errh[:, 2]
    axs[i, 0].fill_between(Zi, xl, xh, color=colors_trial_id, alpha=.5)
    axs[i, 1].fill_between(Zi, yl, yh, color=colors_trial_id, alpha=.5)
    axs[i, 2].fill_between(Zi, zl, zh, color=colors_trial_id, alpha=.5)

    errl_p, errh_p = Si['err_avg_p'] - Si['err_std_p'], Si['err_avg_p'] + Si['err_std_p']
    xl_p, xh_p = errl_p[:, 0], errh_p[:, 0]
    yl_p, yh_p = errl_p[:, 1], errh_p[:, 1]
    zl_p, zh_p = errl_p[:, 2], errh_p[:, 2]
    axs[i, 0].fill_between(Zi, xl_p, xh_p, color=colors_trial_id)
    axs[i, 1].fill_between(Zi, yl_p, yh_p, color=colors_trial_id)
    axs[i, 2].fill_between(Zi, zl_p, zh_p, color=colors_trial_id)

    axs[i, 0].plot(Zi, Si['err_avg'][:, 0], 'gray', lw=2)
    axs[i, 1].plot(Zi, Si['err_avg'][:, 1], 'gray', lw=2)
    axs[i, 2].plot(Zi, Si['err_avg'][:, 2], 'gray', lw=2)

    axs[i, 0].plot(Zi, Si['err_avg_p'][:, 0], 'k', lw=2)
    axs[i, 1].plot(Zi, Si['err_avg_p'][:, 1], 'k', lw=2)
    axs[i, 2].plot(Zi, Si['err_avg_p'][:, 2], 'k', lw=2)

    label = 'Snake {}, {:.1f}g, n={}'.format(snake_id, masses[i], ntrials)
    axs[i, 0].text(8, .45, label, fontsize='x-small', color=colors_trial_id)

#fig.savefig(FIG.format('boosts absolute errors 7x3'), **FIGOPT)


# %% Boost time series - fill

Zhigh, Zlow = 7, 1.5
idx_mid = np.where((Zi <= Zhigh) & (Zi >= Zlow))[0]
idx_out1 = np.where(Zi >= Zhigh)[0]
idx_out2 = np.where(Zi <= Zlow)[0]

avg_boost = np.zeros((nsnakes, 3))

fig, axs = plt.subplots(7, 2, sharex=True, sharey=True,
                        figsize=(6, 12))

sns.despine()

for ax in axs.flatten():
    ax.invert_xaxis()
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
#    ax.axhline(1, color='gray', linestyle='--')

#axs[-1, 0].set_ylabel('Abs. error', fontsize='x-small')
axs[-1, 0].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 1].set_xlabel('Height (m)', fontsize='x-small')

axs[0, 0].set_title('Boost in lift', fontsize='x-small')
axs[0, 1].set_title('Boost in drag', fontsize='x-small')

for i in np.arange(7):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]

    Si = Sn[snake_id]
    ntrials = Si['Fa'].shape[0]

    avg_boost[i] = np.nanmean(Si['boost_avg'][idx_mid], axis=0)

    bl, bh = Si['boost_avg'] - Si['boost_std'], Si['boost_avg'] + Si['boost_std']
    ll, lh = bl[:, 0], bh[:, 0]
    dl, dh = bl[:, 1], bh[:, 1]

    axs[i, 0].fill_between(Zi[idx_mid], ll[idx_mid], lh[idx_mid], color=colors_trial_id)
    axs[i, 0].fill_between(Zi[idx_out1], ll[idx_out1], lh[idx_out1], color=colors_trial_id, alpha=.3)
    axs[i, 0].fill_between(Zi[idx_out2], ll[idx_out2], lh[idx_out2], color=colors_trial_id, alpha=.3)

    axs[i, 1].fill_between(Zi[idx_mid], dl[idx_mid], dh[idx_mid], color=colors_trial_id)
    axs[i, 1].fill_between(Zi[idx_out1], dl[idx_out1], dh[idx_out1], color=colors_trial_id, alpha=.3)
    axs[i, 1].fill_between(Zi[idx_out2], dl[idx_out2], dh[idx_out2], color=colors_trial_id, alpha=.3)

    axs[i, 0].plot(Zi[idx_mid], Si['boost_avg'][idx_mid, 0], 'k', lw=2)
    axs[i, 1].plot(Zi[idx_mid], Si['boost_avg'][idx_mid, 1], 'k', lw=2)

    label_l = '{:.2f}'.format(avg_boost[i, 0])
    label_d = '{:.2f}'.format(avg_boost[i, 1])
    axs[i, 0].axhline(avg_boost[i, 0], color='gray', ls='--')
    axs[i, 1].axhline(avg_boost[i, 1], color='gray', ls='--')

    axs[i, 0].text(1, avg_boost[i, 0] + .2, label_l, fontsize='x-small', color='gray')
    axs[i, 1].text(1, avg_boost[i, 1] + .2, label_d, fontsize='x-small', color='gray')

#fig.savefig(FIG.format('boosts 7x2'), **FIGOPT)


# %% Boost time series - fill --- boost_L / boost_D

Zhigh, Zlow = 7, 1.5
idx_mid = np.where((Zi <= Zhigh) & (Zi >= Zlow))[0]
idx_out1 = np.where(Zi >= Zhigh)[0]
idx_out2 = np.where(Zi <= Zlow)[0]

avg_boost = np.zeros((nsnakes, 3))

fig, axs = plt.subplots(7, 3, sharex=True, sharey=True,
                        figsize=(9, 12))

sns.despine()

for ax in axs.flatten():
    ax.invert_xaxis()
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
#    ax.axhline(1, color='gray', linestyle='--')

#axs[-1, 0].set_ylabel('Abs. error', fontsize='x-small')
axs[-1, 0].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 1].set_xlabel('Height (m)', fontsize='x-small')

axs[0, 0].set_title('Boost in lift', fontsize='x-small')
axs[0, 1].set_title('Boost in drag', fontsize='x-small')

for i in np.arange(7):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]

    Si = Sn[snake_id]
    ntrials = Si['Fa'].shape[0]

    avg_boost[i] = np.nanmean(Si['boost_avg'][idx_mid], axis=0)

    bl, bh = Si['boost_avg'] - Si['boost_std'], Si['boost_avg'] + Si['boost_std']
    ll, lh = bl[:, 0], bh[:, 0]
    dl, dh = bl[:, 1], bh[:, 1]
    ldl, ldh = bl[:, 2], bh[:, 2]

    axs[i, 0].fill_between(Zi[idx_mid], ll[idx_mid], lh[idx_mid], color=colors_trial_id)
    axs[i, 0].fill_between(Zi[idx_out1], ll[idx_out1], lh[idx_out1], color=colors_trial_id, alpha=.3)
    axs[i, 0].fill_between(Zi[idx_out2], ll[idx_out2], lh[idx_out2], color=colors_trial_id, alpha=.3)

    axs[i, 1].fill_between(Zi[idx_mid], dl[idx_mid], dh[idx_mid], color=colors_trial_id)
    axs[i, 1].fill_between(Zi[idx_out1], dl[idx_out1], dh[idx_out1], color=colors_trial_id, alpha=.3)
    axs[i, 1].fill_between(Zi[idx_out2], dl[idx_out2], dh[idx_out2], color=colors_trial_id, alpha=.3)

    axs[i, 2].fill_between(Zi[idx_mid], ldl[idx_mid], ldh[idx_mid], color=colors_trial_id)
    axs[i, 2].fill_between(Zi[idx_out1], ldl[idx_out1], ldh[idx_out1], color=colors_trial_id, alpha=.3)
    axs[i, 2].fill_between(Zi[idx_out2], ldl[idx_out2], ldh[idx_out2], color=colors_trial_id, alpha=.3)

    axs[i, 0].plot(Zi[idx_mid], Si['boost_avg'][idx_mid, 0], 'k', lw=2)
    axs[i, 1].plot(Zi[idx_mid], Si['boost_avg'][idx_mid, 1], 'k', lw=2)
    axs[i, 2].plot(Zi[idx_mid], Si['boost_avg'][idx_mid, 2], 'k', lw=2)

    label_l = '{:.2f}'.format(avg_boost[i, 0])
    label_d = '{:.2f}'.format(avg_boost[i, 1])
    label_ld = '{:.2f}'.format(avg_boost[i, 2])
    axs[i, 0].axhline(avg_boost[i, 0], color='gray', ls='--')
    axs[i, 1].axhline(avg_boost[i, 1], color='gray', ls='--')
    axs[i, 2].axhline(avg_boost[i, 2], color='gray', ls='--')

    axs[i, 0].text(1, avg_boost[i, 0] + .2, label_l, fontsize='x-small', color='gray')
    axs[i, 1].text(1, avg_boost[i, 1] + .2, label_d, fontsize='x-small', color='gray')
    axs[i, 2].text(1, avg_boost[i, 2] + .2, label_d, fontsize='x-small', color='gray')

#fig.savefig(FIG.format('boosts 7x3'), **FIGOPT)


# %% Percent error time series - fill

Zhigh, Zlow = 7, 1.5
idx_mid = np.where((Zi <= Zhigh) & (Zi >= Zlow))[0]
idx_out1 = np.where(Zi >= Zhigh)[0]
idx_out2 = np.where(Zi <= Zlow)[0]

avg_boost = np.zeros((nsnakes, 2))

fig, axs = plt.subplots(7, 2, sharex=True, sharey=True,
                        figsize=(6, 12))

sns.despine()

for ax in axs.flatten():
    ax.invert_xaxis()
    ax.set_ylim(-150, 150)
    ax.set_yticks([-100, 0, 100])
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
#    ax.axhline(1, color='gray', linestyle='--')

#axs[-1, 0].set_ylabel('Abs. error', fontsize='x-small')
axs[-1, 0].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 1].set_xlabel('Height (m)', fontsize='x-small')

axs[0, 0].set_title('Percent error in lift', fontsize='x-small')
axs[0, 1].set_title('Percent error in drag', fontsize='x-small')

for i in np.arange(7):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]

    Si = Sn[snake_id]
    ntrials = Si['Fa'].shape[0]

#    avg_boost[i] = np.nanmean(Si['boost_avg'][idx_mid], axis=0)

    ll, lh = Si['Fl_pererr_avg'] - Si['Fl_pererr_std'], Si['Fl_pererr_avg'] + Si['Fl_pererr_std']
    dl, dh = Si['Fd_pererr_avg'] - Si['Fd_pererr_std'], Si['Fd_pererr_avg'] + Si['Fd_pererr_std']

    axs[i, 0].fill_between(Zi[idx_mid], ll[idx_mid], lh[idx_mid], color=colors_trial_id)
    axs[i, 0].fill_between(Zi[idx_out1], ll[idx_out1], lh[idx_out1], color=colors_trial_id, alpha=.3)
    axs[i, 0].fill_between(Zi[idx_out2], ll[idx_out2], lh[idx_out2], color=colors_trial_id, alpha=.3)

    axs[i, 1].fill_between(Zi[idx_mid], dl[idx_mid], dh[idx_mid], color=colors_trial_id)
    axs[i, 1].fill_between(Zi[idx_out1], dl[idx_out1], dh[idx_out1], color=colors_trial_id, alpha=.3)
    axs[i, 1].fill_between(Zi[idx_out2], dl[idx_out2], dh[idx_out2], color=colors_trial_id, alpha=.3)

    axs[i, 0].plot(Zi[idx_mid], Si['Fl_pererr_avg'][idx_mid], 'k', lw=2)
    axs[i, 1].plot(Zi[idx_mid], Si['Fd_pererr_avg'][idx_mid], 'k', lw=2)

#    label_l = '{:.2f}'.format(avg_boost[i, 0])
#    label_d = '{:.2f}'.format(avg_boost[i, 1])
#    axs[i, 0].axhline(avg_boost[i, 0], color='gray', ls='--')
#    axs[i, 1].axhline(avg_boost[i, 1], color='gray', ls='--')
#
#    axs[i, 0].text(1, avg_boost[i, 0] + .2, label_l, fontsize='x-small', color='gray')
#    axs[i, 1].text(1, avg_boost[i, 1] + .2, label_d, fontsize='x-small', color='gray')

#fig.savefig(FIG.format('boosts percent error 7x2'), **FIGOPT)


# %%

i = 2  # snake index
j = 2  # trial index
#jj = 80  # height index
jj = 0

snake_id = snake_ids[i]
trial_id = Sn[snake_id]['trial_ids'][j]


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

#kw_args = dict(angles='xy', scale_units='xy', scale=1, width=.006)
kw_args = dict(angles='xy', scale_units='xy', scale=1, width=.006)
kw_args_p = dict(angles='xy', scale_units='xy', scale=1, width=.008)


Si = Sn[snake_id]
ddRo = Si['ddRo']
Fa, Fl, Fd = Si['Fa'], Si['Fl'], Si['Fd']
Fa_p, Fl_p, Fd_p = Si['Fa_p'], Si['Fl_p'], Si['Fd_p']


#for jj in np.r_[50, 75, 100, 125, 150]:
for jj in np.r_[150]:

    ax.quiver(0, 0, ddRo[j, jj, 1], ddRo[j, jj, 2], color='k', **kw_args)

    ax.quiver(0, 0, Fa[j, jj, 1], Fa[j, jj, 2], color='r', **kw_args)
    ax.quiver(0, 0, Fl[j, jj, 1], Fl[j, jj, 2], color='b', **kw_args)
    ax.quiver(0, 0, Fd[j, jj, 1], Fd[j, jj, 2], color='y', **kw_args)

    ax.quiver(0, 0, Fa_p[j, jj, 1], Fa_p[j, jj, 2], color='r', **kw_args_p)
    ax.quiver(0, 0, Fl_p[j, jj, 1], Fl_p[j, jj, 2], color='b', **kw_args_p)
    ax.quiver(0, 0, Fd_p[j, jj, 1], Fd_p[j, jj, 2], color='y', **kw_args_p)


ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box-forced')
ax.set_xlabel('Fore-aft direction')
ax.set_ylabel('Vertical direction')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-.25, 1.5)

ax.margins(.1)
sns.despine()
fig.set_tight_layout(True)


# %%

from matplotlib.animation import FuncAnimation


i = 2  # snake index
j = 2  # trial index
#jj = 80  # height index
jj = 0

snake_id = snake_ids[i]
trial_id = Sn[snake_id]['trial_ids'][j]

# %%
Si = Sn[snake_id]
ddRo = Si['ddRo']
Fa, Fl, Fd = Si['Fa'], Si['Fl'], Si['Fd']
Fa_p, Fl_p, Fd_p = Si['Fa_p'], Si['Fl_p'], Si['Fd_p']


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box-forced')
ax.set_xlabel('Fore-aft direction')
ax.set_ylabel('Vertical direction')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-.25, 1.5)

ax.margins(.1)
sns.despine()
fig.set_tight_layout(True)

kw_args = dict(angles='xy', scale_units='xy', scale=1, width=.006)
kw_args_p = dict(angles='xy', scale_units='xy', scale=1, width=.008)

QddRo = ax.quiver(0, 0, ddRo[j, jj, 1], ddRo[j, jj, 2], color='k', **kw_args)

#QFa = ax.quiver(0, 0, Fa[j, jj, 1], Fa[j, jj, 2], color='gray', **kw_args)
#QFl = ax.quiver(0, 0, Fl[j, jj, 1], Fl[j, jj, 2], color='gray', **kw_args)
#QFd = ax.quiver(0, 0, Fd[j, jj, 1], Fd[j, jj, 2], color='gray', **kw_args)
#
#QFa_p = ax.quiver(0, 0, Fa_p[j, jj, 1], Fa_p[j, jj, 2], color='r', **kw_args)
#QFl_p = ax.quiver(0, 0, Fl_p[j, jj, 1], Fl_p[j, jj, 2], color='r', **kw_args)
#QFd_p = ax.quiver(0, 0, Fd_p[j, jj, 1], Fd_p[j, jj, 2], color='r', **kw_args)

QFa = ax.quiver(0, 0, Fa[j, jj, 1], Fa[j, jj, 2], color='r', **kw_args)
QFl = ax.quiver(0, 0, Fl[j, jj, 1], Fl[j, jj, 2], color='b', **kw_args)
QFd = ax.quiver(0, 0, Fd[j, jj, 1], Fd[j, jj, 2], color='y', **kw_args)

QFa_p = ax.quiver(0, 0, Fa_p[j, jj, 1], Fa_p[j, jj, 2], color='r', **kw_args_p)
QFl_p = ax.quiver(0, 0, Fl_p[j, jj, 1], Fl_p[j, jj, 2], color='b', **kw_args_p)
QFd_p = ax.quiver(0, 0, Fd_p[j, jj, 1], Fd_p[j, jj, 2], color='y', **kw_args_p)

#ax.plot(Fl_p[j, :, 1], Fl_p[j, :, 2], c='b', lw=1)
#ax.plot(Fd_p[j, :, 1], Fd_p[j, :, 2], c='y', lw=1)


def update_quiver(jj):  # QddRo, QFa, QFl, QFd, QFa_p, QFl_p, QFd_p):
    QddRo.set_UVC(np.r_[ddRo[j, jj, 1]], np.r_[ddRo[j, jj, 2]])

    QFa.set_UVC(np.r_[Fa[j, jj, 1]], np.r_[Fa[j, jj, 2]])
    QFl.set_UVC(np.r_[Fl[j, jj, 1]], np.r_[Fl[j, jj, 2]])
    QFd.set_UVC(np.r_[Fd[j, jj, 1]], np.r_[Fd[j, jj, 2]])

    QFa_p.set_UVC(np.r_[Fa_p[j, jj, 1]], np.r_[Fa_p[j, jj, 2]])
    QFl_p.set_UVC(np.r_[Fl_p[j, jj, 1]], np.r_[Fl_p[j, jj, 2]])
    QFd_p.set_UVC(np.r_[Fd_p[j, jj, 1]], np.r_[Fd_p[j, jj, 2]])

    return QddRo, QFa, QFl, QFd, QFa_p, QFl_p, QFd_p


slowed = 10
dt = 1 / 179.
ani = FuncAnimation(fig, update_quiver, frames=nheight,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=True)#, init_func=init)


save_movie = False
if save_movie:
    #ani.save('../Movies/s_serp3d/5X aerial serpnoid curve.mp4',
#         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])

    movie_name = '../Movies/s_forces/{0}_{1} 10x quiver.mp4'
    movie_name = movie_name.format(trial_id, snake_id)
#    ani.save(movie_name,
#             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])
    ani.save(movie_name,
             codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])


#anim = animation.FuncAnimation(fig, update_quiver, # fargs=(Q, X, Y),
#                               interval=10, blit=False)


# %%