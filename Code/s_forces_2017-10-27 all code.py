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

S, I = {}, {}
for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    ntrials = len(fn_names)
    colors_trial_id = colors[row]

    Fa_S_all = np.zeros((ntrials, len(Zi), 3))
    Fl_S_all = np.zeros((ntrials, len(Zi), 3))
    Fd_S_all = np.zeros((ntrials, len(Zi), 3))
    ddRo_S_all = np.zeros((ntrials, len(Zi), 3))

    Fa_I_all = np.zeros((ntrials, len(Zi), 3))
    Fl_I_all = np.zeros((ntrials, len(Zi), 3))
    Fd_I_all = np.zeros((ntrials, len(Zi), 3))
    ddRo_I_all = np.zeros((ntrials, len(Zi), 3))

    gamma_all = np.zeros((ntrials, len(Zi)))


    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times'][start:stop]
        Z = d['Ro_S'][start:stop, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * g

        masses[row] = float(d['mass'])

        # forces and accelerations in STRAIGHTENED frame
        Fa_S = d['Fa_S'][start:stop]
        Fl_S = d['Fl_S'][start:stop]
        Fd_S = d['Fd_S'][start:stop]
        assert(np.allclose((Fl_S * Fd_S).sum(axis=2), 0))
        ddRo_S = d['ddRo_S'][start:stop] / 1000
        Fa_S_non = Fa_S.sum(axis=1) / mg
        Fl_S_non = Fl_S.sum(axis=1) / mg
        Fd_S_non = Fd_S.sum(axis=1) / mg
#        Fa_S_non[:, 2] -= 1  # subract off gravity (LHS is forces)
#        Fl_S_non[:, 2] -= .5  # subract off gravity (LHS is forces)
#        Fd_S_non[:, 2] -= .5  # subract off gravity (LHS is forces)
        ddRo_S_non = ddRo_S / g  # RHS is just acceleration
        ddRo_S_non[:, 2] += 1  # move gravity onto acceleration

        gamma = np.rad2deg(d['gamma'])[start:stop]

        # forces and accelerations in INERTIAL frame
        Fa_I = d['Fa_I'][start:stop]
        Fl_I = d['Fl_I'][start:stop]
        Fd_I = d['Fd_I'][start:stop]
        assert(np.allclose((Fl_I * Fd_I).sum(axis=2), 0))
        ddRo_I = d['ddRo_I'][start:stop] / 1000
        Fa_I_non = Fa_S.sum(axis=1) / mg
        Fd_I_non = Fd_S.sum(axis=1) / mg
        Fl_I_non = Fl_S.sum(axis=1) / mg
#        Fa_I_non[:, 2] -= 1  # subract off gravity (LHS is forces)
#        Fl_I_non[:, 2] -= .5  # subract off gravity (LHS is forces)
#        Fd_I_non[:, 2] -= .5  # subract off gravity (LHS is forces)
        ddRo_I_non = ddRo_S / g  # RHS is just acceleration
        ddRo_I_non[:, 2] += 1  # move gravity onto acceleration

#        break

        # interpolate
        gamma_all[i] = interp1d(Z, gamma, bounds_error=False, fill_value=np.nan)(Zi)
        for k in np.arange(3):
            # S interpolation
            Fa_S_all[i, :, k] = interp1d(Z, Fa_S_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fl_S_all[i, :, k] = interp1d(Z, Fl_S_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fd_S_all[i, :, k] = interp1d(Z, Fd_S_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            ddRo_S_all[i, :, k] = interp1d(Z, ddRo_S_non[:, k], bounds_error=False,
                                           fill_value=np.nan)(Zi)

            # I interpolation
            Fa_I_all[i, :, k] = interp1d(Z, Fa_I_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fl_I_all[i, :, k] = interp1d(Z, Fl_I_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fd_I_all[i, :, k] = interp1d(Z, Fd_I_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            ddRo_I_all[i, :, k] = interp1d(Z, ddRo_I_non[:, k], bounds_error=False,
                                           fill_value=np.nan)(Zi)

    # absolute errors
    S_err = ddRo_S_all - Fa_S_all
    I_err = ddRo_I_all - Fa_I_all

    Fa_S_avg = np.nanmean(Fa_S_all, axis=0)
    Fl_S_avg = np.nanmean(Fl_S_all, axis=0)
    Fd_S_avg = np.nanmean(Fd_S_all, axis=0)
    ddRo_S_avg = np.nanmean(ddRo_S_all, axis=0)
    Fa_S_std = np.nanstd(Fa_S_all, axis=0)
    Fl_S_std = np.nanstd(Fl_S_all, axis=0)
    Fd_S_std = np.nanstd(Fd_S_all, axis=0)
    ddRo_S_std = np.nanstd(ddRo_S_all, axis=0)
    S_err_avg = np.nanmean(S_err, axis=0)
    S_err_std = np.nanstd(S_err, axis=0)

    Fa_I_avg = np.nanmean(Fa_I_all, axis=0)
    Fl_I_avg = np.nanmean(Fl_I_all, axis=0)
    Fd_I_avg = np.nanmean(Fd_I_all, axis=0)
    ddRo_I_avg = np.nanmean(ddRo_I_all, axis=0)
    Fa_I_std = np.nanstd(Fa_I_all, axis=0)
    Fl_I_std = np.nanstd(Fl_I_all, axis=0)
    Fd_I_std = np.nanstd(Fd_I_all, axis=0)
    ddRo_I_std = np.nanstd(ddRo_I_all, axis=0)
    I_err_avg = np.nanmean(I_err, axis=0)
    I_err_std = np.nanstd(I_err, axis=0)

    gamma_avg = np.nanmean(gamma_all, axis=0)
    gamma_std = np.nanstd(gamma_all, axis=0)

    Si = {}
    Si['Fa'] = Fa_S_all
    Si['Fl'] = Fl_S_all
    Si['Fd'] = Fd_S_all
    Si['ddRo'] = ddRo_S_all
    Si['Fa_avg'] = Fa_S_avg
    Si['Fl_avg'] = Fl_S_avg
    Si['Fd_avg'] = Fd_S_avg
    Si['ddRo_avg'] = ddRo_S_avg
    Si['Fa_std'] = Fa_S_std
    Si['Fl_std'] = Fl_S_std
    Si['Fd_std'] = Fd_S_std
    Si['ddRo_std'] = ddRo_S_std
    Si['err'] = S_err
    Si['err_avg'] = S_err_avg
    Si['err_std'] = S_err_std
    Si['gamma'] = gamma_all
    Si['gamma_avg'] = gamma_avg
    Si['gamma_std'] = gamma_std

    Ii = {}
    Ii['Fa'] = Fa_I_all
    Ii['Fl'] = Fl_I_all
    Ii['Fd'] = Fd_I_all
    Ii['ddRo'] = ddRo_I_all
    Ii['Fa_avg'] = Fa_I_avg
    Ii['Fl_avg'] = Fl_I_avg
    Ii['Fd_avg'] = Fd_I_avg
    Ii['ddRo_avg'] = ddRo_I_avg
    Ii['Fa_std'] = Fa_I_std
    Ii['Fl_std'] = Fl_I_std
    Ii['Fd_std'] = Fd_I_std
    Ii['ddRo_std'] = ddRo_I_std
    Ii['err'] = I_err
    Ii['err_avg'] = I_err_avg
    Ii['err_std'] = I_err_std
    Ii['gamma'] = gamma_all
    Ii['gamma_avg'] = gamma_avg
    Ii['gamma_std'] = gamma_std

    S[snake_id] = Si
    I[snake_id] = Ii


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
    Si = S[snake_id]
    ntrials = Si['Fa'].shape[0]

    label = 'Snake {}, {:.1f}g, n={}'.format(snake_id, masses[i], ntrials)
    axs[i + 0, 0].text(8, .45, label, fontsize='x-small', color=colors_trial_id)

    # fills
    Fa_l, Fa_h = Si['Fa_avg'] - Si['Fa_std'], Si['Fa_avg'] + Si['Fa_std']
    ddRo_l, ddRo_h = Si['ddRo_avg'] - Si['ddRo_std'], Si['ddRo_avg'] + Si['ddRo_std']
    Fa_l_x, Fa_l_y, Fa_l_z = Fa_l.T
    Fa_h_x, Fa_h_y, Fa_h_z = Fa_h.T
    ddRo_l_x, ddRo_l_y, ddRo_l_z = ddRo_l.T
    ddRo_h_x, ddRo_h_y, ddRo_h_z = ddRo_h.T

    axs[i + 0, 0].fill_between(Zi, Fa_l_x, Fa_h_x, color=colors_trial_id)
    axs[i + 3, 0].fill_between(Zi, Fa_l_y, Fa_h_y, color=colors_trial_id)
    axs[i + 6, 0].fill_between(Zi, Fa_l_z, Fa_h_z, color=colors_trial_id)

    axs[i + 0, 1].fill_between(Zi, ddRo_l_x, ddRo_h_x, color=colors_trial_id)
    axs[i + 3, 1].fill_between(Zi, ddRo_l_y, ddRo_h_y, color=colors_trial_id)
    axs[i + 6, 1].fill_between(Zi, ddRo_l_z, ddRo_h_z, color=colors_trial_id)


    # averages
    axs[i + 0, 0].plot(Zi, Si['Fa_avg'][:, 0], 'k', lw=2)
    axs[i + 3, 0].plot(Zi, Si['Fa_avg'][:, 1], 'k', lw=2)
    axs[i + 6, 0].plot(Zi, Si['Fa_avg'][:, 2], 'k', lw=2)

    axs[i + 0, 1].plot(Zi, Si['ddRo_avg'][:, 0], 'k', lw=2)
    axs[i + 3, 1].plot(Zi, Si['ddRo_avg'][:, 1], 'k', lw=2)
    axs[i + 6, 1].plot(Zi, Si['ddRo_avg'][:, 2], 'k', lw=2)

#fig.savefig(FIG.format('F=ma 81_91_95 9x2'), **FIGOPT)


# %% Glide angle vs. height

delta_all = np.zeros(nsnakes)
delta_poly = np.zeros((nsnakes, 2))

Zhigh, Zlow = 7, 1.5
idx_mid = np.where((Zi <= Zhigh) & (Zi >= Zlow))[0]
#idx_out = np.where((Zi > Zhigh) | (Zi < Zlow))[0]
idx_out1 = np.where(Zi >= Zhigh)[0]
idx_out2 = np.where(Zi <= Zlow)[0]

fig, axs = plt.subplots(nsnakes, 1, sharex=True, sharey=True,
                        figsize=(4, 12))
ax = axs[0]
ax.invert_xaxis()
ax.set_xlim(8.5, 0)
ax.set_xticks(np.r_[0:9:2])
ax.set_ylim(0, 90)
ax.set_yticks([0, 45, 90])
ax.yaxis.set_major_formatter(degree_formatter)
ax = axs[-1]
ax.set_xlabel('Height (m)', fontsize='x-small')
ax.set_ylabel('Glide angle', fontsize='x-small')
sns.despine()

for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    ax = axs[i]
    Si = S[snake_id]
    gamma, gamma_avg, gamma_std = Si['gamma'], Si['gamma_avg'], Si['gamma_std']
    ntrials, ntime = gamma.shape[0], gamma.shape[1]

#    for j in np.arange(ntrials):
#        ax.plot(Zi, gamma[j], c='gray', lw=.5)

    dl, dh = gamma_avg - gamma_std, gamma_avg + gamma_std

    ax.fill_between(Zi[idx_mid], dl[idx_mid], dh[idx_mid],
                    color=colors_trial_id)
    ax.fill_between(Zi[idx_out1], dl[idx_out1], dh[idx_out1],
                    color=colors_trial_id, alpha=.3)
    ax.fill_between(Zi[idx_out2], dl[idx_out2], dh[idx_out2],
                    color=colors_trial_id, alpha=.3)
    ax.plot(Zi[idx_mid], gamma_avg[idx_mid], c='k')


#fig.savefig(FIG.format('glide angle 7x1'), **FIGOPT)


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

    Si = S[snake_id]
    ntrials = Si['Fa'].shape[0]

    errl, errh = Si['err_avg'] - Si['err_std'], Si['err_avg'] + Si['err_std']
    xl, xh = errl[:, 0], errh[:, 0]
    yl, yh = errl[:, 1], errh[:, 1]
    zl, zh = errl[:, 2], errh[:, 2]
    axs[i, 0].fill_between(Zi, xl, xh, color=colors_trial_id)
    axs[i, 1].fill_between(Zi, yl, yh, color=colors_trial_id)
    axs[i, 2].fill_between(Zi, zl, zh, color=colors_trial_id)

    axs[i, 0].plot(Zi, Si['err_avg'][:, 0], 'k', lw=2)
    axs[i, 1].plot(Zi, Si['err_avg'][:, 1], 'k', lw=2)
    axs[i, 2].plot(Zi, Si['err_avg'][:, 2], 'k', lw=2)

    label = 'Snake {}, {:.1f}g, n={}'.format(snake_id, masses[i], ntrials)
    axs[i, 0].text(8, .45, label, fontsize='x-small', color=colors_trial_id)

#fig.savefig(FIG.format('absolute errors 7x3'), **FIGOPT)


# %% Error as an angle - fills

delta_all = np.zeros(nsnakes)
delta_poly = np.zeros((nsnakes, 2))

Zhigh, Zlow = 7, 1.5
idx_mid = np.where((Zi <= Zhigh) & (Zi >= Zlow))[0]
#idx_out = np.where((Zi > Zhigh) | (Zi < Zlow))[0]
idx_out1 = np.where(Zi >= Zhigh)[0]
idx_out2 = np.where(Zi <= Zlow)[0]

fig, axs = plt.subplots(nsnakes, 1, sharex=True, sharey=True,
                        figsize=(4, 12))
ax = axs[0]
ax.invert_xaxis()
ax.set_xlim(8.5, 0)
ax.set_xticks(np.r_[0:9:2])
ax.set_ylim(0, 45)
ax.set_yticks([0, 15, 30, 45])
ax.yaxis.set_major_formatter(degree_formatter)
ax = axs[-1]
ax.set_xlabel('Height (m)', fontsize='x-small')
ax.set_ylabel('Force rotation', fontsize='x-small')
sns.despine()

for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    ax = axs[i]
    Si = S[snake_id]
    ddRo, Fa = Si['ddRo'], Si['Fa']
    ntrials, ntime = ddRo.shape[0], ddRo.shape[1]

    #TODO norm taking care of nans!
    ddRo_n = np.zeros((ntrials, ntime))
    Fa_n = np.zeros((ntrials, ntime))
    for j in np.arange(ntrials):
        ddRo_n[j] = np.linalg.norm(ddRo[j], axis=1)
        Fa_n[j] = np.linalg.norm(Fa[j], axis=1)
    # ddRo_n, Fa_n = np.linalg.norm(ddRo, axis=2), np.linalg.norm(Fa, axis=2)

    # delta = np.rad2deg(np.arctan2(Si['err'][:, :, 2], Si['err'][:, :, 1]))
#    delta = np.arccos(np.dot(ddRo, Fa) / (ddRo_n * Fa_n))
    delta = np.arccos(np.nansum(ddRo * Fa, axis=2) / (ddRo_n * Fa_n))
    delta = np.rad2deg(delta)

    ratio = ddRo_n / Fa_n

    delta_avg = np.nanmean(delta, axis=0)
    delta_std = np.nanstd(delta, axis=0)
    delta_shift = delta_avg[idx_mid].mean()

    # store the shifts
    delta_all[i] = delta_shift

    # linear fit
    delta_poly[i] = np.polyfit(Zi[idx_mid], delta_avg[idx_mid], 1)
    delta_fit = np.polyval(delta_poly[i], Zi[idx_mid])

    dl, dh = delta_avg - delta_std, delta_avg + delta_std

    ax.fill_between(Zi[idx_mid], dl[idx_mid], dh[idx_mid],
                    color=colors_trial_id)
    ax.fill_between(Zi[idx_out1], dl[idx_out1], dh[idx_out1],
                    color=colors_trial_id, alpha=.3)
    ax.fill_between(Zi[idx_out2], dl[idx_out2], dh[idx_out2],
                    color=colors_trial_id, alpha=.3)
    ax.plot(Zi[idx_mid], delta_avg[idx_mid], c='k')

    ax.axhline(delta_shift, ls='--', c='gray')
    label = '{0:.1f}'.format(delta_shift) + u'\u00B0'
    ax.text(1.5, delta_shift + 2.5, label, fontsize='x-small', color='gray')

    ax.plot(Zi[idx_mid], delta_fit, '--', c='gray')

#    break


#fig.savefig(FIG.format('angles errors 7x1'), **FIGOPT)


# %% Error as an angle and magnitude --- fills

delta_all = np.zeros(nsnakes)
delta_poly = np.zeros((nsnakes, 2))
ratio_all = np.zeros(nsnakes)

Zhigh, Zlow = 7, 1.5
idx_mid = np.where((Zi <= Zhigh) & (Zi >= Zlow))[0]
#idx_out = np.where((Zi > Zhigh) | (Zi < Zlow))[0]
idx_out1 = np.where(Zi >= Zhigh)[0]
idx_out2 = np.where(Zi <= Zlow)[0]

fig, axs = plt.subplots(nsnakes, 2, sharex=True, sharey=False,
                        figsize=(6, 12))

for ax in axs[:, 0]:
    ax.invert_xaxis()
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.set_ylim(0, 45)
    ax.set_yticks([0, 15, 30, 45])
    ax.yaxis.set_major_formatter(degree_formatter)

for ax in axs[:, 1]:
    ax.invert_xaxis()
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.set_ylim(0, 1.25)

ax = axs[-1, 0]
axs[-1, 0].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 1].set_xlabel('Height (m)', fontsize='x-small')
axs[0, 0].set_title('Fa rotation', fontsize='x-small')
axs[0, 1].set_title('|ddRo| / |Fa|', fontsize='x-small')
sns.despine()

for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    Si = S[snake_id]
    ddRo, Fa = Si['ddRo'], Si['Fa']
    ntrials, ntime = ddRo.shape[0], ddRo.shape[1]

    #TODO norm taking care of nans!
    ddRo_n = np.zeros((ntrials, ntime))
    Fa_n = np.zeros((ntrials, ntime))
    for j in np.arange(ntrials):
        ddRo_n[j] = np.linalg.norm(ddRo[j], axis=1)
        Fa_n[j] = np.linalg.norm(Fa[j], axis=1)
    # ddRo_n, Fa_n = np.linalg.norm(ddRo, axis=2), np.linalg.norm(Fa, axis=2)

    # delta = np.rad2deg(np.arctan2(Si['err'][:, :, 2], Si['err'][:, :, 1]))
#    delta = np.arccos(np.dot(ddRo, Fa) / (ddRo_n * Fa_n))
    delta = np.arccos(np.nansum(ddRo * Fa, axis=2) / (ddRo_n * Fa_n))
    delta = np.rad2deg(delta)

    delta_avg = np.nanmean(delta, axis=0)
    delta_std = np.nanstd(delta, axis=0)
    delta_shift = delta_avg[idx_mid].mean()

    # store the shifts
    delta_all[i] = delta_shift

    # linear fit
    delta_poly[i] = np.polyfit(Zi[idx_mid], delta_avg[idx_mid], 1)
    delta_fit = np.polyval(delta_poly[i], Zi)

    # ratios
    ratio = ddRo_n / Fa_n
    ratio_avg = np.nanmean(ratio, axis=0)
    ratio_std = np.nanstd(ratio, axis=0)
    ratio_shift = ratio_avg[idx_mid].mean()

    ratio_all[i] = ratio_shift

    S[snake_id]['delta_avg'] = delta_avg


    ax = axs[i, 0]
    dl, dh = delta_avg - delta_std, delta_avg + delta_std
    ax.fill_between(Zi[idx_mid], dl[idx_mid], dh[idx_mid],
                    color=colors_trial_id)
    ax.fill_between(Zi[idx_out1], dl[idx_out1], dh[idx_out1],
                    color=colors_trial_id, alpha=.3)
    ax.fill_between(Zi[idx_out2], dl[idx_out2], dh[idx_out2],
                    color=colors_trial_id, alpha=.3)
    ax.plot(Zi[idx_mid], delta_avg[idx_mid], c='k')

#    ax.axhline(delta_shift, ls='--', c='gray')
#    label = '{0:.1f}'.format(delta_shift) + u'\u00B0'
#    ax.text(1, delta_shift + 2.5, label, fontsize='x-small', color='gray')

    ax.plot(Zi, delta_fit, '--', c='gray')


    ax = axs[i, 1]
    rl, rh = ratio_avg - ratio_std, ratio_avg + ratio_std
    ax.fill_between(Zi[idx_mid], rl[idx_mid], rh[idx_mid],
                    color=colors_trial_id)
    ax.fill_between(Zi[idx_out1], rl[idx_out1], rh[idx_out1],
                    color=colors_trial_id, alpha=.3)
    ax.fill_between(Zi[idx_out2], rl[idx_out2], rh[idx_out2],
                    color=colors_trial_id, alpha=.3)
    ax.plot(Zi[idx_mid], ratio_avg[idx_mid], c='k')

    ax.axhline(ratio_shift, ls='--', c='gray')
    label = '{0:.2f}'.format(ratio_shift)
    ax.text(1, ratio_shift + .1, label, fontsize='x-small', color='gray')

#fig.savefig(FIG.format('absolute errors rotation and magnitude'), **FIGOPT)


# %% Iterate through each trial, rotate by the average shift, calculate errors
# rotate new lift and drag vectors
#
# constant values

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


for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    Si = S[snake_id]
    de = -np.deg2rad(delta_all[i])
    Fa_S, Fl_S, Fd_S = Si['Fa'], Si['Fl'], Si['Fd']
    ddRo_S = Si['ddRo']
    ntrials = Fl_S.shape[0]

    R = np.array([[1, 0, 0],
                  [0, np.cos(de), -np.sin(de)],
                  [0, np.sin(de),  np.cos(de)]])

    Fa_S_prime = np.zeros_like(Fa_S)
    Fl_S_prime = np.zeros_like(Fl_S)
    Fd_S_prime = np.zeros_like(Fd_S)

    for j in np.arange(ntrials):
        Fa_S_prime[j] = ratio_all[i] * np.dot(R, Fa_S[j].T).T
        Fl_S_prime[j] = ratio_all[i] * np.dot(R, Fl_S[j].T).T
        Fd_S_prime[j] = ratio_all[i] * np.dot(R, Fd_S[j].T).T

    err_prime = ddRo_S - Fa_S_prime
#    err_prime = ddRo_S - (Fl_S_prime + Fd_S_prime)
    err_p_avg = np.nanmean(err_prime, axis=0)
    err_p_std = np.nanstd(err_prime, axis=0)


    errl, errh = err_p_avg - err_p_std, err_p_avg + err_p_std
    xl, xh = errl[:, 0], errh[:, 0]
    yl, yh = errl[:, 1], errh[:, 1]
    zl, zh = errl[:, 2], errh[:, 2]
    axs[i, 0].fill_between(Zi, xl, xh, color=colors_trial_id)
    axs[i, 1].fill_between(Zi, yl, yh, color=colors_trial_id)
    axs[i, 2].fill_between(Zi, zl, zh, color=colors_trial_id)

    axs[i, 0].plot(Zi, err_p_avg[:, 0], 'k', lw=2)
    axs[i, 1].plot(Zi, err_p_avg[:, 1], 'k', lw=2)
    axs[i, 2].plot(Zi, err_p_avg[:, 2], 'k', lw=2)

    label = 'Snake {}, {:.1f}g'.format(snake_id, masses[i])
    axs[i, 0].text(8, .45, label, fontsize='x-small', color=colors_trial_id)

#fig.savefig(FIG.format('absolute error Fa rot and mag'), **FIGOPT)


# %% Iterate through each trial, rotate by the average shift, calculate errors
# rotate new lift and drag vectors
#
# polyfit values

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


for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    Si = S[snake_id]
    # de = -np.deg2rad(delta_all[i])
    Fa_S, Fl_S, Fd_S = Si['Fa'], Si['Fl'], Si['Fd']
    ddRo_S, ddRo_S_avg = Si['ddRo'], Si['ddRo_avg']
    ntrials, nheight = Fl_S.shape[0], Fl_S.shape[1]

    Rs = np.zeros((nheight, 3, 3))
    for j in np.arange(nheight):
#        de = -np.deg2rad(np.polyval(delta_poly[i], Zi[j]))
        de = -np.deg2rad(Si['delta_avg'][j])
        Rs[j] = np.array([[1, 0, 0],
                          [0, np.cos(de), -np.sin(de)],
                          [0, np.sin(de),  np.cos(de)]])

#    Fa_S_p = np.zeros_like(Fa_S)
#    Fl_S_p = np.zeros_like(Fl_S)
#    Fd_S_p = np.zeros_like(Fd_S)
#    Fl_S_pa = np.zeros_like(Fl_S)  # from the accelerations
#    Fd_S_pa = np.zeros_like(Fd_S)  # from the accelerations
    Fa_S_p = Fa_S.copy()
    Fl_S_p = Fl_S.copy()
    Fd_S_p = Fd_S.copy()
    Fl_S_pa = Fl_S.copy()  # from the accelerations
    Fd_S_pa = Fd_S.copy()  # from the accelerations
    ratio_Fl_S_p = np.zeros((ntrials, nheight))
    # ratio_Fl_S_p_rel = np.zeros((ntrials, nheight))
    ratio_Fd_S_p = np.zeros((ntrials, nheight))
    ratio_Fl_S_pa = np.zeros((ntrials, nheight))
    ratio_Fd_S_pa = np.zeros((ntrials, nheight))

    for j in np.arange(ntrials):
        for jj in np.arange(nheight):
            # from the average rotation best fit line
            Fa_S_p[j, jj] = ratio_all[i] * np.dot(Rs[jj], Fa_S[j, jj].T).T

            nL = Fl_S[j, jj, 1:] / norm(Fl_S[j, jj, 1:])
            nD = Fd_S[j, jj, 1:] / norm(Fd_S[j, jj, 1:])

            Fl_S_p[j, jj, 1:] = np.dot(Fa_S_p[j, jj, 1:], nL) * nL
            Fd_S_p[j, jj, 1:] = np.dot(Fa_S_p[j, jj, 1:], nD) * nD
            Fl_S_pa[j, jj, 1:] = np.dot(ddRo_S[j, jj, 1:], nL) * nL  # to wiggly
            Fd_S_pa[j, jj, 1:] = np.dot(ddRo_S[j, jj, 1:], nD) * nD
#            Fl_S_pa[j, jj, 1:] = np.dot(ddRo_S_avg[jj, 1:], nL) * nL
#            Fd_S_pa[j, jj, 1:] = np.dot(ddRo_S_avg[jj, 1:], nD) * nD

            # how to modify the lift and drag forces
            ratio_Fl_S_p[j, jj] = norm(Fl_S_p[j, jj]) / norm(Fl_S[j, jj])
            ratio_Fd_S_p[j, jj] = norm(Fd_S_p[j, jj]) / norm(Fd_S[j, jj])
            ratio_Fl_S_pa[j, jj] = norm(Fl_S_pa[j, jj]) / norm(Fl_S[j, jj])
            ratio_Fd_S_pa[j, jj] = norm(Fd_S_pa[j, jj]) / norm(Fd_S[j, jj])

#            ratio_Fl_S_p[j, jj] = (norm(Fl_S_p[j, jj]) - norm(Fl_S[j, jj])) / norm(Fl_S[j, jj])
#            ratio_Fd_S_p[j, jj] = (norm(Fd_S_p[j, jj]) - norm(Fd_S[j, jj])) / norm(Fd_S[j, jj])
#            ratio_Fl_S_pa[j, jj] = (norm(Fl_S_pa[j, jj]) - norm(Fl_S[j, jj])) / norm(Fl_S[j, jj])
#            ratio_Fd_S_pa[j, jj] = (norm(Fd_S_pa[j, jj]) - norm(Fd_S[j, jj])) / norm(Fd_S[j, jj])


    Fa_S_pld = Fl_S_p + Fd_S_p
    Fa_S_pald = Fl_S_pa + Fd_S_pa

    err_p = ddRo_S - Fa_S_p
    err_pld = ddRo_S - Fa_S_pld
    err_pald = ddRo_S - Fa_S_pald

    err_p = err_p
    err_p = err_pald
    err_p_avg = np.nanmean(err_p, axis=0)
    err_p_std = np.nanstd(err_p, axis=0)

    # now store these values
    S[snake_id]['Fa_p'] = Fa_S_p
    S[snake_id]['Fl_p'] = Fl_S_p
    S[snake_id]['Fd_p'] = Fd_S_p
    S[snake_id]['Fa_pld'] = Fa_S_pld
    S[snake_id]['Fl_pa'] = Fl_S_pa
    S[snake_id]['Fd_pa'] = Fd_S_pa
    S[snake_id]['Fa_pald'] = Fa_S_pald

    S[snake_id]['ratio_Fl_p'] = ratio_Fl_S_p
    S[snake_id]['ratio_Fd_p'] = ratio_Fd_S_p
    S[snake_id]['ratio_Fl_pa'] = ratio_Fl_S_pa
    S[snake_id]['ratio_Fd_pa'] = ratio_Fd_S_pa

    S[snake_id]['ratio_Fl_p_avg'] = np.nanmean(ratio_Fl_S_p, axis=0)
    S[snake_id]['ratio_Fd_p_avg'] = np.nanmean(ratio_Fd_S_p, axis=0)
    S[snake_id]['ratio_Fl_p_std'] = np.nanstd(ratio_Fl_S_p, axis=0)
    S[snake_id]['ratio_Fd_p_std'] = np.nanstd(ratio_Fd_S_p, axis=0)

    S[snake_id]['ratio_Fl_pa_avg'] = np.nanmean(ratio_Fl_S_pa, axis=0)
    S[snake_id]['ratio_Fd_pa_avg'] = np.nanmean(ratio_Fd_S_pa, axis=0)
    S[snake_id]['ratio_Fl_pa_std'] = np.nanstd(ratio_Fl_S_pa, axis=0)
    S[snake_id]['ratio_Fd_pa_std'] = np.nanstd(ratio_Fd_S_pa, axis=0)

    S[snake_id]['err_p'] = err_p
    S[snake_id]['err_pld'] = err_pld
    S[snake_id]['err_pald'] = err_pald


    errl, errh = err_p_avg - err_p_std, err_p_avg + err_p_std
    xl, xh = errl[:, 0], errh[:, 0]
    yl, yh = errl[:, 1], errh[:, 1]
    zl, zh = errl[:, 2], errh[:, 2]
    axs[i, 0].fill_between(Zi, xl, xh, color=colors_trial_id)
    axs[i, 1].fill_between(Zi, yl, yh, color=colors_trial_id)
    axs[i, 2].fill_between(Zi, zl, zh, color=colors_trial_id)

    axs[i, 0].plot(Zi, err_p_avg[:, 0], 'k', lw=2)
    axs[i, 1].plot(Zi, err_p_avg[:, 1], 'k', lw=2)
    axs[i, 2].plot(Zi, err_p_avg[:, 2], 'k', lw=2)

    label = 'Snake {}, {:.1f}g'.format(snake_id, masses[i])
    axs[i, 0].text(8, .45, label, fontsize='x-small', color=colors_trial_id)

#fig.savefig(FIG.format('absolute errors 7x3 first correction'), **FIGOPT)


# %% Plot of the Fd and Fl magnitude ratios

ratio_Fl, ratio_Fd = np.zeros(nsnakes), np.zeros(nsnakes)
ratio_Fl_S_p_poly = np.zeros((nsnakes, 2))

fig, axs = plt.subplots(nsnakes, 2, sharex=True, sharey=True,
                        figsize=(6, 12))

for ax in axs[:, 0]:
    ax.invert_xaxis()
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    # ax.axhline(1, color='gray', lw=1)
    ax.set_yticks([0, 1, 2, 3])
#    ax.set_ylim(0, 45)
#    ax.set_yticks([0, 15, 30, 45])
#    ax.yaxis.set_major_formatter(degree_formatter)

for ax in axs[:, 1]:
    ax.invert_xaxis()
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.set_ylim(0, 3)
    # ax.axhline(1, color='gray', lw=1)
    ax.set_yticks([0, 1, 2, 3])

axs[-1, 0].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 1].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 0].set_ylabel('Boost', fontsize='x-small')
axs[-1, 0].set_ylabel('Boost', fontsize='x-small')
axs[0, 0].set_title('|Fl new| / |Fl old|', fontsize='x-small')
axs[0, 1].set_title('|Fd new| / |Fd old|', fontsize='x-small')
sns.despine()

for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    Si = S[snake_id]
    ddRo, Fa = Si['ddRo'], Si['Fa']
    ntrials, ntime = ddRo.shape[0], ddRo.shape[1]

    ratio_Fl_S_p = Si['ratio_Fl_p'].copy()
    ratio_Fd_S_p = Si['ratio_Fd_p'].copy()
    ratio_Fl_S_p_avg = Si['ratio_Fl_p_avg'].copy()
    ratio_Fd_S_p_avg = Si['ratio_Fd_p_avg'].copy()
    ratio_Fl_S_p_std = Si['ratio_Fl_p_std'].copy()
    ratio_Fd_S_p_std = Si['ratio_Fd_p_std'].copy()

#    ratio_Fl_S_p = Si['ratio_Fl_pa'].copy()
#    ratio_Fd_S_p = Si['ratio_Fd_pa'].copy()
#    ratio_Fl_S_p_avg = Si['ratio_Fl_pa_avg'].copy()
#    ratio_Fd_S_p_avg = Si['ratio_Fd_pa_avg'].copy()
#    ratio_Fl_S_p_std = Si['ratio_Fl_pa_std'].copy()
#    ratio_Fd_S_p_std = Si['ratio_Fd_pa_std'].copy()

#    ratio_Fl_S_p_avg = np.nanmean(ratio_Fl_S_p, axis=0)
#    ratio_Fd_S_p_avg = np.nanmean(ratio_Fd_S_p, axis=0)
#    ratio_Fl_S_p_std = np.nanstd(ratio_Fl_S_p, axis=0)
#    ratio_Fd_S_p_std = np.nanstd(ratio_Fd_S_p, axis=0)

    ratio_Fl_S_p_mean = ratio_Fl_S_p_avg[idx_mid].mean()
    ratio_Fd_S_p_mean = ratio_Fd_S_p_avg[idx_mid].mean()
    ratio_Fl[i] = ratio_Fl_S_p_mean
    ratio_Fd[i] = ratio_Fd_S_p_mean

    # linear fit
    ratio_Fl_S_p_poly[i] = np.polyfit(Zi[idx_mid], ratio_Fl_S_p_avg[idx_mid], 1)
    ratio_Fl_S_p_fit = np.polyval(ratio_Fl_S_p_poly[i], Zi)

    ax = axs[i, 0]
#    for j in np.arange(ntrials):
#        ax.plot(Zi, ratio_Fl_S_p[j], c='gray')
    rl = ratio_Fl_S_p_avg - ratio_Fl_S_p_std
    rh = ratio_Fl_S_p_avg + ratio_Fl_S_p_std
    ax.fill_between(Zi[idx_mid], rl[idx_mid], rh[idx_mid],
                    color=colors_trial_id)
    ax.fill_between(Zi[idx_out1], rl[idx_out1], rh[idx_out1],
                    color=colors_trial_id, alpha=.3)
    ax.fill_between(Zi[idx_out2], rl[idx_out2], rh[idx_out2],
                    color=colors_trial_id, alpha=.3)
    ax.plot(Zi[idx_mid], ratio_Fl_S_p_avg[idx_mid], c='k')

    ax.plot(Zi, ratio_Fl_S_p_fit, '--', c='gray')

    ax.axhline(ratio_Fl_S_p_mean, ls='--', c='gray')
    label = '{0:.1f}'.format(ratio_Fl_S_p_mean)
    ax.text(1, ratio_Fl_S_p_mean + .1, label, fontsize='x-small', color='gray')


    ax = axs[i, 1]
#    for j in np.arange(ntrials):
#        ax.plot(Zi, ratio_Fd_S_p[j], c='gray')
    rl = ratio_Fd_S_p_avg - ratio_Fd_S_p_std
    rh = ratio_Fd_S_p_avg + ratio_Fd_S_p_std
    ax.fill_between(Zi[idx_mid], rl[idx_mid], rh[idx_mid],
                    color=colors_trial_id)
    ax.fill_between(Zi[idx_out1], rl[idx_out1], rh[idx_out1],
                    color=colors_trial_id, alpha=.3)
    ax.fill_between(Zi[idx_out2], rl[idx_out2], rh[idx_out2],
                    color=colors_trial_id, alpha=.3)
    ax.plot(Zi[idx_mid], ratio_Fd_S_p_avg[idx_mid], c='k')

    ax.axhline(ratio_Fd_S_p_mean, ls='--', c='gray')
    label = '{0:.2f}'.format(ratio_Fd_S_p_mean)
    ax.text(1, ratio_Fd_S_p_mean + .1, label, fontsize='x-small', color='gray')

#fig.savefig(FIG.format('Boost Fl and Fd'), **FIGOPT)


# %% Now change Fl and Fd ALONG THE BODY according to this analysis

#ratio_Fl.mean()
#Out[853]: 1.7595542892152243

#ratio_Fd.mean()
#Out[854]: 0.63044384480184323

#ratio_Fd[:4].mean()
#Out[855]: 0.66062444850474633

#ratio_Fl[:4].mean()
#Out[856]: 1.6375356232302378

boost_Fl = 1.5
boost_Fd = .63


Sn, In = {}, {}
for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    ntrials = len(fn_names)
    colors_trial_id = colors[row]

    Fa_S_all = np.zeros((ntrials, len(Zi), 3))
    Fl_S_all = np.zeros((ntrials, len(Zi), 3))
    Fd_S_all = np.zeros((ntrials, len(Zi), 3))
    ddRo_S_all = np.zeros((ntrials, len(Zi), 3))

    Fa_I_all = np.zeros((ntrials, len(Zi), 3))
    Fl_I_all = np.zeros((ntrials, len(Zi), 3))
    Fd_I_all = np.zeros((ntrials, len(Zi), 3))
    ddRo_I_all = np.zeros((ntrials, len(Zi), 3))

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times'][start:stop]
        Z = d['Ro_S'][start:stop, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * g

        masses[row] = float(d['mass'])

        # forces and accelerations in STRAIGHTENED frame
#        Fa_S = d['Fa_S'][start:stop]
        Fl_S = boost_Fl * d['Fl_S'][start:stop]
        Fd_S = boost_Fd * d['Fd_S'][start:stop]
        Fa_S = Fl_S + Fd_S
        assert(np.allclose((Fl_S * Fd_S).sum(axis=2), 0))
        ddRo_S = d['ddRo_S'][start:stop] / 1000
        Fa_S_non = Fa_S.sum(axis=1) / mg
        Fl_S_non = Fl_S.sum(axis=1) / mg
        Fd_S_non = Fd_S.sum(axis=1) / mg
        ddRo_S_non = ddRo_S / g  # RHS is just acceleration
        ddRo_S_non[:, 2] += 1  # move gravity onto acceleration

        # forces and accelerations in INERTIAL frame
#        Fa_I = d['Fa_I'][start:stop]
        Fl_I = boost_Fl * d['Fl_I'][start:stop]
        Fd_I = boost_Fd * d['Fd_I'][start:stop]
        Fa_I = Fl_I + Fd_I
        assert(np.allclose((Fl_I * Fd_I).sum(axis=2), 0))
        ddRo_I = d['ddRo_I'][start:stop] / 1000
        Fa_I_non = Fa_S.sum(axis=1) / mg
        Fd_I_non = Fd_S.sum(axis=1) / mg
        Fl_I_non = Fl_S.sum(axis=1) / mg
        ddRo_I_non = ddRo_S / g  # RHS is just acceleration
        ddRo_I_non[:, 2] += 1  # move gravity onto acceleration

#        break

        # interpolate
        for k in np.arange(3):
            # S interpolation
            Fa_S_all[i, :, k] = interp1d(Z, Fa_S_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fl_S_all[i, :, k] = interp1d(Z, Fl_S_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fd_S_all[i, :, k] = interp1d(Z, Fd_S_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            ddRo_S_all[i, :, k] = interp1d(Z, ddRo_S_non[:, k], bounds_error=False,
                                           fill_value=np.nan)(Zi)

            # I interpolation
            Fa_I_all[i, :, k] = interp1d(Z, Fa_I_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fl_I_all[i, :, k] = interp1d(Z, Fl_I_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fd_I_all[i, :, k] = interp1d(Z, Fd_I_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            ddRo_I_all[i, :, k] = interp1d(Z, ddRo_I_non[:, k], bounds_error=False,
                                           fill_value=np.nan)(Zi)

    # absolute errors
    S_err = ddRo_S_all - Fa_S_all
    I_err = ddRo_I_all - Fa_I_all

    Fa_S_avg = np.nanmean(Fa_S_all, axis=0)
    Fl_S_avg = np.nanmean(Fl_S_all, axis=0)
    Fd_S_avg = np.nanmean(Fd_S_all, axis=0)
    ddRo_S_avg = np.nanmean(ddRo_S_all, axis=0)
    Fa_S_std = np.nanstd(Fa_S_all, axis=0)
    Fl_S_std = np.nanstd(Fl_S_all, axis=0)
    Fd_S_std = np.nanstd(Fd_S_all, axis=0)
    ddRo_S_std = np.nanstd(ddRo_S_all, axis=0)
    S_err_avg = np.nanmean(S_err, axis=0)
    S_err_std = np.nanstd(S_err, axis=0)

    Fa_I_avg = np.nanmean(Fa_I_all, axis=0)
    Fl_I_avg = np.nanmean(Fl_I_all, axis=0)
    Fd_I_avg = np.nanmean(Fd_I_all, axis=0)
    ddRo_I_avg = np.nanmean(ddRo_I_all, axis=0)
    Fa_I_std = np.nanstd(Fa_I_all, axis=0)
    Fl_I_std = np.nanstd(Fl_I_all, axis=0)
    Fd_I_std = np.nanstd(Fd_I_all, axis=0)
    ddRo_I_std = np.nanstd(ddRo_I_all, axis=0)
    I_err_avg = np.nanmean(I_err, axis=0)
    I_err_std = np.nanstd(I_err, axis=0)

    Si = {}
    Si['Fa'] = Fa_S_all
    Si['Fl'] = Fl_S_all
    Si['Fd'] = Fd_S_all
    Si['ddRo'] = ddRo_S_all
    Si['Fa_avg'] = Fa_S_avg
    Si['Fl_avg'] = Fl_S_avg
    Si['Fd_avg'] = Fd_S_avg
    Si['ddRo_avg'] = ddRo_S_avg
    Si['Fa_std'] = Fa_S_std
    Si['Fl_std'] = Fl_S_std
    Si['Fd_std'] = Fd_S_std
    Si['ddRo_std'] = ddRo_S_std
    Si['err'] = S_err
    Si['err_avg'] = S_err_avg
    Si['err_std'] = S_err_std

    Ii = {}
    Ii['Fa'] = Fa_I_all
    Ii['Fl'] = Fl_I_all
    Ii['Fd'] = Fd_I_all
    Ii['ddRo'] = ddRo_I_all
    Ii['Fa_avg'] = Fa_I_avg
    Ii['Fl_avg'] = Fl_I_avg
    Ii['Fd_avg'] = Fd_I_avg
    Ii['ddRo_avg'] = ddRo_I_avg
    Ii['Fa_std'] = Fa_I_std
    Ii['Fl_std'] = Fl_I_std
    Ii['Fd_std'] = Fd_I_std
    Ii['ddRo_std'] = ddRo_I_std
    Ii['err'] = I_err
    Ii['err_avg'] = I_err_avg
    Ii['err_std'] = I_err_std

    Sn[snake_id] = Si
    In[snake_id] = Ii


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
    ddRo_l, ddRo_h = Si['ddRo_avg'] - Si['ddRo_std'], Si['ddRo_avg'] + Si['ddRo_std']
    Fa_l_x, Fa_l_y, Fa_l_z = Fa_l.T
    Fa_h_x, Fa_h_y, Fa_h_z = Fa_h.T
    ddRo_l_x, ddRo_l_y, ddRo_l_z = ddRo_l.T
    ddRo_h_x, ddRo_h_y, ddRo_h_z = ddRo_h.T

    axs[i + 0, 0].fill_between(Zi, Fa_l_x, Fa_h_x, color=colors_trial_id)
    axs[i + 3, 0].fill_between(Zi, Fa_l_y, Fa_h_y, color=colors_trial_id)
    axs[i + 6, 0].fill_between(Zi, Fa_l_z, Fa_h_z, color=colors_trial_id)

    axs[i + 0, 1].fill_between(Zi, ddRo_l_x, ddRo_h_x, color=colors_trial_id)
    axs[i + 3, 1].fill_between(Zi, ddRo_l_y, ddRo_h_y, color=colors_trial_id)
    axs[i + 6, 1].fill_between(Zi, ddRo_l_z, ddRo_h_z, color=colors_trial_id)


    # averages
    axs[i + 0, 0].plot(Zi, Si['Fa_avg'][:, 0], 'k', lw=2)
    axs[i + 3, 0].plot(Zi, Si['Fa_avg'][:, 1], 'k', lw=2)
    axs[i + 6, 0].plot(Zi, Si['Fa_avg'][:, 2], 'k', lw=2)

    axs[i + 0, 1].plot(Zi, Si['ddRo_avg'][:, 0], 'k', lw=2)
    axs[i + 3, 1].plot(Zi, Si['ddRo_avg'][:, 1], 'k', lw=2)
    axs[i + 6, 1].plot(Zi, Si['ddRo_avg'][:, 2], 'k', lw=2)

#fig.savefig(FIG.format('F=ma 81_91_95 9x2'), **FIGOPT)


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
    axs[i, 0].fill_between(Zi, xl, xh, color=colors_trial_id)
    axs[i, 1].fill_between(Zi, yl, yh, color=colors_trial_id)
    axs[i, 2].fill_between(Zi, zl, zh, color=colors_trial_id)

    axs[i, 0].plot(Zi, Si['err_avg'][:, 0], 'k', lw=2)
    axs[i, 1].plot(Zi, Si['err_avg'][:, 1], 'k', lw=2)
    axs[i, 2].plot(Zi, Si['err_avg'][:, 2], 'k', lw=2)

    label = 'Snake {}, {:.1f}g, n={}'.format(snake_id, masses[i], ntrials)
    axs[i, 0].text(8, .45, label, fontsize='x-small', color=colors_trial_id)

#fig.savefig(FIG.format('absolute errors 7x3'), **FIGOPT)


# %% Error as an angle and magnitude --- fills

delta_all = np.zeros(nsnakes)
delta_poly = np.zeros((nsnakes, 2))
ratio_all = np.zeros(nsnakes)

Zhigh, Zlow = 7, 1.5
idx_mid = np.where((Zi <= Zhigh) & (Zi >= Zlow))[0]
#idx_out = np.where((Zi > Zhigh) | (Zi < Zlow))[0]
idx_out1 = np.where(Zi >= Zhigh)[0]
idx_out2 = np.where(Zi <= Zlow)[0]

fig, axs = plt.subplots(nsnakes, 2, sharex=True, sharey=False,
                        figsize=(6, 12))

for ax in axs[:, 0]:
    ax.invert_xaxis()
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.set_ylim(0, 45)
    ax.set_yticks([0, 15, 30, 45])
    ax.yaxis.set_major_formatter(degree_formatter)

for ax in axs[:, 1]:
    ax.invert_xaxis()
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.set_ylim(0, 1.25)

ax = axs[-1, 0]
axs[-1, 0].set_xlabel('Height (m)', fontsize='x-small')
axs[-1, 1].set_xlabel('Height (m)', fontsize='x-small')
axs[0, 0].set_title('Fa rotation', fontsize='x-small')
axs[0, 1].set_title('|ddRo| / |Fa|', fontsize='x-small')
sns.despine()

for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    # Si = S[snake_id]
    Si = Sn[snake_id]
    ddRo, Fa = Si['ddRo'], Si['Fa']
    ntrials, ntime = ddRo.shape[0], ddRo.shape[1]

    #TODO norm taking care of nans!
    ddRo_n = np.zeros((ntrials, ntime))
    Fa_n = np.zeros((ntrials, ntime))
    for j in np.arange(ntrials):
        ddRo_n[j] = np.linalg.norm(ddRo[j], axis=1)
        Fa_n[j] = np.linalg.norm(Fa[j], axis=1)
    # ddRo_n, Fa_n = np.linalg.norm(ddRo, axis=2), np.linalg.norm(Fa, axis=2)

    # delta = np.rad2deg(np.arctan2(Si['err'][:, :, 2], Si['err'][:, :, 1]))
#    delta = np.arccos(np.dot(ddRo, Fa) / (ddRo_n * Fa_n))
    delta = np.arccos(np.nansum(ddRo * Fa, axis=2) / (ddRo_n * Fa_n))
    delta = np.rad2deg(delta)

    delta_avg = np.nanmean(delta, axis=0)
    delta_std = np.nanstd(delta, axis=0)
    delta_shift = delta_avg[idx_mid].mean()

    # store the shifts
    delta_all[i] = delta_shift

    # linear fit
    delta_poly[i] = np.polyfit(Zi[idx_mid], delta_avg[idx_mid], 1)
    delta_fit = np.polyval(delta_poly[i], Zi)

    # ratios
    ratio = ddRo_n / Fa_n
    ratio_avg = np.nanmean(ratio, axis=0)
    ratio_std = np.nanstd(ratio, axis=0)
    ratio_shift = ratio_avg[idx_mid].mean()

    ratio_all[i] = ratio_shift

    S[snake_id]['delta_avg'] = delta_avg


    ax = axs[i, 0]
    dl, dh = delta_avg - delta_std, delta_avg + delta_std
    ax.fill_between(Zi[idx_mid], dl[idx_mid], dh[idx_mid],
                    color=colors_trial_id)
    ax.fill_between(Zi[idx_out1], dl[idx_out1], dh[idx_out1],
                    color=colors_trial_id, alpha=.3)
    ax.fill_between(Zi[idx_out2], dl[idx_out2], dh[idx_out2],
                    color=colors_trial_id, alpha=.3)
    ax.plot(Zi[idx_mid], delta_avg[idx_mid], c='k')

#    ax.axhline(delta_shift, ls='--', c='gray')
#    label = '{0:.1f}'.format(delta_shift) + u'\u00B0'
#    ax.text(1, delta_shift + 2.5, label, fontsize='x-small', color='gray')

    ax.plot(Zi, delta_fit, '--', c='gray')


    ax = axs[i, 1]
    rl, rh = ratio_avg - ratio_std, ratio_avg + ratio_std
    ax.fill_between(Zi[idx_mid], rl[idx_mid], rh[idx_mid],
                    color=colors_trial_id)
    ax.fill_between(Zi[idx_out1], rl[idx_out1], rh[idx_out1],
                    color=colors_trial_id, alpha=.3)
    ax.fill_between(Zi[idx_out2], rl[idx_out2], rh[idx_out2],
                    color=colors_trial_id, alpha=.3)
    ax.plot(Zi[idx_mid], ratio_avg[idx_mid], c='k')

    ax.axhline(ratio_shift, ls='--', c='gray')
    label = '{0:.2f}'.format(ratio_shift)
    ax.text(1, ratio_shift + .1, label, fontsize='x-small', color='gray')

#fig.savefig(FIG.format('absolute errors rotation and magnitude'), **FIGOPT)


# %% How the lift and drag components change

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
ax1.invert_xaxis()
ax1.set_xlim(8.5, 0)
ax1.set_xticks(np.r_[0:9:2])

ax1.plot(Zi, Fl_S[:, :, 1].T, color='gray')
ax1.plot(Zi, Fl_S_p[:, :, 1].T, color='r')
ax1.plot(Zi, Fl_S_pa[:, :, 1].T, color='g')

ax3.plot(Zi, Fl_S[:, :, 2].T, color='gray')
ax3.plot(Zi, Fl_S_p[:, :, 2].T, color='r')
ax3.plot(Zi, Fl_S_pa[:, :, 2].T, color='g')

ax2.plot(Zi, Fd_S[:, :, 1].T, color='gray')
ax2.plot(Zi, Fd_S_p[:, :, 1].T, color='r')
ax2.plot(Zi, Fd_S_pa[:, :, 1].T, color='g')

ax4.plot(Zi, Fd_S[:, :, 2].T, color='gray')
ax4.plot(Zi, Fd_S_p[:, :, 2].T, color='r')
ax4.plot(Zi, Fd_S_pa[:, :, 2].T, color='g')

#ax1.plot(Zi, Fl_S_mag.T)
#ax2.plot(Zi, Fl_S_prime_mag.T - Fl_S_mag.T)

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
sns.despine()


# %% Quiver plot for how the net force needs to change

i = 2  # snake index
j = 0  # trial index
jj = 130  # height index

snake_id = snake_ids[i]
colors_trial_id = colors[i]
Si = S[snake_id]

ddRo_S = Si['ddRo']
Fa_S, Fl_S, Fd_S = Si['Fa'], Si['Fl'], Si['Fd']
Fa_S_pld, Fl_S_p, Fd_S_p = Si['Fa_pld'], Si['Fl_p'], Si['Fd_p']
Fa_S_pald, Fl_S_pa, Fd_S_pa = Si['Fa_pald'], Si['Fl_pa'], Si['Fd_pa']


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

kw_args = dict(angles='xy', scale_units='xy', scale=1, width=.006)

# without corrections
ax.quiver(0, 0, Fa_S[j, jj, 1], Fa_S[j, jj, 2], color='gray', **kw_args)
ax.quiver(0, 0, Fl_S[j, jj, 1], Fl_S[j, jj, 2], color='gray', **kw_args)
ax.quiver(0, 0, Fd_S[j, jj, 1], Fd_S[j, jj, 2], color='gray', **kw_args)

# with Fa corrections
ax.quiver(0, 0, Fa_S_pld[j, jj, 1], Fa_S_pld[j, jj, 2], color='b', **kw_args)
ax.quiver(0, 0, Fl_S_p[j, jj, 1], Fl_S_p[j, jj, 2], color='b', **kw_args)
ax.quiver(0, 0, Fd_S_p[j, jj, 1], Fd_S_p[j, jj, 2], color='b', **kw_args)

# with ddRo corrections
ax.quiver(0, 0, Fa_S_pald[j, jj, 1], Fa_S_pald[j, jj, 2], color='r', **kw_args)
ax.quiver(0, 0, Fl_S_pa[j, jj, 1], Fl_S_pa[j, jj, 2], color='r', **kw_args)
ax.quiver(0, 0, Fd_S_pa[j, jj, 1], Fd_S_pa[j, jj, 2], color='r', **kw_args)

# acceleration vector
ax.quiver(0, 0, ddRo_S[j, jj, 1], ddRo_S[j, jj, 2], color='k', **kw_args)

ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box-forced')
ax.set_xlabel('Fore-aft direction')
ax.set_ylabel('Vertical direction')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-.25, 1.5)

ax.margins(.1)
sns.despine()
fig.set_tight_layout(True)


# %% Now change Fl and Fd ALONG THE BODY according to this analysis

Sn, In = {}, {}
for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    ntrials = len(fn_names)
    colors_trial_id = colors[row]

    ddRo_S_all = np.zeros((ntrials, len(Zi), 3))
    Fa_S_all = np.zeros((ntrials, len(Zi), 3))
    Fl_S_all = np.zeros((ntrials, len(Zi), 3))
    Fd_S_all = np.zeros((ntrials, len(Zi), 3))
    Fa_S_all_p = np.zeros((ntrials, len(Zi), 3))
    Fl_S_all_p = np.zeros((ntrials, len(Zi), 3))
    Fd_S_all_p = np.zeros((ntrials, len(Zi), 3))


    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times'][start:stop]
        Z = d['Ro_S'][start:stop, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * g

        masses[row] = float(d['mass'])

        # forces and accelerations in STRAIGHTENED frame
        ddRo_S = d['ddRo_S'][start:stop] / 1000
        Fl_S = d['Fl_S'][start:stop]
        Fd_S = d['Fd_S'][start:stop]
        Fa_S = Fl_S + Fd_S
        Fa_S_non = Fa_S.sum(axis=1) / mg
        Fl_S_non = Fl_S.sum(axis=1) / mg
        Fd_S_non = Fd_S.sum(axis=1) / mg
        ddRo_S_non = ddRo_S / g
        ddRo_S_non[:, 2] += 1  # move gravity onto acceleration

        # interpolate
        for k in np.arange(3):
            # S interpolation
            Fa_S_all[i, :, k] = interp1d(Z, Fa_S_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fl_S_all[i, :, k] = interp1d(Z, Fl_S_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            Fd_S_all[i, :, k] = interp1d(Z, Fd_S_non[:, k], bounds_error=False,
                                         fill_value=np.nan)(Zi)
            ddRo_S_all[i, :, k] = interp1d(Z, ddRo_S_non[:, k], bounds_error=False,
                                           fill_value=np.nan)(Zi)

        # shift lift and drag forces
        nD = (Fd_S_all[i].T / norm(Fd_S_all[i], axis=1)).T
        nL = (Fl_S_all[i].T / norm(Fl_S_all[i], axis=1)).T

#        Fd_S_all_p[i] = ((ddRo_S_all[i] * nD).sum(axis=1) * nD.T).T
#        Fl_S_all_p[i] = ddRo_S_all[i] - Fd_S_all_p[i]
        Fl_S_all_p[i] = ((ddRo_S_all[i] * nL).sum(axis=1) * nL.T).T
        Fd_S_all_p[i] = ddRo_S_all[i] - Fl_S_all_p[i]
        Fa_S_all_p[i] = Fl_S_all_p[i] + Fd_S_all_p[i]

    break


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
    boost_all = np.zeros((ntrials, nheight, 2))
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

        boost = np.zeros((ntime, 2))
        for ii in np.arange(ntime):
            args = (Fl_non[ii], Fd_non[ii], ddRo_non[ii])
            res = minimize(to_min, boosts0, args=(args,))
            boost[ii] = res.x

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

        for k in np.arange(2):
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


# %% Minimze ||ddRo - Fl + Fd|| by changing the angle of attack THE SAME
# for each time point (single aoa bias)

from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b

import m_aerodynamics as aerodynamics
aero_interp = aerodynamics.extend_wind_tunnel_data(plot=False)


def dot(a, b):
    """Dot product for two (n x 3) arrays. This is a dot product
    for two vectors.

    Example from the aerodynamics function:
    a1 = np.array([np.dot(dR[i], Tv[i]) for i in np.arange(nbody)])
    a2 = np.diag(np.inner(dR, Tv))
    a3 = np.sum(dR * Tv, axis=1)
    np.allclose(a1, a2)
    np.allclose(a1, a3)
    """
    return np.sum(a * b, axis=1)


def calc_forces(aoa_shift, args):

    ddRo, mg, Tv, Cv, Bv, dR, ds, c, rho, aero_interp = args

    nbody = dR.shape[0]

    # we need consistent units -- meters
    mm2m = .001  # conversion from mm to m (length unit of c, ds, dRi)
    c = mm2m * c.copy()
    ds = mm2m * ds.copy()
    dR = mm2m * dR.copy()

    # velocity components parallel and perpendicular to arifoil
    dR_T = (dot(dR, Tv) * Tv.T).T  # dR_T = dot(dR, Tv) * Tv
    dR_BC = dR - dR_T  # velocity in B-C plan

    U_BC = np.linalg.norm(dR_BC, axis=1)  # reduced velocity in BC plane
    U_tot = np.linalg.norm(dR, axis=1)  # total velocity hitting mass (for Re calc)

    # angle of velocity in BC coordinate system
    cos_c = dot(dR_BC, Cv) / U_BC
    cos_b = dot(dR_BC, Bv) / U_BC

    # arccos is constrainted to [-1, 1] (due to numerical error)
    rad_c = np.arccos(np.clip(cos_c, -1, 1))
    rad_b = np.arccos(np.clip(cos_b, -1, 1))
    deg_c = np.rad2deg(rad_c)
    deg_b = np.rad2deg(rad_b)

    # unit vectors for drag and lift directions
    Dh = (-dR_BC.T / U_BC).T  # -dR_BC / U_BC
    Lh = np.cross(Tv, Dh)  # np.cross(Ti, Dh)
    aoa = np.zeros(nbody)

    # chat in -xhat, bhat = chat x that, bhat in +zhat
    Q1 = (deg_c < 90) & (deg_b >= 90)  # lower right
    Q2 = (deg_c < 90) & (deg_b < 90)  # upper right
    Q3 = (deg_c >= 90) & (deg_b < 90)  # upper left
    Q4 = (deg_c >= 90) & (deg_b >= 90)  # lower left
    # Q1 = (cos_c > 0) & (cos_b <= 0)  # lower right
    # Q2 = (cos_c > 0) & (cos_b > 0)  # upper right
    # Q3 = (cos_c <= 0) & (cos_b > 0)  # upper left
    # Q4 = (cos_c <= 0) & (cos_b <= 0)  # lower left

    # get sign and value of aoa and sign of Lh vector correct
    aoa = np.zeros(nbody)
    aoa[Q1] = rad_c[Q1]
    aoa[Q2] = -rad_c[Q2]
    aoa[Q3] = rad_c[Q3] - np.pi
    aoa[Q4] = np.pi - rad_c[Q4]
    Lh[Q1] = -Lh[Q1]
    Lh[Q2] = -Lh[Q2]

    # NOW MODIFY THE AOA
    aoa = aoa + aoa_shift

    # dynamic pressure
    dynP = .5 * rho * U_BC**2
    dA = ds * c  # area of each segment

    # now calculate the forces
    cl, cd, clcd, Re = aero_interp(aoa, U_tot, c)
    Fl = (dynP * dA * cl * Lh.T).T  # Fl = dynP * cl * Lh
    Fd = (dynP * dA * cd * Dh.T).T  # Fd = dynP * cd * Dh
    Fa = Fl + Fd  # total aerodynamic force

    return Fl, Fd, Fa, aoa


def to_min_aoa(aoa_shift, args):
    ddRo, mg, Tv, Cv, Bv, dR, ds, c, rho, aero_interp = args

    Fl, Fd, Fa, aoa = calc_forces(aoa_shift, args)

    # from before
    Fa_b = Fa.sum(axis=0) / mg
    error = np.linalg.norm(ddRo - Fa_b)
    return error


Sa = {}
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
    boost_all = np.zeros((ntrials, nheight, 2))


    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
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

        ntime, nbody = Fa_non.shape[0], Fa_non.shape[1]

        aoa_shift0 = np.deg2rad(-10)
#        bounds = [(-np.deg2rad(35), 0)]
        bounds = [(np.deg2rad(-45), np.deg2rad(10))]

        rho = 1.17  # kg/m^3
        Tv = d['Tdir_S'][start:stop]
        Cv = d['Cdir_S'][start:stop]
        Bv = d['Bdir_S'][start:stop]
        dR = d['dR_S'][start:stop]
        ds, c = d['spl_ds'][start:stop], d['chord_spl'][start:stop]
#        ddRo_to_min = d['ddRo_S'][start:stop]

        aoa_old = d['aoa']

        aoa_shifts = np.zeros(ntime)

        Fl_new = np.zeros_like(Fl_non)
        Fd_new = np.zeros_like(Fd_non)
        Fa_new = np.zeros_like(Fa_non)
        aoa_new = np.zeros((Fl.shape[0], Fl.shape[1]))
        for ii in np.arange(ntime):
            args = (ddRo_non[ii], mg, Tv[ii], Cv[ii], Bv[ii], dR[ii], ds[ii],
                    c[ii], rho, aero_interp)

            if ii > 0:
                aoa_shift0 = aoa_shifts[ii - 1]

            res = minimize(to_min_aoa, aoa_shift0, args=(args,), bounds=bounds)
            aoa_shifts[ii] = res.x

            Fl, Fd, Fa, aoa = calc_forces(aoa_shifts[ii], args)
            Fl_new[ii] = Fl
            Fd_new[ii] = Fd
            Fa_new[ii] = Fa
            aoa_new[ii] = aoa

            print(ii)

        Fa_non = Fa_non.sum(axis=1)
        Fl_non = Fl_non.sum(axis=1)
        Fd_non = Fd_non.sum(axis=1)

        Fa_new = Fa_new.sum(axis=1) / mg
        Fl_new = Fl_new.sum(axis=1) / mg
        Fd_new = Fd_new.sum(axis=1) / mg

        Fa_non -= iiudfh


# %%

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)

F = [Fa_non, Fa_new]

for jj in np.arange(2):
    Fa = F[jj]
    for k in np.arange(3):
        axs[k, jj].plot(ddRo_non[:, k] - Fa[:, k])
        axs[k, jj].grid(True)
sns.despine()


# %%

fig, ax = plt.subplots()
#for ii in np.arange(ntime)[::50]:
for ii in [100]:
#ax.plot(np.rad2deg(aoa_shifts))
    ax.plot(np.rad2deg(aoa_new[ii]))
    ax.plot(np.rad2deg(aoa_old[ii]))
sns.despine()


# %% Boosts

boost_Fl = np.linalg.norm(Fl_new, axis=1) / np.linalg.norm(Fl_non, axis=1)
boost_Fd = np.linalg.norm(Fd_new, axis=1) / np.linalg.norm(Fd_non, axis=1)

fig, ax = plt.subplots()
ax.plot(boost_Fl)
ax.plot(boost_Fd)
sns.despine()


# %%

fig, ax = plt.subplots()
ax.plot(np.rad2deg(aoa_shifts))
sns.despine()


# %%
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

        for k in np.arange(2):
            boost_all[i, :, k] = interp1d(Z, boost[:, k], bounds_error=False,
                                           fill_value=np.nan)(Zi)

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

    Sa[snake_id] = Si


# %%

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

avg_boost = np.zeros((nsnakes, 2))

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


def update_quiver(num, Q, X, Y):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """

    U = np.cos(X + num*0.1)
    V = np.sin(Y + num*0.1)

    Q.set_UVC(U,V)

    return Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                               interval=10, blit=False)

plt.show()





# %%

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

kw_args = dict(angles='xy', scale_units='xy', scale=1, width=.006)


Si = Sn[snake_id]
ddRo = Si['ddRo']
Fa, Fl, Fd = Si['Fa'], Si['Fl'], Si['Fd']
Fa_p, Fl_p, Fd_p = Si['Fa_p'], Si['Fl_p'], Si['Fd_p']


#for jj in np.r_[50, 75, 100, 125, 150]:
for jj in np.r_[100]:

    ax.quiver(0, 0, ddRo[j, jj, 0], ddRo[j, jj, 1], color='k', **kw_args)

    ax.quiver(0, 0, Fa[j, jj, 0], Fa[j, jj, 1], color='gray', **kw_args)
    ax.quiver(0, 0, Fl[j, jj, 0], Fl[j, jj, 1], color='gray', **kw_args)
    ax.quiver(0, 0, Fd[j, jj, 0], Fd[j, jj, 1], color='gray', **kw_args)

    ax.quiver(0, 0, Fa_p[j, jj, 0], Fa_p[j, jj, 1], color='m', **kw_args)
    ax.quiver(0, 0, Fl_p[j, jj, 0], Fl_p[j, jj, 1], color='r', **kw_args)
    ax.quiver(0, 0, Fd_p[j, jj, 0], Fd_p[j, jj, 1], color='r', **kw_args)


ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box-forced')
ax.set_xlabel('Fore-aft direction')
ax.set_ylabel('Vertical direction')

ax.set_ylim(-.5, .5)
ax.set_ylim(-1.5, 1.5)

ax.margins(.1)
sns.despine()
fig.set_tight_layout(True)

# %%


# %%


# %%

j = 0
jj = 50

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

kw_args = dict(angles='xy', scale_units='xy', scale=1, width=.006)


#Si = S[snake_id]
#ddRo_S = Si['ddRo']
#Fa_S, Fl_S, Fd_S = Si['Fa'], Si['Fl'], Si['Fd']

ax.quiver(0, 0, Fa_S_all[j, jj, 1], Fa_S_all[j, jj, 2], color='gray', **kw_args)
ax.quiver(0, 0, Fl_S_all[j, jj, 1], Fl_S_all[j, jj, 2], color='gray', **kw_args)
ax.quiver(0, 0, Fd_S_all[j, jj, 1], Fd_S_all[j, jj, 2], color='gray', **kw_args)
ax.quiver(0, 0, ddRo_S_all[j, jj, 1], ddRo_S_all[j, jj, 2], color='k', **kw_args)

ax.quiver(0, 0, Fa_S_all_p[j, jj, 1], Fa_S_all_p[j, jj, 2], color='r', **kw_args)
ax.quiver(0, 0, Fl_S_all_p[j, jj, 1], Fl_S_all_p[j, jj, 2], color='r', **kw_args)
ax.quiver(0, 0, Fd_S_all_p[j, jj, 1], Fd_S_all_p[j, jj, 2], color='r', **kw_args)
#ax.quiver(0, 0, ddRo_S_all_p[j, jj, 1], ddRo_S_all_p[j, jj, 2], color='m', **kw_args)

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

    # absolute errors
    S_err = ddRo_S_all - Fa_S_all
#    I_err = ddRo_I_all - Fa_I_all

    Fa_S_avg = np.nanmean(Fa_S_all, axis=0)
    Fl_S_avg = np.nanmean(Fl_S_all, axis=0)
    Fd_S_avg = np.nanmean(Fd_S_all, axis=0)
    ddRo_S_avg = np.nanmean(ddRo_S_all, axis=0)
    Fa_S_std = np.nanstd(Fa_S_all, axis=0)
    Fl_S_std = np.nanstd(Fl_S_all, axis=0)
    Fd_S_std = np.nanstd(Fd_S_all, axis=0)
    ddRo_S_std = np.nanstd(ddRo_S_all, axis=0)
    S_err_avg = np.nanmean(S_err, axis=0)
    S_err_std = np.nanstd(S_err, axis=0)

#    Fa_I_avg = np.nanmean(Fa_I_all, axis=0)
#    Fl_I_avg = np.nanmean(Fl_I_all, axis=0)
#    Fd_I_avg = np.nanmean(Fd_I_all, axis=0)
#    ddRo_I_avg = np.nanmean(ddRo_I_all, axis=0)
#    Fa_I_std = np.nanstd(Fa_I_all, axis=0)
#    Fl_I_std = np.nanstd(Fl_I_all, axis=0)
#    Fd_I_std = np.nanstd(Fd_I_all, axis=0)
#    ddRo_I_std = np.nanstd(ddRo_I_all, axis=0)
#    I_err_avg = np.nanmean(I_err, axis=0)
#    I_err_std = np.nanstd(I_err, axis=0)

    Si = {}
    Si['Fa'] = Fa_S_all
    Si['Fl'] = Fl_S_all
    Si['Fd'] = Fd_S_all
    Si['ddRo'] = ddRo_S_all
    Si['Fa_avg'] = Fa_S_avg
    Si['Fl_avg'] = Fl_S_avg
    Si['Fd_avg'] = Fd_S_avg
    Si['ddRo_avg'] = ddRo_S_avg
    Si['Fa_std'] = Fa_S_std
    Si['Fl_std'] = Fl_S_std
    Si['Fd_std'] = Fd_S_std
    Si['ddRo_std'] = ddRo_S_std
    Si['err'] = S_err
    Si['err_avg'] = S_err_avg
    Si['err_std'] = S_err_std

#    Ii = {}
#    Ii['Fa'] = Fa_I_all
#    Ii['Fl'] = Fl_I_all
#    Ii['Fd'] = Fd_I_all
#    Ii['ddRo'] = ddRo_I_all
#    Ii['Fa_avg'] = Fa_I_avg
#    Ii['Fl_avg'] = Fl_I_avg
#    Ii['Fd_avg'] = Fd_I_avg
#    Ii['ddRo_avg'] = ddRo_I_avg
#    Ii['Fa_std'] = Fa_I_std
#    Ii['Fl_std'] = Fl_I_std
#    Ii['Fd_std'] = Fd_I_std
#    Ii['ddRo_std'] = ddRo_I_std
#    Ii['err'] = I_err
#    Ii['err_avg'] = I_err_avg
#    Ii['err_std'] = I_err_std

    Sn[snake_id] = Si
#    In[snake_id] = Ii


# %%

i = 2  # snake index
j = 1  # trial index
jj = 130  # height index
snake_id = snake_ids[i]


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

kw_args = dict(angles='xy', scale_units='xy', scale=1, width=.006)


Si = S[snake_id]
ddRo_S = Si['ddRo']
Fa_S, Fl_S, Fd_S = Si['Fa'], Si['Fl'], Si['Fd']

ax.quiver(0, 0, Fa_S[j, jj, 1], Fa_S[j, jj, 2], color='gray', **kw_args)
ax.quiver(0, 0, Fl_S[j, jj, 1], Fl_S[j, jj, 2], color='gray', **kw_args)
ax.quiver(0, 0, Fd_S[j, jj, 1], Fd_S[j, jj, 2], color='gray', **kw_args)
ax.quiver(0, 0, ddRo_S[j, jj, 1], ddRo_S[j, jj, 2], color='k', **kw_args)


Si = Sn[snake_id]
ddRo_S = Si['ddRo']
Fa_S, Fl_S, Fd_S = Si['Fa'], Si['Fl'], Si['Fd']

ax.quiver(0, 0, Fa_S[j, jj, 1], Fa_S[j, jj, 2], color='r', **kw_args)
ax.quiver(0, 0, Fl_S[j, jj, 1], Fl_S[j, jj, 2], color='r', **kw_args)
ax.quiver(0, 0, Fd_S[j, jj, 1], Fd_S[j, jj, 2], color='r', **kw_args)
#ax.quiver(0, 0, ddRo_S[j, jj, 1], ddRo_S[j, jj, 2], color='b', **kw_args)


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

skip = 2

fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

ax.plot(p[:, 0], p[:, 1], '-o', c='g', markevery=skip, lw=3)

ax.quiver(p[::skip, 0], p[::skip, 1], tv[::skip, 0], tv[::skip, 1], color='r',
          units='xy', width=.0015, zorder=10, label='tv')
ax.quiver(p[::skip, 0], p[::skip, 1], cv[::skip, 0], cv[::skip, 1], color='b',
          units='xy', width=.0015, zorder=10, label='cv')

ax.set_title('Body shape')

ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box-forced')
ax.set_xlabel('Lateral direction')
ax.set_ylabel('Transverse direction')
ax.margins(.1)
sns.despine()
fig.set_tight_layout(True)


fig, ax = plt.subplots()
ax.axvline(0, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

ax.plot(p[:, 0], p[:, 2], '-o', c='g', markevery=skip, lw=3)
ax.plot(p[0, 0], p[0, 2], 'ko')

ax.quiver(p[::skip, 0], p[::skip, 2], bv[::skip, 0], bv[::skip, 2], color='r',
          units='xy', width=.0015, zorder=10, label='bv')
ax.quiver(p[::skip, 0], p[::skip, 2], cv[::skip, 0], cv[::skip, 2], color='b',
          units='xy', width=.0015, zorder=10, label='cv')

ax.set_title('Body shape')

ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box-forced')
ax.set_xlabel('Lateral direction')
ax.set_ylabel('Transverse direction')
ax.margins(.1)
sns.despine()
fig.set_tight_layout(True)


# %%

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
ax1.invert_xaxis()
ax1.set_xlim(8.5, 0)
ax1.set_xticks(np.r_[0:9:2])

ax1.plot(Zi, Fd_S[:, :, 1].T, color='gray')
ax1.plot(Zi, Fd_S_prime[:, :, 1].T, color='r')

ax2.plot(Zi, Fd_S[:, :, 2].T, color='gray')
ax2.plot(Zi, Fd_S_prime[:, :, 2].T, color='r')

#ax1.plot(Zi, Fl_S_mag.T)
#ax2.plot(Zi, Fl_S_prime_mag.T - Fl_S_mag.T)

ax1.grid(True)
ax2.grid(True)
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
    Si = S[snake_id]

    label = 'Snake {}, {:.1f}g'.format(snake_id, masses[i])
    axs[i + 0, 0].text(8, .45, label, fontsize='x-small', color=colors_trial_id)

    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    Si = S[snake_id]
    de = np.deg2rad(delta_all[i]) / 2
    Fa_S, Fl_S, Fd_S = Si['Fa'], Si['Fl'], Si['Fd']
    ddRo_S = Si['ddRo']
    ntrials = Fl_S.shape[0]

    R = np.array([[1, 0, 0],
                  [0, np.cos(de), -np.sin(de)],
                  [0, np.sin(de),  np.cos(de)]])

    Fa_S_prime = np.zeros_like(Fa_S)
    Fl_S_prime = np.zeros_like(Fl_S)
    Fd_S_prime = np.zeros_like(Fd_S)

    for j in np.arange(ntrials):
        Fa_S_prime[j] = np.dot(R, Fa_S[j].T).T
        Fl_S_prime[j] = np.dot(R, Fl_S[j].T).T
        Fd_S_prime[j] = np.dot(R, Fd_S[j].T).T

    Fa_avg, Fa_std = np.nanmean(Fa_S_prime, axis=0), np.nanstd(Fa_S_prime, axis=0)
    Fl_avg, Fl_std = np.nanmean(Fl_S_prime, axis=0), np.nanstd(Fl_S_prime, axis=0)
    Fd_avg, Fd_std = np.nanmean(Fd_S_prime, axis=0), np.nanstd(Fd_S_prime, axis=0)

    # fills
    Fa_l, Fa_h = Fa_avg - Fa_std, Fa_avg + Fa_std
    ddRo_l, ddRo_h = Si['ddRo_avg'] - Si['ddRo_std'], Si['ddRo_avg'] + Si['ddRo_std']
    Fa_l_x, Fa_l_y, Fa_l_z = Fa_l.T
    Fa_h_x, Fa_h_y, Fa_h_z = Fa_h.T
    ddRo_l_x, ddRo_l_y, ddRo_l_z = ddRo_l.T
    ddRo_h_x, ddRo_h_y, ddRo_h_z = ddRo_h.T

    axs[i + 0, 0].fill_between(Zi, Fa_l_x, Fa_h_x, color=colors_trial_id)
    axs[i + 3, 0].fill_between(Zi, Fa_l_y, Fa_h_y, color=colors_trial_id)
    axs[i + 6, 0].fill_between(Zi, Fa_l_z, Fa_h_z, color=colors_trial_id)

    axs[i + 0, 1].fill_between(Zi, ddRo_l_x, ddRo_h_x, color=colors_trial_id)
    axs[i + 3, 1].fill_between(Zi, ddRo_l_y, ddRo_h_y, color=colors_trial_id)
    axs[i + 6, 1].fill_between(Zi, ddRo_l_z, ddRo_h_z, color=colors_trial_id)


    # averages
    axs[i + 0, 0].plot(Zi, Si['Fa_avg'][:, 0], 'k', lw=2)
    axs[i + 3, 0].plot(Zi, Si['Fa_avg'][:, 1], 'k', lw=2)
    axs[i + 6, 0].plot(Zi, Si['Fa_avg'][:, 2], 'k', lw=2)

    axs[i + 0, 1].plot(Zi, Si['ddRo_avg'][:, 0], 'k', lw=2)
    axs[i + 3, 1].plot(Zi, Si['ddRo_avg'][:, 1], 'k', lw=2)
    axs[i + 6, 1].plot(Zi, Si['ddRo_avg'][:, 2], 'k', lw=2)


# %%

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

    Si = S[snake_id]
    ntrials = Si['Fa'].shape[0]

    errl, errh = Si['err_avg'] - Si['err_std'], Si['err_avg'] + Si['err_std']
    xl, xh = errl[:, 0], errh[:, 0]
    yl, yh = errl[:, 1], errh[:, 1]
    zl, zh = errl[:, 2], errh[:, 2]
    axs[i, 0].fill_between(Zi, xl, xh, color=colors_trial_id)
    axs[i, 1].fill_between(Zi, yl, yh, color=colors_trial_id)
    axs[i, 2].fill_between(Zi, zl, zh, color=colors_trial_id)

    axs[i, 0].plot(Zi, Si['err_avg'][:, 0], 'k', lw=2)
    axs[i, 1].plot(Zi, Si['err_avg'][:, 1], 'k', lw=2)
    axs[i, 2].plot(Zi, Si['err_avg'][:, 2], 'k', lw=2)

    label = 'Snake {}, {:.1f}g'.format(snake_id, masses[i])
    axs[i, 0].text(8, .45, label, fontsize='x-small', color=colors_trial_id)

#fig.savefig(FIG.format('absolute errors 7x3'), **FIGOPT)



# %%

# %% Plot 9 x 2 force and acceleration traces

#snake_ids_i = [81, 91, 95]

fig, axs = plt.subplots(9, 2, #sharex=True, sharey=True,
                        figsize=(8, 12))


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
#    ax.set_ylim(-.2, 1)
    ax.set_xlim(8.5, 0)
#    ax.set_yticks([0, .25, .5, .75])
    ax.set_yticks([0, .75])
#    ax.set_yticks([0, 1])
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

# Z
for ax in axs[6:].flatten():
    ax.invert_xaxis()
    ax.set_ylim(-1.1, 1)
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

for i in np.arange(9):
    for j in np.arange(2):
        if i < 8:
            axs[i, j].set_xticklabels([])
        if j > 0:
            axs[i, j].set_yticklabels([])

for i in np.arange(3):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]

    Si = S[snake_id]
    ntrials = Si['Fa'].shape[0]

    # individual trials
    for j in np.arange(ntrials):
        axs[i + 0, 0].plot(Zi, Si['Fa'][j, :, 0], c=colors_trial_id)
        axs[i + 3, 0].plot(Zi, Si['Fa'][j, :, 1], c=colors_trial_id)
        axs[i + 6, 0].plot(Zi, Si['Fa'][j, :, 2], c=colors_trial_id)

        axs[i + 0, 1].plot(Zi, Si['ddRo'][j, :, 0], c=colors_trial_id)
        axs[i + 3, 1].plot(Zi, Si['ddRo'][j, :, 1], c=colors_trial_id)
        axs[i + 6, 1].plot(Zi, Si['ddRo'][j, :, 2], c=colors_trial_id)

    # averages
    axs[i + 0, 0].plot(Zi, Si['Fa_avg'][:, 0], 'k', lw=2)
    axs[i + 3, 0].plot(Zi, Si['Fa_avg'][:, 1], 'k', lw=2)
    axs[i + 6, 0].plot(Zi, Si['Fa_avg'][:, 2], 'k', lw=2)

    axs[i + 0, 1].plot(Zi, Si['ddRo_avg'][:, 0], 'k', lw=2)
    axs[i + 3, 1].plot(Zi, Si['ddRo_avg'][:, 1], 'k', lw=2)
    axs[i + 6, 1].plot(Zi, Si['ddRo_avg'][:, 2], 'k', lw=2)

sns.despine()

# %% Error time series

#fig, axs = plt.subplots(3, 3, sharex=True, sharey=True,
#                        figsize=(8, 8))
fig, axs = plt.subplots(7, 3, sharex=True, sharey=True,
                        figsize=(8, 8))

for ax in axs.flatten():
    ax.invert_xaxis()
    ax.set_ylim(-.75, .75)
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

#for i in np.arange(3):
for i in np.arange(7):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]

    Si = S[snake_id]
    ntrials = Si['Fa'].shape[0]

    for j in np.arange(ntrials):
        axs[i, 0].plot(Zi, Si['err'][j, :, 0], c=colors_trial_id)
        axs[i, 1].plot(Zi, Si['err'][j, :, 1], c=colors_trial_id)
        axs[i, 2].plot(Zi, Si['err'][j, :, 2], c=colors_trial_id)

    axs[i, 0].plot(Zi, Si['err_avg'][:, 0], 'k', lw=2)
    axs[i, 1].plot(Zi, Si['err_avg'][:, 1], 'k', lw=2)
    axs[i, 2].plot(Zi, Si['err_avg'][:, 2], 'k', lw=2)

sns.despine()


# %% Error as an angle

Zhigh, Zlow = 7, 1.5
idx_mid = np.where((Zi <= Zhigh) & (Zi >= Zlow))[0]
#idx_out = np.where((Zi > Zhigh) | (Zi < Zlow))[0]
idx_out1 = np.where(Zi > Zhigh)[0]
idx_out2 = np.where(Zi < Zlow)[0]

fig, axs = plt.subplots(nsnakes, 1, sharex=True, sharey=True,
                        figsize=(4, 9))
ax = axs[0]
ax.invert_xaxis()
#ax.set_ylim(-.75, .75)
ax.set_xlim(8.5, 0)
ax.set_xticks(np.r_[0:9:2])
ax.set_ylim(-90, 0)
sns.despine()

for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    ax = axs[i]
    Si = S[snake_id]
    delta = np.arctan2(Si['err'][:, :, 2], Si['err'][:, :, 1])
    delta_avg = np.arctan2(Si['err_avg'][:, 2], Si['err_avg'][:, 1])

    ntrials = Si['Fa'].shape[0]
#    for j in np.arange(ntrials):
#        ax.plot(Zi, np.rad2deg(delta[j]), c=colors_trial_id)
#    ax.plot(Zi, np.rad2deg(delta_avg), c='k')

    for j in np.arange(ntrials):
        ax.plot(Zi[idx_mid], np.rad2deg(delta[j, idx_mid]), c=colors_trial_id)
        ax.plot(Zi[idx_out1], np.rad2deg(delta[j, idx_out1]), c=colors_trial_id,
                alpha=.3)
        ax.plot(Zi[idx_out2], np.rad2deg(delta[j, idx_out2]), c=colors_trial_id,
                alpha=.3)
    ax.plot(Zi[idx_mid], np.rad2deg(delta_avg[idx_mid]), c='k')
    ax.plot(Zi[idx_out1], np.rad2deg(delta_avg[idx_out1]), c='k',
            alpha=.3)
    ax.plot(Zi[idx_out2], np.rad2deg(delta_avg[idx_out2]), c='k',
            alpha=.3)

#    ax.axhline(np.rad2deg(delta_avg[idx_mid]).mean(), ls='--',
#               c=colors_trial_id)


# %% Error as a rotation of the Fa,Y and Fa,Z vectors

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal', adjustable='box')
sns.despine()

for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    Si = S[snake_id]
    ax.plot(Si['err_avg'][:, 1], Si['err_avg'][:, 2],
            c=colors_trial_id)


fig, ax = plt.subplots()
ax.invert_xaxis()
#ax.set_ylim(-.75, .75)
ax.set_xlim(8.5, 0)
ax.set_xticks(np.r_[0:9:2])
sns.despine()

idx_mid = np.where((Zi < 6) & (Zi > 2))[0]

for i in np.arange(nsnakes):
    snake_id = snake_ids[i]
    colors_trial_id = colors[i]
    Si = S[snake_id]
    # delta_i = np.angle(Si['err_avg'][:, 1] + Si['err_avg'][:, 2] * 1j)
    delta_i = np.arctan2(Si['err_avg'][:, 2], Si['err_avg'][:, 1])
    ax.plot(Zi, np.rad2deg(delta_i), c=colors_trial_id)

    ax.axhline(np.rad2deg(delta_i[idx_mid]).mean(), ls='--',
               c=colors_trial_id)


# %% A_Z vs. Fa_Z vs. Z

# indices to use (determined from the ball drop experiment)
start = 8
stop = -10

g = 9.81

# C. paradisi to analyze
snake_ids = [81, 91, 95, 88, 90, 86, 94]
nsnakes = len(snake_ids)

# interpolation grid for forces
ZZ = np.linspace(8.5, 0, 200)
zr = np.zeros((nsnakes, len(ZZ)))
Fa_avg, Fa_std = zr.copy(), zr.copy()
ddRo_avg, ddRo_std = zr.copy(), zr.copy()

# colors for plots
colors = sns.color_palette('husl', nsnakes)

# make the plot
fig, axs = plt.subplots(nsnakes, 2, sharex=True, sharey=True,
                        figsize=(8, 12))

for ax in axs.flatten():
    ax.invert_xaxis()
    ax.set_ylim(-1.1, 1)
    ax.set_xlim(8.5, 0)
#    ax.set_yticks([0, .25, .5, .75])
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = colors[row]

    data_Fa = {}
    data_ddRo = {}

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times'][start:stop]
        Z = d['Ro_S'][start:stop, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * g

        Fa_S = d['Fa_S'][start:stop]
        ddRo_S = d['ddRo_S'][start:stop] / 1000

        Fa_S_non = Fa_S.sum(axis=1) / mg
        Fa_S_non[:, 2] -= 1  # subract off gravity (LHS is forces)
        ddRo_S_non = ddRo_S / g  # RHS is just acceleration
        # ddRo_S_non[:, 2] += 1  # add gravity on RHS

        # Z component
        Fa_non = Fa_S_non[:, 2]
        ddRo_non = ddRo_S_non[:, 2]

        Fa_interp = interp1d(Z, Fa_non, bounds_error=False, fill_value=np.nan)(ZZ)
        ddRo_interp = interp1d(Z, ddRo_non, bounds_error=False, fill_value=np.nan)(ZZ)

        data_Fa[i] = Fa_interp
        data_ddRo[i] = ddRo_interp

        axs[row, 0].plot(Z, ddRo_non, lw=1.5, c=colors_trial_id)
        axs[row, 1].plot(Z, Fa_non, lw=1.5, c=colors_trial_id)

    # mean force calculations
    if len(data_Fa) > 1:
        df_Fa = pd.DataFrame(data=data_Fa, index=ZZ)
        df_ddRo = pd.DataFrame(data=data_ddRo, index=ZZ)

        Fa_avg[row] = df_Fa.mean(axis=1)
        ddRo_avg[row] = df_ddRo.mean(axis=1)
        Fa_std[row] = df_Fa.std(axis=1)
        ddRo_std[row] = df_ddRo.std(axis=1)

        axs[row, 0].plot(ZZ, ddRo_avg[row], c='k', lw=2)
        axs[row, 1].plot(ZZ, Fa_avg[row], c='k', lw=2)
    else:
        Fa_avg[row] = np.nan
        ddRo_avg[row] = np.nan
        Fa_std[row] = np.nan
        ddRo_std[row] = np.nan

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('forces_Z'), **FIGOPT)


# store the averages
Fa_avg_Z = Fa_avg.copy()
ddRo_avg_Z = ddRo_avg.copy()
Fa_std_Z = Fa_std.copy()
ddRo_std_Z = ddRo_std.copy()


# %% A_Y vs. Fa_Y vs. Z

# indices to use (determined from the ball drop experiment)
start = 8
stop = -10

g = 9.81

# C. paradisi to analyze
snake_ids = [81, 91, 95, 88, 90, 86, 94]
nsnakes = len(snake_ids)

# interpolation grid for forces
ZZ = np.linspace(8.5, 0, 200)
zr = np.zeros((nsnakes, len(ZZ)))
Fa_avg, Fa_std = zr.copy(), zr.copy()
ddRo_avg, ddRo_std = zr.copy(), zr.copy()


# colors for plots
colors = sns.color_palette('husl', nsnakes)

# make the plot
fig, axs = plt.subplots(nsnakes, 2, sharex=True, sharey=True,
                        figsize=(8, 12))

for ax in axs.flatten():
    ax.invert_xaxis()
    ax.set_ylim(-.2, .8)
    ax.set_xlim(8.5, 0)
    ax.set_yticks([0, .25, .5, .75])
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = colors[row]

    data_Fa = {}
    data_ddRo = {}

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times'][start:stop]
        Z = d['Ro_S'][start:stop, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * g

        Fa_S = d['Fa_S'][start:stop]
        ddRo_S = d['ddRo_S'][start:stop] / 1000

        Fa_S_non = Fa_S.sum(axis=1) / mg
        # Fa_S_non[:, 2] -= 1  # subract off gravity
        ddRo_S_non = ddRo_S / g
        ddRo_S_non[:, 2] += 1  # add gravity on RHS

        # Y component
        Fa_non = Fa_S_non[:, 1]
        ddRo_non = ddRo_S_non[:, 1]

        Fa_interp = interp1d(Z, Fa_non, bounds_error=False, fill_value=np.nan)(ZZ)
        ddRo_interp = interp1d(Z, ddRo_non, bounds_error=False, fill_value=np.nan)(ZZ)

        data_Fa[i] = Fa_interp
        data_ddRo[i] = ddRo_interp

        axs[row, 0].plot(Z, ddRo_non, lw=1.5, c=colors_trial_id)
        axs[row, 1].plot(Z, Fa_non, lw=1.5, c=colors_trial_id)

    # mean force calculations
    if len(data_Fa) > 1:
        df_Fa = pd.DataFrame(data=data_Fa, index=ZZ)
        df_ddRo = pd.DataFrame(data=data_ddRo, index=ZZ)

        Fa_avg[row] = df_Fa.mean(axis=1)
        ddRo_avg[row] = df_ddRo.mean(axis=1)
        Fa_std[row] = df_Fa.std(axis=1)
        ddRo_std[row] = df_ddRo.std(axis=1)

        axs[row, 0].plot(ZZ, ddRo_avg[row], c='k', lw=2)
        axs[row, 1].plot(ZZ, Fa_avg[row], c='k', lw=2)
    else:
        Fa_avg[row] = np.nan
        ddRo_avg[row] = np.nan
        Fa_std[row] = np.nan
        ddRo_std[row] = np.nan

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('forces_Y'), **FIGOPT)


# store the averages
Fa_avg_Y = Fa_avg.copy()
ddRo_avg_Y = ddRo_avg.copy()
Fa_std_Y = Fa_std.copy()
ddRo_std_Y = ddRo_std.copy()


# %% A_X vs. Fa_X vs. Z

# indices to use (determined from the ball drop experiment)
start = 8
stop = -10

g = 9.81

# C. paradisi to analyze
snake_ids = [81, 91, 95, 88, 90, 86, 94]
nsnakes = len(snake_ids)

# interpolation grid for forces
ZZ = np.linspace(8.5, 0, 200)
zr = np.zeros((nsnakes, len(ZZ)))
Fa_avg, Fa_std = zr.copy(), zr.copy()
ddRo_avg, ddRo_std = zr.copy(), zr.copy()


# colors for plots
colors = sns.color_palette('husl', nsnakes)

# make the plot
fig, axs = plt.subplots(nsnakes, 2, sharex=True, sharey=True,
                        figsize=(8, 12))

for ax in axs.flatten():
    ax.invert_xaxis()
    ax.set_ylim(-.75, .75)
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = colors[row]

    data_Fa = {}
    data_ddRo = {}

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)

        times = d['times'][start:stop]
        Z = d['Ro_S'][start:stop, 2] / 1000
        mass_kg = float(d['mass']) / 1000
        mg = mass_kg * g

        Fa_S = d['Fa_S'][start:stop]
        ddRo_S = d['ddRo_S'][start:stop] / 1000

        Fa_S_non = Fa_S.sum(axis=1) / mg
        # Fa_S_non[:, 2] -= 1  # subract off gravity
        ddRo_S_non = ddRo_S / g
        ddRo_S_non[:, 2] += 1  # add gravity on RHS

        # X component
        Fa_non = Fa_S_non[:, 0]
        ddRo_non = ddRo_S_non[:, 0]

        Fa_interp = interp1d(Z, Fa_non, bounds_error=False, fill_value=np.nan)(ZZ)
        ddRo_interp = interp1d(Z, ddRo_non, bounds_error=False, fill_value=np.nan)(ZZ)

        data_Fa[i] = Fa_interp
        data_ddRo[i] = ddRo_interp

        axs[row, 0].plot(Z, ddRo_non, lw=1.5, c=colors_trial_id)
        axs[row, 1].plot(Z, Fa_non, lw=1.5, c=colors_trial_id)

    # mean force calculations
    if len(data_Fa) > 1:
        df_Fa = pd.DataFrame(data=data_Fa, index=ZZ)
        df_ddRo = pd.DataFrame(data=data_ddRo, index=ZZ)

        Fa_avg[row] = df_Fa.mean(axis=1)
        ddRo_avg[row] = df_ddRo.mean(axis=1)
        Fa_std[row] = df_Fa.std(axis=1)
        ddRo_std[row] = df_ddRo.std(axis=1)

        axs[row, 0].plot(ZZ, ddRo_avg[row], c='k', lw=2)
        axs[row, 1].plot(ZZ, Fa_avg[row], c='k', lw=2)
    else:
        Fa_avg[row] = np.nan
        ddRo_avg[row] = np.nan
        Fa_std[row] = np.nan
        ddRo_std[row] = np.nan

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('forces_X'), **FIGOPT)


# store the averages
Fa_avg_X = Fa_avg.copy()
ddRo_avg_X = ddRo_avg.copy()
Fa_std_X = Fa_std.copy()
ddRo_std_X = ddRo_std.copy()


# %% Analyzie the averages

Fa_avg = np.zeros((nsnakes, Fa_avg_X.shape[1], 3))
ddRo_avg = np.zeros((nsnakes, Fa_avg_X.shape[1], 3))

Fa_avg[:, :, 0], Fa_avg[:, :, 1], Fa_avg[:, :, 2] = Fa_avg_X, Fa_avg_Y, Fa_avg_Z
ddRo_avg[:, :, 0], ddRo_avg[:, :, 1], ddRo_avg[:, :, 2] = ddRo_avg_X, ddRo_avg_Y, ddRo_avg_Z


fig, axs = plt.subplots(3, 2, sharex=True, figsize=(9, 9))

for ax in axs.flatten():
    ax.invert_xaxis()
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])

for ax in axs[0]:
#    ax.set_ylim(-.75, .75)
    ax.set_ylim(-.25, .25)
    ax.axhline(0, color='gray', linestyle='--')

for ax in axs[1]:
    ax.set_ylim(-.2, .8)
    ax.axhline(0, color='gray', linestyle='--')

for ax in axs[2]:
    ax.set_ylim(-.1, 1.75)
    ax.axhline(1, color='gray', linestyle='--')

for ax in axs[:, 1]:
    ax.set_yticklabels([])

for j in np.arange(3):
    ax1, ax2 = axs[j]
    for i in np.arange(nsnakes):
        ax1.plot(ZZ, ddRo_avg[i, :, j], c=colors[i], lw=2)
        ax2.plot(ZZ, Fa_avg[i, :, j], c=colors[i], lw=2)
    ddRo_avg_all = np.nanmean(ddRo_avg, axis=0)
    Fa_avg_all = np.nanmean(Fa_avg, axis=0)
    ax1.plot(ZZ, ddRo_avg_all[:, j], 'k', lw=2.5)
    ax2.plot(ZZ, Fa_avg_all[:, j], 'k', lw=2.5)

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('forces_means'), **FIGOPT)


# %% Percent difference b/n the averages

#snake_ids = [81, 91, 95]
snake_ids = [81, 91, 95, 88, 90, 86]

error = ddRo_avg - Fa_avg


#fig, axs = plt.subplots(len(snake_ids), 3, sharex=True, sharey=True)
fig, axs = plt.subplots(len(snake_ids), 3, sharex=True, sharey=True,
                        figsize=(8, 10.25))

for ax in axs.flatten():
    ax.invert_xaxis()
#    ax.set_ylim(-.55, .55)
    ax.set_ylim(-.75, .75)
    ax.set_xlim(8.5, 0)
    ax.set_xticks(np.r_[0:9:2])
    ax.axhline(0, color='gray', linestyle='--')

for row, snake_id in enumerate(snake_ids):
    ax1, ax2, ax3 = axs[row]
    label = 'Snake {}, {:.1f}g'.format(snake_id, masses[row])
    ax1.text(8, .5, label, fontsize='x-small')
    ax1.plot(ZZ, error[row, :, 0], c=colors[row])
    ax2.plot(ZZ, error[row, :, 1], c=colors[row])
    ax3.plot(ZZ, error[row, :, 2], c=colors[row])
    ax1.legend(loc='upper right', fontsize='x-small')

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('forces_errors'), **FIGOPT)
fig.savefig(FIG.format('forces_errors_labels'), **FIGOPT)


# %%

snake_ids = [81, 91, 95]

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True,
                        figsize=(3.5, 8.15))

for ax in axs:
    ax.set_ylim(-.55, .55)
    ax.set_xlim(-.55, .55)
    ax.set_yticks([-.5, 0, .5])
    ax.set_xticks([-.5, 0, .5])
    ax.axvline(0, color='gray', lw=1)
    ax.axhline(0, color='gray', lw=1)

for row, snake_id in enumerate(snake_ids):
    ax = axs[row]
    ax.plot(error[row, :, 1], error[row, :, 2], c=colors[row])

plt.setp(axs, aspect=1.0, adjustable='box-forced')
sns.despine()
fig.set_tight_layout(True)


# %% Errors back into physical units


# %% Increased boost per unit area




# %% A_X vs. Fa_X vs. Z




# %%

# %% HO ANGULAR MOMENTUM AND YAW ANGLES




# %%


#fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
#                        figsize=(12, 10.7))
#axs = axs.flatten()

d_sn = {}
for col, snake_id in enumerate([81, 91, 95]):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = sns.husl_palette(n_colors=len(fn_names))

#    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#    ax2.set_title(snake_id)
#    sns.despine()
#    fig1.set_tight_layout(True)
#
#    fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True)
#    ax4.set_title(snake_id)
#    sns.despine()
#    fig2.set_tight_layout(True)
#
#    fig3, ax = plt.subplots()
#    ax.axhline(0, color='gray', lw=1)
#    ax.axvline(0, color='gray', lw=1)
#    sns.despine()

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True,
                                        figsize=(6, 12))
    ax1, ax2, ax3 = axs

    sns.despine()
    fig.set_tight_layout(True)
    ax1.set_title(snake_id)
#
#    fig, ax = plt.subplots()
#    ax.set_title(snake_id)
#    sns.despine()
#    fig.set_tight_layout(True)

    d_trials = {}
    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        d = np.load(fname)
        _label = '{0}'.format(trial_id)
        X, Y, Z = d['Ro_S'].T / 1000
        X -= Xo
        Y -= Yo
#        idx = np.where(Z < 7.25)[0]  # remove transient (used for COD as well)
        idx = np.where(Z > 0)[0]

        R, dR, ddR = d['R_Ic'][idx], d['dR_Ic'][idx], d['ddR_Ic'][idx]
#        R, dR, ddR = d['R_I'][idx], d['dR_I'][idx], d['ddR_I'][idx]
        R, dR, ddR = d['R_Sc'][idx], d['dR_Sc'][idx], d['ddR_Sc'][idx]
        R, dR, ddR = R / 1000, dR / 1000, ddR / 1000

        mass_spl = d['mass_spl'][idx]
        times = d['times'] - d['times'][0]
        times = times[idx]
        ntime = len(times)

        times2D = d['times2D'][idx]

        nbody = R.shape[1]
        vent_idx_spl = int(d['vent_idx_spl'])
        vent_loc = vent_idx_spl + 1

        t_coord, s_coord, SVL = d['t_coord'][idx], d['s_coord'][idx], d['SVL']
        t_coord_non = t_coord / SVL
        s_coord_non = s_coord / SVL

        idx_B = np.arange(nbody)[:vent_loc]
        idx_T = np.arange(nbody)[vent_loc + 1:]

        n_B, n_T = len(idx_B), len(idx_T)

        zr_B = np.zeros((ntime, n_B, 3))
        zr_T = np.zeros((ntime, n_T, 3))
        zr_tot = np.zeros((ntime, nbody, 3))
        ho_B, ho_T, ho_tot = zr_B.copy(), zr_T.copy(), zr_tot.copy()
        dho_B, dho_T, dho_tot = zr_B.copy(), zr_T.copy(), zr_tot.copy()
        l_B, l_T, l_tot = zr_B.copy(), zr_T.copy(), zr_tot.copy()

        for k in np.arange(ntime):
            R_B, dR_B, ddR_B = R[k, idx_B], dR[k, idx_B], ddR[k, idx_B]
            m_B = mass_spl[k, idx_B]
            ho_B[k] = np.cross(R_B, (m_B * dR_B.T).T)
            dho_B[k] = np.cross(R_B, (m_B * ddR_B.T).T)
            l_B[k] = (m_B * dR_B.T).T

            R_T, dR_T, ddR_T = R[k, idx_T], dR[k, idx_T], ddR[k, idx_T]
            m_T = mass_spl[k, idx_T]
            ho_T[k] = np.cross(R_T, (m_T * dR_T.T).T)
            dho_T[k] = np.cross(R_T, (m_T * ddR_T.T).T)
            l_T[k] = (m_T * dR_T.T).T

            R_tot, dR_tot, ddR_tot = R[k], dR[k], ddR[k]
            m_tot = mass_spl[k]
            ho_tot[k] = np.cross(R_tot, (m_tot * dR_tot.T).T)
            dho_tot[k] = np.cross(R_tot, (m_tot * ddR_tot.T).T)
            l_tot[k] = (m_tot * dR_tot.T).T

        # integrate along the body
        Ho_B = ho_B.sum(axis=1)
        Ho_T = ho_T.sum(axis=1)
        Ho_tot = ho_tot.sum(axis=1)
        dHo_B = dho_B.sum(axis=1)
        dHo_T = dho_T.sum(axis=1)
        dHo_tot = dho_tot.sum(axis=1)
        L_B = l_B.sum(axis=1)
        L_T = l_T.sum(axis=1)
        L_tot = l_tot.sum(axis=1)

        # norms
        Ho_B_norm = np.linalg.norm(Ho_B, axis=1)
        Ho_T_norm = np.linalg.norm(Ho_T, axis=1)
        Ho_tot_norm = np.linalg.norm(Ho_tot, axis=1)
        dHo_B_norm = np.linalg.norm(dHo_B, axis=1)
        dHo_T_norm = np.linalg.norm(dHo_T, axis=1)
        dHo_tot_norm = np.linalg.norm(dHo_tot, axis=1)
        L_B_norm = np.linalg.norm(L_B, axis=1)
        L_T_norm = np.linalg.norm(L_T, axis=1)
        L_tot_norm = np.linalg.norm(L_tot, axis=1)

        # store the values
        d_trials[trial_id] = dict(ho_B=ho_B, ho_T=ho_T, ho_tot=ho_tot,
                                  Ho_B=Ho_B, Ho_T=Ho_T, Ho_tot=Ho_tot,
                                  Ho_B_norm=Ho_B_norm, Ho_norm_T=Ho_T_norm, Ho_tot_norm=Ho_tot_norm,
                                  dho_B=dho_B, dho_T=dho_T, dho_tot=dho_tot,
                                  dHo_B=dHo_B, dHo_T=dHo_T, dHo_tot=dHo_tot,
                                  dHo_B_norm=dHo_B_norm, dHo_T_norm=dHo_T_norm, dHo_tot_norm=dHo_tot_norm,
                                  l_B=l_B, l_T=l_T, l_tot=l_tot,
                                  L_B=L_B, L_T=L_T, L_tot=L_tot,
                                  L_B_norm=L_B_norm, L_T_norm=L_T_norm, L_tot_norm=L_tot_norm,
                                  s_coord=s_coord, s_coord_non=s_coord_non,
                                  t_coord=t_coord, t_coord_non=t_coord_non,
                                  SVL=SVL, times2D=times2D, times=times,
                                  idx_B=idx_B, idx_T=idx_T)

#        ax1.plot(times, Ho_B_norm)
#        ax2.plot(times, Ho_T_norm)
##        ax3.plot(times, Ho_T_norm / Ho_B_norm)
#        ax3.plot(times, Ho_tot_norm)

#        k = 2
#        ax1.plot(times, Ho_B[:, k])
#        ax2.plot(times, Ho_T[:, k])
#        ax3.plot(times, Ho_T[:, k] / Ho_B[:, k])

#        k = 1
#        ax1.plot(times, np.abs(Ho_B[:, k]))
#        ax2.plot(times, np.abs(Ho_T[:, k]))
#        ax3.plot(times, np.abs(Ho_T[:, k]) / np.abs(Ho_B[:, k]))

        for k in np.arange(3):
            axs[k].plot(times, Ho_B[:, k], 'b')
            axs[k].plot(times, Ho_T[:, k], 'g')
#            axs[k].plot(times, Ho_tot[:, k], 'k')

#        # lots of noise here...
#        for k in np.arange(3):
#            axs[k].plot(times, dHo_B[:, k], 'b')
#            axs[k].plot(times, dHo_T[:, k], 'g')

#        for k in np.arange(3):
#            axs[k].plot(times, L_B[:, k], 'b')
#            axs[k].plot(times, L_T[:, k], 'g')

    d_sn[snake_id] = d_trials


# %%

d = d_sn[91][703]

k = 0

fig, ax = plt.subplots()
cax = ax.pcolormesh(d['times2D'], d['t_coord_non'], d['ho_tot'][:, :, k],
                    cmap=plt.cm.coolwarm)
fig.colorbar(cax, ax=ax)


fig, ax = plt.subplots()
ax.axhline(0, color='gray', linestyle='--')
ax.plot(d['times'], d['Ho_tot'][:, k])
sns.despine()


# %%

k = 1

fig, ax = plt.subplots()
cax = ax.pcolormesh(d['times2D'], d['t_coord_non'], d['dho_tot'][:, :, k],
                    cmap=plt.cm.coolwarm)
fig.colorbar(cax, ax=ax)


fig, ax = plt.subplots()
ax.axhline(0, color='gray', linestyle='--')
ax.plot(d['times'], d['dHo_tot'][:, k])
sns.despine()


# %%

k = 0

fig, ax = plt.subplots()
cax = ax.pcolormesh(d['times2D'], d['t_coord_non'], d['l_tot'][:, :, k],
                    cmap=plt.cm.coolwarm)
fig.colorbar(cax, ax=ax)


fig, ax = plt.subplots()
ax.axhline(0, color='gray', linestyle='--')
ax.plot(d['times'], d['L_tot'][:, k])
sns.despine()


# %%

fig, ax = plt.subplots()
cax = ax.pcolormesh(d['times2D'], d['t_coord_non'],
                    np.linalg.norm(d['ho_tot'], axis=2),
                    cmap=plt.cm.viridis)
fig.colorbar(cax, ax=ax)


# %%

figure(); plot(times, Ho_B_norm); plot(times, Ho_T_norm)


figure(); plot(times, Ho_T_norm / Ho_B_norm)


# %%
#        Ho_T = ho_T
#            dho_k = np.cross(R[k], (mass_spl[k] * ddR[k].T).T).sum(axis=0)
#            dho[k] = dho_k
#
#            L_k = (mass_spl[k] * dR[k].T).T.sum(axis=0)
#            L[k] = L_k
#
#        KE_k = .5 * mass_spl * np.sum(dR**2, axis=2)
#        KE = KE_k.sum(axis=1)
#
#        KE_zi = .5 * mass_spl * dR[:, :, 2]**2
#        KE_z = KE_zi.sum(axis=1)
#
#        L_i = (mass_spl.T * dR_Sc.T).T
#        L = L_i.sum(axis=1)

#        times2D = d['times2D'] - d['times2D'][0]
#        tc = d['t_coord'] / d['SVL']
#        fig, ax1 = plt.subplots()
#        cax = ax1.pcolormesh(times2D, tc, KE_k, cmap=plt.cm.viridis,
#                             vmin=0)
#        cbar = fig.colorbar(cax, ax=ax1, orientation='vertical', shrink=.5)
#        ax1.set_title('Snake {0}, trial {1}'.format(snake_id, trial_id))
#        ax1.set_ylim(0, tc.max())
#        ax1.set_xlim(0, times2D.max())
#        sns.despine(ax=ax1)
#        fig.set_tight_layout(True)

#        times2D = d['times2D'] - d['times2D'][0]
#        tc = d['t_coord'] / d['SVL']
#        fig, ax1 = plt.subplots()
#        cax = ax1.pcolormesh(times2D, tc, dR[:, :, 0], cmap=plt.cm.coolwarm)
#        cbar = fig.colorbar(cax, ax=ax1, orientation='vertical', shrink=.5)
#        ax1.set_title('Snake {0}, trial {1}'.format(snake_id, trial_id))
#        ax1.set_ylim(0, tc.max())
#        ax1.set_xlim(0, times2D.max())
#        sns.despine(ax=ax1)
#        fig.set_tight_layout(True)

#        ax1.plot(times, ho[:, 0], c=colors_trial_id[i], label=_label)
#        ax2.plot(times, ho[:, 1], c=colors_trial_id[i], label=_label)
#        ax3.plot(times, ho[:, 2], c=colors_trial_id[i], label=_label)

        ax1.plot(Z, ho[:, 0], c=colors_trial_id[i], label=_label)
        ax2.plot(Z, ho[:, 1], c=colors_trial_id[i], label=_label)
        ax3.plot(Z, ho[:, 2], c=colors_trial_id[i], label=_label)
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax1.set_xlim(8.5, 0)
        ax1.legend(loc='best', fontsize='xx-small', ncol=5)

#        ax1.plot(Z, ho[:, 0], c=colors_trial_id[i], label=_label)
#        ax2.plot(Z, ho[:, 1], c=colors_trial_id[i], label=_label)
#        ax3.plot(Z, ho[:, 2], c=colors_trial_id[i], label=_label)
#        ax1.set_xlim(-.25, 5.5)
#        ax1.legend(loc='best', fontsize='xx-small', ncol=5)

#        ax.plot(Z, KE, c=colors_trial_id[i], label=_label)
#        ax.set_xlim(8.5, 0)
#        ax.plot(times, KE, c=colors_trial_id[i], label=_label)
#        ax.plot(Z, KE_z, c=colors_trial_id[i], label=_label)
#        ax.set_xlim(8.5, 0)
#        ax.legend(loc='best', fontsize='xx-small', ncol=5)
#
#        ax1.plot(Z, L[:, 0], c=colors_trial_id[i], label=_label)
#        ax2.plot(Z, L[:, 1], c=colors_trial_id[i], label=_label)
#        ax3.plot(Z, L[:, 2], c=colors_trial_id[i], label=_label)
#        ax1.grid(True)
#        ax2.grid(True)
#        ax3.grid(True)
#        ax1.set_xlim(8.5, 0)
#        ax1.legend(loc='best', fontsize='xx-small', ncol=5)

#        ax1.plot(times, yaw, c=colors_trial_id[i], label=_label)
#        ax2.plot(times, ho[:, 2], c=colors_trial_id[i], label=_label)
#
#        ax3.plot(times, dyaw, c=colors_trial_id[i], label=_label)
#        ax4.plot(times, dho[:, 2], c=colors_trial_id[i], label=_label)
#
#        yaw_mean = np.mean(yaw - yaw[0])
#        ho_mean = np.mean(ho - ho[0], axis=0)
#
#        ax.plot(yaw_mean, ho_mean[2], 'o')
#
#        ax.plot(yaw, ho[:, 2], c=colors_trial_id[i], lw=2)



# %% Number of markers on the body

fnames = ret_fnames()

zr_int = np.zeros(ntrials, dtype=np.int)
nmark_body, nmark_tail = zr_int.copy(), zr_int.copy()
sn_id, t_id = zr_int.copy(), zr_int.copy()
for i, fname in enumerate(fnames):
    d = np.load(fname)
    snake_id, trial_id = trial_info(fname)
    nmark_body[i] = int(d['vent_idx']) + 1
    nmark_tail[i] = d['pf_I'].shape[1] - int(d['vent_idx']) - 1
    sn_id[i] = snake_id
    t_id[i] = trial_id

data = dict(ID=sn_id, trial=t_id, body=nmark_body, tail=nmark_tail)

df = pd.DataFrame(data)

fig, ax = plt.subplots()
sns.swarmplot(x="ID", y="body", hue="ID", data=df, ax=ax)
sns.swarmplot(x="ID", y="tail", hue="ID", data=df, ax=ax)
ax.get_legend().set_visible(False)
ax.grid(axis='y')
ax.set_ylim(0, 15)
ax.set_xlabel('Snake ID')
ax.set_ylabel('Number of body and tail markers')
sns.despine()

fig.savefig(FIG.format('marker_counts'), **FIGOPT)
