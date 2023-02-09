#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:31:06 2017

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

from m_smoothing import findiff

import seaborn as sns
from mayavi import mlab

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Helvetica'}
sns.set('notebook', 'ticks', font='Helvetica',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_wings/{}.pdf'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}


# %%

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


def ret_fnames_may(snake=None, trial=None):

    from glob import glob

    if snake is None:
        snake = '*'
    if trial is None:
        trial = '*'

    fn_trial = '{0}_{1}.npz'.format(trial, snake)
    fn_proc = '../Data/Processed Qualisys output - March 2016/'
    fn_search = fn_proc + fn_trial

    return sorted(glob(fn_search))


def trial_info(fname):
    trial_id = fname.split('/')[-1][:3]
    snake_id = fname.split('/')[-1][4:6]

    return int(snake_id), int(trial_id)


# %% Format with degrees

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


# %% Information about the trials

fn_names = ret_fnames()
snakes = []
for fname in fn_names:
    snake_id, trial_id = trial_info(fname)
    snakes.append(int(snake_id))

snakes = np.array(sorted(snakes))

snake_ids = np.unique(snakes)
#snake_ids =  np.array([30, 31, 32, 33, 35, 81, 88, 90, 91, 95])

ntrials = len(snakes)
nsnakes = len(snake_ids)

colors = sns.husl_palette(n_colors=nsnakes)

total_counts = 0
snake_counts = {}
snake_colors = {}
for i, snake_id in enumerate(snake_ids):
    snake_counts[snake_id] = np.sum(snakes == snake_id)
    snake_colors[snake_id] = colors[i]
    total_counts += snake_counts[snake_id]


# average start in X and Y of com as the new
X0, Y0 = np.zeros(ntrials), np.zeros(ntrials)
for i, fname in enumerate(ret_fnames()):
    snake_id, trial_id = trial_info(fname)
    d = np.load(fname)
    X, Y, Z = d['Ro_I'].T / 1000  # m
    X0[i] = X[0]
    Y0[i] = Y[0]
Xo = X0.mean()  # 0.44458394864868178
Yo = Y0.mean()  # -4.684127769871548

# for fource calculations
grav = 9.81  # m/s^2 (gravitational acceleration)
rho = 1.17  # kg/m^3 (air density)


# %% Rotate to the flow frame

def rotate_to_flow(d):
    """Rotate from inertial to flow frame, such that CoM velocity
    is in the forward y direction. Also check that the rotation
    is correct from I2F and S2F.
    """

    # unpack variables
    C_I2S = d['C_I2S']
    gamma = d['gamma']
    R_Ic, R_Sc = d['R_Ic'], d['R_Sc']
    dRo_I = d['dRo_I']
    foils_Ic = d['foils_Ic']
    foils_Sc = d['foils_Sc']
    ntime = d['ntime']

    # what we want to return
    R_Fc = np.zeros_like(R_Sc)
    R_Fcheck = np.zeros_like(R_Sc)
    C_I2F = np.zeros_like(C_I2S)
    C_S2F = np.zeros_like(C_I2S)
    foils_Fc = np.zeros_like(foils_Ic)
    foils_Fcheck = np.zeros_like(foils_Sc)
    dRo_F = np.zeros_like(dRo_I)

    for i in np.arange(ntime):
        gamma_i = -gamma[i]

        # rotate about the x axis
        C_S2F[i] = np.array([[1, 0, 0],
                             [0, np.cos(gamma_i), np.sin(gamma_i)],
                             [0, -np.sin(gamma_i), np.cos(gamma_i)]])

        C_I2F[i] = np.dot(C_S2F[i], C_I2S[i])
        R_Fc[i] = np.dot(C_I2F[i], R_Ic[i].T).T
        R_Fcheck[i] = np.dot(C_S2F[i], R_Sc[i].T).T
        dRo_F[i] = np.dot(C_I2F[i], dRo_I[i].T).T

        for j in np.arange(foils_Ic.shape[1]):
            foils_Fc[i, j] = np.dot(C_I2F[i], foils_Ic[i, j].T).T
            foils_Fcheck[i, j] = np.dot(C_S2F[i], foils_Sc[i, j].T).T

    assert(np.allclose(R_Fc, R_Fcheck))
    assert(np.allclose(foils_Fc, foils_Fcheck))

    return R_Fc, foils_Fc, dRo_F


# %%

# load in trial
fname = ret_fnames(81, 507)[0]
d = np.load(fname)

# unpack varaibles
dt = float(d['dt'])
times = d['times']
ntime = d['ntime']
vent_loc = d['vent_idx_spl'] + 1
SVL = d['SVL_avg']
start = d['idx_pts'][1]  # 0 is the virtual marker
#start = 0
snon = d['t_coord'][0, start:vent_loc] / SVL
snonf = d['t_coord'][0] / SVL
aoa = d['aoa'][:, start:vent_loc]
beta = d['beta'][:, start:vent_loc]
s_plot = np.arange(vent_loc)
nbody = len(snon)

# flow frame
R_Fc, foils_Fc, dRo_F = rotate_to_flow(d)

# body position
R = R_Fc

x, y, z = R[:, start:vent_loc].T  # TODO
xf, yf, zf = R.T

x, y, z = x.T, y.T, z.T
xf, yf, zf = xf.T, yf.T, zf.T

# bending angles
dRds = d['Tdir_I']

psi = np.arcsin(dRds[:, start:vent_loc, 2])
theta = np.arctan2(dRds[:, start:vent_loc, 0], -dRds[:, start:vent_loc, 1])

psi = np.unwrap(psi, axis=1)
theta = np.unwrap(theta, axis=1)

# mean remove
psi_mean = psi.mean(axis=1)
theta_mean = theta.mean(axis=1)
psi = (psi.T - psi_mean).T
theta = (theta.T - theta_mean).T

# detrent the angles
d_psi_pp = np.zeros((ntime, 2))
d_psi_fit = np.zeros((ntime, nbody))
psi_detrend = np.zeros((ntime, nbody))
d_theta_pp = np.zeros((ntime, 2))
d_theta_fit = np.zeros((ntime, nbody))
theta_detrend = np.zeros((ntime, nbody))
for i in np.arange(ntime):
    pp = np.polyfit(snon, psi[i], 1)
    y_lin = np.polyval(pp, snon)
    y_fit = psi[i] - y_lin
    d_psi_pp[i] = pp
    d_psi_fit[i] = y_lin
    psi_detrend[i] = y_fit

    pp = np.polyfit(snon, theta[i], 1)
    y_lin = np.polyval(pp, snon)
    y_fit = theta[i] - y_lin
    d_theta_pp[i] = pp
    d_theta_fit[i] = y_lin
    theta_detrend[i] = y_fit

# only remove trend on vertical wave
psi = psi_detrend.copy()

# zero crossings of theta = U-bends
snon_zr = []
diff_snon_zr = []
theta_zr, psi_zr = [], []
x_zr, y_zr, z_zr = [], [], []
aoa_zr, beta_zr = [], []
for i in np.arange(ntime):
    ti, pi = theta[i], psi[i]
    xi, yi, zi = x[i], y[i], z[i]
    aoai, betai = np.rad2deg(aoa[i]), np.rad2deg(beta[i])

    i0 = np.where(np.diff(np.signbit(theta[i])))[0]
    i1 = i0 + 1
    frac = np.abs(ti[i0] / (ti[i1] - ti[i0]))

    zrs_i = snon[i0] + frac * (snon[i1] - snon[i0])
    snon_zr.append(zrs_i)
    diff_snon_zr.append(np.diff(zrs_i))

    theta_zr.append(ti[i0] + frac * (ti[i1] - ti[i0]))
    psi_zr.append(pi[i0] + frac * (pi[i1] - pi[i0]))
    x_zr.append(xi[i0] + frac * (xi[i1] - xi[i0]))
    y_zr.append(yi[i0] + frac * (yi[i1] - yi[i0]))
    z_zr.append(zi[i0] + frac * (zi[i1] - zi[i0]))

    aoa_zr.append(aoai[i0] + frac * (aoai[i1] - aoai[i0]))
    beta_zr.append(betai[i0] + frac * (betai[i1] - betai[i0]))


# zero crossings of x = wings for gap + stagger calculations
snon_x0 = []
theta_x0, psi_x0 = [], []
x_x0, y_x0, z_x0 = [], [], []
aoa_x0, beta_x0 = [], []
for i in np.arange(ntime):
    ti, pi = theta[i], psi[i]
    xi, yi, zi = x[i], y[i], z[i]
    aoai, betai = np.rad2deg(aoa[i]), np.rad2deg(beta[i])

    i0 = np.where(np.diff(np.signbit(xi)))[0]
    i1 = i0 + 1
    frac = np.abs(xi[i0] / (xi[i1] - xi[i0]))

    zrs_i = snon[i0] + frac * (snon[i1] - snon[i0])
    snon_x0.append(zrs_i)

    theta_x0.append(ti[i0] + frac * (ti[i1] - ti[i0]))
    psi_x0.append(pi[i0] + frac * (pi[i1] - pi[i0]))
    x_x0.append(xi[i0] + frac * (xi[i1] - xi[i0]))
    y_x0.append(yi[i0] + frac * (yi[i1] - yi[i0]))
    z_x0.append(zi[i0] + frac * (zi[i1] - zi[i0]))

    aoa_x0.append(aoai[i0] + frac * (aoai[i1] - aoai[i0]))
    beta_x0.append(betai[i0] + frac * (betai[i1] - betai[i0]))


# number of wings
nwing = np.zeros(ntime).astype(np.int)
for i in np.arange(ntime):
    nwing[i] = len(x_x0[i])

# make the zero corssings 2D arrays
nwing_max = np.max(nwing)
x_X0 = np.zeros((ntime, nwing_max)) * np.nan
y_X0 = np.zeros((ntime, nwing_max)) * np.nan
z_X0 = np.zeros((ntime, nwing_max)) * np.nan
aoa_X0 = np.zeros((ntime, nwing_max)) * np.nan
beta_X0 = np.zeros((ntime, nwing_max)) * np.nan
for i in np.arange(ntime):
    x_X0[i, :nwing[i]] = x_x0[i]
    y_X0[i, :nwing[i]] = y_x0[i]
    z_X0[i, :nwing[i]] = z_x0[i]
    aoa_X0[i, :nwing[i]] = aoa_x0[i]
    beta_X0[i, :nwing[i]] = beta_x0[i]

# gaps and staggers --- dimensional
gaps_dim = np.diff(y_X0)
staggers_dim = np.diff(z_X0)

# gaps and staggers --- non-dimensional by max chord length
cmax = d['chord_spl'][0].max()
gaps = gaps_dim / cmax
staggers = staggers_dim / cmax


# %%

# plot the number of wings
fig, ax = plt.subplots()
#ax.step(times, nwing, where='pre')
ax.step(np.arange(len(nwing)), nwing, where='pre')
ax.set_yticks([0, 1, 2, 3, 4])
sns.despine()


# plot the staggers and gaps - normalized
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.axhline(0, color='gray', lw=1)
ax1.axvline(0, color='gray', lw=1)
ax2.axhline(0, color='gray', lw=1)
ax2.axvline(0, color='gray', lw=1)

ax1.plot(gaps[:, 0], staggers[:, 0], 'b')
ax2.plot(gaps[:, 1], staggers[:, 1], 'g')

#ax1.set_aspect('equal', adjustable='box')
#ax2.set_aspect('equal', adjustable='box')
plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')
ax1.set_xlabel('gap (c)')
ax2.set_xlabel('gap (c)')
ax1.set_ylabel('stagger (c)')
sns.despine()


# %% Angle of attack and sweep angle

fig, ax = plt.subplots()

for i in np.arange(nwing_max)[::-1]:
    ax.plot(aoa_X0[:, i], 'o-')
sns.despine()


fig, ax = plt.subplots()

for i in np.arange(nwing_max):
    ax.plot(beta_X0[:, i], 'o-')
sns.despine()


# %%

i = 80

aoai = np.rad2deg(aoa[i])
betai = np.rad2deg(beta[i])

fig, ax = plt.subplots()

ax.plot(snon, aoai)
ax.plot(snon, betai)

ax.plot(snon_x0[i], aoa_x0[i], 'ko', mfc='none', mew=2, mec='k')
ax.plot(snon_x0[i], beta_x0[i], 'ko', mfc='none', mew=2, mec='k')

ax.plot(snon_zr[i], aoa_zr[i], 'ro', mfc='none', mew=2, mec='r')
ax.plot(snon_zr[i], beta_zr[i], 'ro', mfc='none', mew=2, mec='r')

sns.despine()


# %%

i = 120

fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.axhline(0, lw=1, color='gray')
#ax1.axvline(0, lw=1, color='gray')
#ax2.axhline(0, lw=1, color='gray')
#ax2.axvline(0, lw=1, color='gray')

ax1.plot(1.05 * np.r_[x.min(), x.max()], [0, 0], c='gray', lw=1)
ax1.plot([0, 0], 1.05 * np.r_[y.min(), y.max()], c='gray', lw=1)
ax2.plot(1.05 * np.r_[y.min(), y.max()], [0, 0], c='gray', lw=1)
ax2.plot([0, 0], 1.05 * np.r_[z.min(), z.max()], c='gray', lw=1)

ax1.plot(x[i], y[i], 'g-', lw=4)
ax2.plot(y[i], z[i], 'g-', lw=4)
ax1.plot(xf[i], yf[i], 'go-', markevery=1000)
ax2.plot(yf[i], zf[i], 'go-', markevery=1000)

ax1.plot(x_zr[i], y_zr[i], 'ro', mfc='none', mew=2, mec='r')
ax1.plot(x_x0[i], y_x0[i], 'ko', mfc='none', mew=2, mec='k')
ax2.plot(y_zr[i], z_zr[i], 'ro', mfc='none', mew=2, mec='r')
ax2.plot(y_x0[i], z_x0[i], 'ko', mfc='none', mew=2, mec='k')

ax1.set_aspect('equal', adjustable='box')
ax2.set_aspect('equal', adjustable='box')
ax1.axis('off')
ax2.axis('off')


# %%

# %%

fig, ax = plt.subplots()
ax.axis('equal', adjustable='box')
#ax.axhline(0, color='gray', lw=1)
#ax.axvline(0, color='gray', lw=1)
ax.plot([0, 0], 1.05 * np.r_[y.min(), y.max()], c='gray', lw=1)
ax.plot(1.05 * np.r_[x.min(), x.max()], [0, 0], c='gray', lw=1)
#ax.set_ylim(-150, 150)
#ax.set_yticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
#ax.set_xlim(0, 100)
#ax.grid(True, axis='y')
#sns.despine()
ax.axis('off')
fig.set_tight_layout(True)

#i = 116
body_line, = ax.plot(x[i], y[i], 'go-', lw=5, ms=10, markevery=1000)

o_zr, = ax.plot(x_zr[i], y_zr[i], 'ro',
                mfc='none', mew=2, mec='r')

o_zr, = ax.plot(x_x0[i], y_x0[i], 'ko',
                mfc='none', mew=2, mec='k')


# %% Single frame from the movie

figsize = (5.5, 4)
fig, ax = plt.subplots(figsize=figsize)
ax.axhline(0, color='gray', lw=1)
ax.set_ylim(-150, 150)
ax.set_yticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
#ax.grid(True, axis='y')
sns.despine()
fig.set_tight_layout(True)

#i = 116
i = 170

theta_line, = ax.plot(100 * snon, np.rad2deg(theta[i]), lw=3, label='Lateral')
theta_line, = ax.plot(100 * snon, np.rad2deg(psi[i]), lw=3, label='Lateral')

o_zr_theta, = ax.plot(100 * snon_zr[i], np.rad2deg(theta_zr[i]), 'ro',
                      mfc='none', mew=2, mec='r')

o_zr_theta, = ax.plot(100 * snon_x0[i], np.rad2deg(theta_x0[i]), 'ko',
                      mfc='none', mew=2, mec='k')


o_zr_theta, = ax.plot(100 * snon_zr[i], np.rad2deg(psi_zr[i]), 'ro',
                      mfc='none', mew=2, mec='r')

o_zr_theta, = ax.plot(100 * snon_x0[i], np.rad2deg(psi_x0[i]), 'ko',
                      mfc='none', mew=2, mec='k')


# add degree symbol to angles
fig.canvas.draw()
ticks = ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
ax.set_yticklabels(newticks)

ax.set_xlabel('Distance along body (%SVL)')
ax.set_ylabel('Body angles')


# %%

fig, ax = plt.subplots()
ax.hexbin(gaps.flatten(), staggers.flatten(), gridsize=50, cmap='viridis',
          extent=[-12, 2, -12, 2])
ax.plot(gaps[:, 2], staggers[:, 2], 'r+', mew=1)
ax.plot(gaps[:, 1], staggers[:, 1], 'y+', mew=1)
ax.plot(gaps[:, 0], staggers[:, 0], 'w+', mew=1)
ax.set_xlim(-12, 2)
ax.set_ylim(-12, 2)
ax.set_aspect('equal', adjustable='box')
sns.despine()


# %%

z0 = d['Ro_S'][:, 2].copy()
gamma = np.rad2deg(d['gamma'])

z0_non = np.abs(z0 - z0[0]) / np.ptp(z0)

figure()
plot(z0_non, gamma)
#plot(z0, gamma)




# %% Iterate through all trials, calculate gaps and staggers

nwing_all, nwing_max_all, nwing_avg_all, nwing_std_all = [], [], [], []
gaps_all, staggers_all = [], []
aoa_all, beta_all = [], []
znons, zdims, gammas = [], [], []

fnames = ret_fnames()
for i, fname in enumerate(fnames):
    print(i, fname.split('/')[-1].split('.')[0])

    snake_id, trial_id = trial_info(fname)
    if snake_id < 60:
        print('skipping C. ornata')
        continue

    d = np.load(fname)

    # unpack varaibles
    dt = float(d['dt'])
    times = d['times']
    ntime = d['ntime']
    vent_loc = d['vent_idx_spl'] + 1
    SVL = d['SVL_avg']
    start = d['idx_pts'][1]  # 0 is the virtual marker
    #start = 0
    snon = d['t_coord'][0, start:vent_loc] / SVL
    snonf = d['t_coord'][0] / SVL
    aoa = d['aoa'][:, start:vent_loc]
    beta = d['beta'][:, start:vent_loc]
    s_plot = np.arange(vent_loc)
    nbody = len(snon)

    # flow frame
    R_Fc, foils_Fc, dRo_F = rotate_to_flow(d)

    # body position
    R = R_Fc

    x, y, z = R[:, start:vent_loc].T  # TODO
    xf, yf, zf = R.T

    x, y, z = x.T, y.T, z.T
    xf, yf, zf = xf.T, yf.T, zf.T

    # bending angles
    dRds = d['Tdir_I']

    psi = np.arcsin(dRds[:, start:vent_loc, 2])
    theta = np.arctan2(dRds[:, start:vent_loc, 0], -dRds[:, start:vent_loc, 1])

    psi = np.unwrap(psi, axis=1)
    theta = np.unwrap(theta, axis=1)

    # mean remove
    psi_mean = psi.mean(axis=1)
    theta_mean = theta.mean(axis=1)
    psi = (psi.T - psi_mean).T
    theta = (theta.T - theta_mean).T

    # detrent the angles
    d_psi_pp = np.zeros((ntime, 2))
    d_psi_fit = np.zeros((ntime, nbody))
    psi_detrend = np.zeros((ntime, nbody))
    d_theta_pp = np.zeros((ntime, 2))
    d_theta_fit = np.zeros((ntime, nbody))
    theta_detrend = np.zeros((ntime, nbody))
    for i in np.arange(ntime):
        pp = np.polyfit(snon, psi[i], 1)
        y_lin = np.polyval(pp, snon)
        y_fit = psi[i] - y_lin
        d_psi_pp[i] = pp
        d_psi_fit[i] = y_lin
        psi_detrend[i] = y_fit

        pp = np.polyfit(snon, theta[i], 1)
        y_lin = np.polyval(pp, snon)
        y_fit = theta[i] - y_lin
        d_theta_pp[i] = pp
        d_theta_fit[i] = y_lin
        theta_detrend[i] = y_fit

    # only remove trend on vertical wave
    psi = psi_detrend.copy()

    # zero crossings of theta = U-bends
    snon_zr = []
    diff_snon_zr = []
    theta_zr, psi_zr = [], []
    x_zr, y_zr, z_zr = [], [], []
    for i in np.arange(ntime):
        ti, pi = theta[i], psi[i]
        xi, yi, zi = x[i], y[i], z[i]

        i0 = np.where(np.diff(np.signbit(theta[i])))[0]
        i1 = i0 + 1
        frac = np.abs(ti[i0] / (ti[i1] - ti[i0]))

        zrs_i = snon[i0] + frac * (snon[i1] - snon[i0])
        snon_zr.append(zrs_i)
        diff_snon_zr.append(np.diff(zrs_i))

        theta_zr.append(ti[i0] + frac * (ti[i1] - ti[i0]))
        psi_zr.append(pi[i0] + frac * (pi[i1] - pi[i0]))
        x_zr.append(xi[i0] + frac * (xi[i1] - xi[i0]))
        y_zr.append(yi[i0] + frac * (yi[i1] - yi[i0]))
        z_zr.append(zi[i0] + frac * (zi[i1] - zi[i0]))

    # zero crossings of x = wings for gap + stagger calculations
    snon_x0 = []
    theta_x0, psi_x0 = [], []
    x_x0, y_x0, z_x0 = [], [], []
    aoa_x0, beta_x0 = [], []
    for i in np.arange(ntime):
        ti, pi = theta[i], psi[i]
        xi, yi, zi = x[i], y[i], z[i]
        aoai, betai = np.rad2deg(aoa[i]), np.rad2deg(beta[i])

        i0 = np.where(np.diff(np.signbit(xi)))[0]
        i1 = i0 + 1
        frac = np.abs(xi[i0] / (xi[i1] - xi[i0]))

        zrs_i = snon[i0] + frac * (snon[i1] - snon[i0])
        snon_x0.append(zrs_i)

        theta_x0.append(ti[i0] + frac * (ti[i1] - ti[i0]))
        psi_x0.append(pi[i0] + frac * (pi[i1] - pi[i0]))
        x_x0.append(xi[i0] + frac * (xi[i1] - xi[i0]))
        y_x0.append(yi[i0] + frac * (yi[i1] - yi[i0]))
        z_x0.append(zi[i0] + frac * (zi[i1] - zi[i0]))

        aoa_x0.append(aoai[i0] + frac * (aoai[i1] - aoai[i0]))
        beta_x0.append(betai[i0] + frac * (betai[i1] - betai[i0]))


    # number of wings
    nwing = np.zeros(ntime).astype(np.int)
    for i in np.arange(ntime):
        nwing[i] = len(x_x0[i])

    # make the zero corssings 2D arrays
    # nwing_max = np.max(nwing)
    nwing_max = 8
    x_X0 = np.zeros((ntime, nwing_max)) * np.nan
    y_X0 = np.zeros((ntime, nwing_max)) * np.nan
    z_X0 = np.zeros((ntime, nwing_max)) * np.nan
    aoa_X0 = np.zeros((ntime, nwing_max)) * np.nan
    beta_X0 = np.zeros((ntime, nwing_max)) * np.nan
    for i in np.arange(ntime):
        x_X0[i, :nwing[i]] = x_x0[i]
        y_X0[i, :nwing[i]] = y_x0[i]
        z_X0[i, :nwing[i]] = z_x0[i]
        aoa_X0[i, :nwing[i]] = aoa_x0[i]
        beta_X0[i, :nwing[i]] = beta_x0[i]

    # gaps and staggers --- dimensional
    gaps_dim = -np.diff(y_X0, axis=1)
    staggers_dim = -np.diff(z_X0, axis=1)

    # gaps and staggers --- non-dimensional by max chord length
    cmax = d['chord_spl'][0].max()
    gaps = gaps_dim / cmax
    staggers = staggers_dim / cmax

    # store the arrays
    nwing_all.append(nwing)
    nwing_max_all.append(nwing_max)
    nwing_avg_all.append(nwing.mean())
    nwing_std_all.append(np.array(nwing.std()))
    gaps_all.append(gaps)
    staggers_all.append(staggers)
    aoa_all.append(aoa_X0)
    beta_all.append(beta_X0)

    # store arrays to plot against
    z0 = d['Ro_S'][:, 2].copy()
#    z0_non = np.abs(z0 - z0[0]) / np.ptp(z0)
#    z0_non = z0 / np.ptp(z0)
    z0_non = z0 / z0[0]
    znons.append(z0_non)
    zdims.append(z0 / 1000)
    gammas.append(np.rad2deg(d['gamma']))

nwing_max_all = np.array(nwing_max_all)
nwing_avg_all = np.array(nwing_avg_all)


# %%

ntrials = 0
for fname in fnames:
    snake_id, trial_id = trial_info(fname)
    if snake_id > 60:
        ntrials += 1


g, s = np.array([]), np.array([])
g1, s1 = np.array([]), np.array([])
g2, s2 = np.array([]), np.array([])
g3, s3 = np.array([]), np.array([])

aoa_1, aoa_2, aoa_3 = np.array([]), np.array([]), np.array([])
beta_1, beta_2, beta_3 = np.array([]), np.array([]), np.array([])

gamma = np.array([])

for i in np.arange(ntrials):
    g = np.r_[g, gaps_all[i].flatten()]
    s = np.r_[s, staggers_all[i].flatten()]

    try:
        g3 = np.r_[g3, gaps_all[i][:, 2].flatten()]
        s3 = np.r_[s3, staggers_all[i][:, 2].flatten()]
    except:
        pass

    g1 = np.r_[g1, gaps_all[i][:, 0].flatten()]
    s1 = np.r_[s1, staggers_all[i][:, 0].flatten()]
    g2 = np.r_[g2, gaps_all[i][:, 1].flatten()]
    s2 = np.r_[s2, staggers_all[i][:, 1].flatten()]
    # g3 = np.r_[g3, gaps_all[i][:, 2].flatten()]
    # s3 = np.r_[s3, staggers_all[i][:, 2].flatten()]

    aoa_1 = np.r_[aoa_1, aoa_all[i][:, 0].flatten()]
    aoa_2 = np.r_[aoa_2, aoa_all[i][:, 1].flatten()]
    aoa_3 = np.r_[aoa_3, aoa_all[i][:, 2].flatten()]

    beta_1 = np.r_[beta_1, beta_all[i][:, 0].flatten()]
    beta_2 = np.r_[beta_2, beta_all[i][:, 1].flatten()]
    beta_3 = np.r_[beta_3, beta_all[i][:, 2].flatten()]

    gamma = np.r_[gamma, gammas[i]]


# %% Gap and stagger hexbin --- ALL COUNTS

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=.5, ls='-')
ax.axhline(0, color='k', lw=.5, ls='-')
#ax.plot(0, 0, 'kx', ms=8, mew=1.5)

cax = ax.hexbin(g, s, mincnt=0, gridsize=(30, 12), cmap=plt.cm.gray_r,
                linewidths=0.2, extent=(-5, 10, -2, 10))
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='small')

#counts = cax.get_array()
#ncnts = np.count_nonzero(counts)
#verts = cax.get_offsets()
#ax.plot(verts[:, 0], verts[:, 1], 'k.')  # https://stackoverflow.com/a/13754416

#for offc in np.arange(verts.shape[0]):
#    binx, biny = verts[offc][0], verts[offc][1]
#    if counts[offc]:
#        plt.plot(binx, biny, 'k.', zorder=100)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='r', mew=1.)

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xlim(10, -5)
ax.set_ylim(10, -2)

ax.set_yticks([10, 8, 6, 4, 2, 0, -2])
ax.set_xticks([10, 8, 6, 4, 2, 0, -2, -4])

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)


#fig.savefig(FIG.format('stagger vs gap all'), **FIGOPT)


# %% COUNTS

# https://stackoverflow.com/a/38940369
# https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(13, 6))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="3%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)
axs_flat = axs.flatten()

for k in np.arange(2):
    for j in np.arange(4):
        ax = axs[k, j]

        ub = 1 - j * .25
        lb = 1 - (j + 1) * .25

        gaps_j = np.array([])
        staggers_j = np.array([])
        aoas_j0, aoas_j1 = np.array([]), np.array([])
        betas_j0, betas_j1 = np.array([]), np.array([])
        for i in np.arange(ntrials):
            znon = znons[i]
            idx = np.where((znon < ub) & (znon >= lb))[0]
            gaps_j = np.r_[gaps_j, gaps_all[i][idx, k]]
            staggers_j = np.r_[staggers_j, staggers_all[i][idx, k]]
            aoas_j0 = np.r_[aoas_j0, aoa_all[i][idx, 0]]  # first foil aoa
            aoas_j1 = np.r_[aoas_j1, aoa_all[i][idx, 1]]  # second foil aoa
            betas_j0 = np.r_[betas_j0, beta_all[i][idx, 0]]  # first foil aoa
            betas_j1 = np.r_[betas_j1, beta_all[i][idx, 1]]  # second foil aoa

        hax = ax.hexbin(gaps_j, staggers_j, gridsize=(30, 12),
                        cmap=plt.cm.gray_r,
                        extent=(-5, 10, -2, 10), mincnt=0,
                       vmin=0, vmax=40,
                        linewidths=0.2)
        ax.axhline(0, color='gray', lw=.5)
        ax.axvline(0, color='gray', lw=.5)

        if k == 0:
            ax.set_title(r'{0} > z/h$_o$ > {1}'.format(ub, lb),
                         fontsize='small')

        # for some reason this is not working...
        if k == 0 and j == 1:
            ax.set_xlabel('Gap (c)')
            ax.set_ylabel('Stagger (c)')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlim(10, -5)
    ax.set_ylim(10, -2)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
ax.cax.set_yticks(np.arange(0, 41, 10))
ax.cax.set_ylabel('Counts')

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

fig.savefig(FIG.format('stagger vs gap z_h0 2x4'), **FIGOPT)


# %% Gap + stagger by height, combine 1st and 2nd wings

# gaps and staggers for the different heights
gaps_height = []
stag_height = []
for j in np.arange(4):
    gaps_j = np.array([])
    stag_j = np.array([])

    ub = 1 - j * .25
    lb = 1 - (j + 1) * .25

    for i in np.arange(ntrials):
        znon = znons[i]
        idx = np.where((znon < ub) & (znon >= lb))[0]

        # just look at the first two wings
        gaps_j = np.r_[gaps_j, gaps_all[i][idx, :2].flatten()]
        stag_j = np.r_[stag_j, staggers_all[i][idx, :2].flatten()]

    gaps_height.append(gaps_j)
    stag_height.append(stag_j)


# Set up figure and image grid
fig = plt.figure(figsize=(5, 12))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(4, 1),
                 direction='column',
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="top",
                 cbar_mode="single",
                 cbar_size="5%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_column)[0]
axs_flat = axs.flatten()

for j in np.arange(4):
    ax = axs[j]

    hax = ax.hexbin(gaps_height[j], stag_height[j], gridsize=(30, 12),
                    cmap=plt.cm.gray_r,
                    extent=(-5, 10, -2, 10), mincnt=0,
                    vmin=0, vmax=75,
                    linewidths=0.2)
    ax.axhline(0, color='gray', lw=.5)
    ax.axvline(0, color='gray', lw=.5)

#    ax.set_title(r'{0} > z/h$_o$ > {1}'.format(ub, lb),
#                 fontsize='small')

    if j == 3:
        ax.set_xlabel('Gap (c)', fontsize='small')
        ax.set_ylabel('Stagger (c)', fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlim(10, -5)
    ax.set_ylim(10, -2)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
ax.cax.set_xticks(np.arange(0, 76, 25))
ax.cax.set_xlabel('Counts')

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('stagger vs gap z_h0 4x1'), **FIGOPT)


# %%

# gaps and staggers for the different heights
gaps_height = []
stag_height = []
aoa1_height = []
aoa2_height = []
beta1_height = []
beta2_height = []
ubs, lbs = [], []

ub = np.r_[1, .75, .5, .25]
lb = np.r_[.75, .5, .25, 0]

for j in np.arange(4):
    gaps_j = np.array([])
    stag_j = np.array([])
    aoa1_j, aoa2_j = np.array([]), np.array([])
    beta1_j, beta2_j = np.array([]), np.array([])

#    ub = 1 - j * .25
#    lb = 1 - (j + 1) * .25
#    ubs.append(ub)
#    lbs.append(lb)

    for i in np.arange(ntrials):
        znon = znons[i]
        idx = np.where((znon < ub[j]) & (znon >= lb[j]))[0]

        # just look at the first two wings
        gaps_j = np.r_[gaps_j, gaps_all[i][idx, :2].flatten()]
        stag_j = np.r_[stag_j, staggers_all[i][idx, :2].flatten()]
        aoa1_j = np.r_[aoa1_j, aoa_all[i][idx, :2].flatten()]
        aoa2_j = np.r_[aoa2_j, aoa_all[i][idx, 1:3].flatten()]
        beta1_j = np.r_[beta1_j, beta_all[i][idx, :2].flatten()]
        beta2_j = np.r_[beta2_j, beta_all[i][idx, 1:3].flatten()]

    gaps_height.append(gaps_j)
    stag_height.append(stag_j)
    aoa1_height.append(aoa1_j)
    aoa2_height.append(aoa2_j)
    beta1_height.append(beta1_j)
    beta2_height.append(beta2_j)


# %%

from scipy.stats import iqr

# Set up figure and image grid
fig = plt.figure(figsize=(13, 3))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="5%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)[0]
axs_flat = axs.flatten()

for j in np.arange(4):
    ax = axs[j]

    hax = ax.hexbin(gaps_height[j], stag_height[j], gridsize=(30, 12),
                    cmap=plt.cm.gray_r,
                    extent=(-5, 10, -2, 10), mincnt=0,
                    vmin=0, vmax=75,
                    linewidths=0.2)
    ax.axhline(0, color='gray', lw=.5)
    ax.axvline(0, color='gray', lw=.5)

#    ax.axhline(np.nanmean(stag_height[j]), color='k', lw=1)
#    ax.axvline(np.nanmean(gaps_height[j]), color='k', lw=1)
#    ax.axhline(np.nanmedian(stag_height[j]), color='b', lw=1)
#    ax.axvline(np.nanmedian(gaps_height[j]), color='b', lw=1)
    med_gaps = np.nanmedian(gaps_height[j])
    med_stag = np.nanmedian(stag_height[j])
    iqr_gaps = iqr(gaps_height[j], nan_policy='omit')
    iqr_stag = iqr(stag_height[j], nan_policy='omit')
    q75_gaps, q25_gaps = np.nanpercentile(gaps_height[j], [75, 25])
    q75_stag, q25_stag = np.nanpercentile(stag_height[j], [75, 25])

    print('{0} > z/h_o > {1}'.format(ub[j], lb[j]))
    print('med g s: ({:.1f} {:.1f})'.format(med_gaps, med_stag))
    print('G IQR(Q1, Q3): {:.1f} ({:.1f}, {:.1f})'.format(iqr_gaps, q25_gaps, q75_gaps))
    print('S IQR(Q1, Q3): {:.1f} ({:.1f}, {:.1f})'.format(iqr_stag, q25_stag, q75_stag))
    print()

#    ax.plot(med_gaps, med_stag, 'o', mfc='none', mec='y', mew=2)

    # med iqr
#    ax.errorbar(med_gaps, med_stag, yerr=iqr_stag / 2, xerr=iqr_gaps / 2,
#                ecolor='y', elinewidth=2)

    # med q25 q75
    ax.plot([q25_gaps, q75_gaps], [med_stag, med_stag], 'y')
    ax.plot([med_gaps, med_gaps], [q25_stag, q75_stag], 'y')

    ax.set_title(r'{0} > z/h$_o$ > {1}'.format(ub[j], lb[j]),
                 fontsize='small')

    # for some reason this is not working...
    if j == 0:
        ax.set_xlabel('Gap (c)', fontsize='small')
        ax.set_ylabel('Stagger (c)', fontsize='small')

for gp in gaps_piv:
    for sp in staggers_piv:
        for ax in axs_flat:
            ax.plot(gp, sp, '+', color='r', ms=4.5, mew=1)


plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlim(10, -5)
    ax.set_ylim(10, -2)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
ax.cax.set_yticks(np.arange(0, 76, 25))
ax.cax.set_ylabel('Counts')

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('0 stagger vs gap z_h0 piv med iqr 1x4'), **FIGOPT)
#fig.savefig(FIG.format('0 stagger vs gap z_h0 piv med q25 q75 1x4'), **FIGOPT)


# %% ANGLE OF ATTACK --- JOINT BY FRAC THROUGH GLIDE

# Set up figure and image grid
fig = plt.figure(figsize=(13, 3))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="5%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)[0]
axs_flat = axs.flatten()

for j in np.arange(4):
    ax = axs[j]

    hax = ax.hexbin(aoa1_height[j], aoa2_height[j], gridsize=(20, 20),
                    cmap=plt.cm.Purples,
                    extent=[0, 90, 0, 90], mincnt=1,
                    linewidths=0.2, vmin=0, vmax=150)
#                    extent=[-10, 90, -10, 90], mincnt=0,
#    ax.axhline(0, color='gray', lw=.5)
#    ax.axvline(0, color='gray', lw=.5)

#    med_aoa1 = np.nanmedian(aoa1_height[j])
#    med_aoa2 = np.nanmedian(aoa2_height[j])
#    iqr_aoa1 = iqr(aoa1_height[j], nan_policy='omit')
#    iqr_aoa2 = iqr(aoa2_height[j], nan_policy='omit')
#    q75_aoa1, q25_aoa1 = np.nanpercentile(aoa1_height[j], [75, 25])
#    q75_aoa2, q25_aoa2 = np.nanpercentile(aoa2_height[j], [75, 25])

    med_aoa1 = np.nanmedian(aoa1_height[j])
    med_aoa2 = np.nanmedian(aoa2_height[j])
    iqr_aoa1 = iqr(aoa1_height[j], nan_policy='omit')
    iqr_aoa2 = iqr(aoa2_height[j], nan_policy='omit')
    q75_aoa1, q25_aoa1 = np.nanpercentile(aoa1_height[j], [75, 25])
    q75_aoa2, q25_aoa2 = np.nanpercentile(aoa2_height[j], [75, 25])

    print('{0} > z/h_o > {1}'.format(ub[j], lb[j]))
    print('med a1 a2: ({:.1f} {:.1f})'.format(med_aoa1, med_aoa2))
    print('a1 IQR(Q1, Q3): {:.1f} ({:.1f}, {:.1f})'.format(iqr_aoa1, q25_aoa1, q75_aoa1))
    print('a2 IQR(Q1, Q3): {:.1f} ({:.1f}, {:.1f})'.format(iqr_aoa2, q25_aoa2, q75_aoa2))
    print()

    # med q25 q75
    ax.plot([q25_aoa1, q75_aoa1], [med_aoa2, med_aoa2], 'r')
    ax.plot([med_aoa1, med_aoa1], [q25_aoa2, q75_aoa2], 'r')

    ax.set_title(r'{0} > z/h$_o$ > {1}'.format(ub[j], lb[j]),
                 fontsize='small')

    ax.plot([-10, 95], [-10, 95], '-', lw=1, color='gray')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([0, 30, 60, 90])

#    ax.invert_xaxis()
#    ax.invert_yaxis()

#    ax.set_xlim(-10, 90)
#    ax.set_ylim(-10, 90)
#    ax.set_xlim(30, 90)
#    ax.set_ylim(30, 90)
    ax.set_xlim(27.5, 92.5)
    ax.set_ylim(27.5, 92.5)

    ax.xaxis.set_major_formatter(degree_formatter)
    ax.yaxis.set_major_formatter(degree_formatter)

#    ax.grid(True)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
#ax.cax.set_yticks([-10, 0, 30, 60, 90])
#ax.cax.set_yticks([30, 45, 60, 75, 90])
ax.cax.set_ylabel('Counts', fontsize='small')

#for piv_aoa in [0, 20, 30, 40, 60]:
#    # cbar.ax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)
#    ax.cax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('0 aoa2 vs aoa1 z_h0 1x4 med q25 q75'), **FIGOPT)


# %% |SWEEP ANGLE| --- JOINT BY FRAC THROUGH GLIDE

# Set up figure and image grid
fig = plt.figure(figsize=(13, 3))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="5%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)[0]
axs_flat = axs.flatten()

for j in np.arange(4):
    ax = axs[j]

    hax = ax.hexbin(np.abs(beta1_height[j]), np.abs(beta2_height[j]),
                    gridsize=(20, 20),
                    cmap=plt.cm.Oranges,
                    extent=[0, 90, 0, 90], mincnt=1,
                    linewidths=0.2, vmin=0, vmax=50)

    ax.set_title(r'{0} > z/h$_o$ > {1}'.format(ub[j], lb[j]),
                 fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([0, 30, 60, 90])

    ax.set_xlim(-2.5, 92.5)
    ax.set_ylim(-2.5, 92.5)

    ax.xaxis.set_major_formatter(degree_formatter)
    ax.yaxis.set_major_formatter(degree_formatter)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
#ax.cax.set_yticks([-10, 0, 30, 60, 90])
#ax.cax.set_yticks([30, 45, 60, 75, 90])
ax.cax.set_ylabel('Counts', fontsize='small')

#for piv_aoa in [0, 20, 30, 40, 60]:
#    # cbar.ax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)
#    ax.cax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('0 beta2 vs beta1 z_h0 1x4'), **FIGOPT)


# %% SWEEP ANGLE ABS --- JOINT BY FRAC THROUGH GLIDE

# https://stackoverflow.com/a/38940369
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(13, 6))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="3%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)
axs_flat = axs.flatten()

for k in np.arange(2):
    for j in np.arange(4):
        ax = axs[k, j]
        gaps_j = np.array([])
        staggers_j = np.array([])
        aoas_j0, aoas_j1 = np.array([]), np.array([])
        betas_j0, betas_j1 = np.array([]), np.array([])
        for i in np.arange(ntrials):
            znon = znons[i]
            lb, ub = j * .25, (j + 1) * .25
            idx = np.where((znon >= lb) & (znon < ub))[0]
            gaps_j = np.r_[gaps_j, gaps_all[i][idx, k]]
            staggers_j = np.r_[staggers_j, staggers_all[i][idx, k]]
            aoas_j0 = np.r_[aoas_j0, aoa_all[i][idx, k]]  # first foil aoa
            aoas_j1 = np.r_[aoas_j1, aoa_all[i][idx, k + 1]]  # second foil aoa
            betas_j0 = np.r_[betas_j0, beta_all[i][idx, k]]  # first foil aoa
            betas_j1 = np.r_[betas_j1, beta_all[i][idx, k + 1]]  # second foil aoa

        hax = ax.hexbin(np.abs(betas_j0), np.abs(betas_j1),
                        gridsize=30, cmap='inferno',
                        extent=[0, 90, 0, 90],
                        vmin=0, mincnt=0,
                        linewidths=0.2)
#        ax.axhline(0, color='k', lw=.5)
#        ax.axvline(0, color='k', lw=.5)

        if k == 0:
            # txt = '$\overline{\mathrm{z}}$'
            txt = '$\\bar \mathrm{z}$'
            ax.set_title(r'{0} < {1} < {2}'.format(lb, txt, ub), fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([0, 30, 60, 90])

#    ax.invert_xaxis()
#    ax.invert_yaxis()

    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)

    ax.xaxis.set_major_formatter(degree_formatter)
    ax.yaxis.set_major_formatter(degree_formatter)

    ax.grid(True)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
#ax.cax.set_yticks([-10, 0, 30, 60, 90])
#ax.cax.set_yticks([30, 45, 60, 75, 90])
ax.cax.set_ylabel('Counts', fontsize='small')

#for piv_aoa in [0, 20, 30, 40, 60]:
#    # cbar.ax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)
#    ax.cax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('stagger vs gap beta_all z_h0'), **FIGOPT)


# %% Gap and stagger hexbin --- ALL COUNTS

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=.5, ls='-')
ax.axhline(0, color='k', lw=.5, ls='-')
#ax.plot(0, 0, 'kx', ms=8, mew=1.5)

cax = ax.hexbin(gaps_height[3], stag_height[3], mincnt=0, gridsize=(30, 12),
                cmap=plt.cm.gray_r,
                linewidths=0.2, extent=(-5, 10, -2, 10))
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='small')

#counts = cax.get_array()
#ncnts = np.count_nonzero(counts)
#verts = cax.get_offsets()
#ax.plot(verts[:, 0], verts[:, 1], 'k.')  # https://stackoverflow.com/a/13754416

#for offc in np.arange(verts.shape[0]):
#    binx, biny = verts[offc][0], verts[offc][1]
#    if counts[offc]:
#        plt.plot(binx, biny, 'k.', zorder=100)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='r', mew=1.)

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xlim(10, -5)
ax.set_ylim(10, -2)

ax.set_yticks([10, 8, 6, 4, 2, 0, -2])
ax.set_xticks([10, 8, 6, 4, 2, 0, -2, -4])

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)


#fig.savefig(FIG.format('stagger vs gap all end of glide'), **FIGOPT)


# %% Relationship between glide angle and angle of attack

fig, ax = plt.subplots()

#for i in np.arange(len(fnames)):
#    ax.plot(gammas[i], aoa_all[i][:, 2], 'o')

for i in np.arange(len(fnames)):
    ax.plot(gamma, aoa_1, 'o')

sns.despine()


# %% Gap and stagger --- vs. height

k = 0

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.4, 6.875))

for i in np.arange(ntrials):
    ax1.plot(znons[i], gaps_all[i][:, k], 'o', ms=3)
    ax2.plot(znons[i], staggers_all[i][:, k], 'o', ms=3)

#ax.invert_xaxis()
ax2.set_xlabel('z/h0')
ax1.set_ylabel('Gap (c)')
ax2.set_ylabel('Stagger (c)')
sns.despine()


# %% Gap and stagger --- vs. height

k = 0

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.4, 6.875))

for i in np.arange(ntrials):
    ax1.plot(zdims[i], gaps_all[i][:, k], 'o', ms=3)
    ax2.plot(zdims[i], staggers_all[i][:, k], 'o', ms=3)

ax1.invert_xaxis()
ax2.set_xlabel('Z (m)')
ax1.set_ylabel('Gap (c)')
ax2.set_ylabel('Stagger (c)')
sns.despine()


# %%

fig, ax = plt.subplots()

for i in np.arange(ntrials):
    ax.plot(znons[i], staggers_all[i][:, k], 'o')
sns.despine()


# %% Hexbin of gamma and aoa

fig, ax = plt.subplots()
#ax.axvline(0, color='w', lw=.5)
#ax.axhline(0, color='w', lw=.5)
# ax.plot(0, 0, 'ro', ms=5)

cax = ax.hexbin(gamma, aoa_1, gridsize=30, cmap='plasma',
                vmin=0, mincnt=1, extent=[0, 93, 0, 93],
                linewidths=0.2)
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
ax.plot([0, 93], [0, 93], 'k--')
cbar.set_label('Counts')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 93)
ax.set_ylim(0, 93)
ax.set_xticks([0, 30, 60, 90])
ax.set_yticks([0, 30, 60, 90])
ax.set_xlabel("Glide angle")
ax.set_ylabel("Angle of attack")
ax.xaxis.set_major_formatter(degree_formatter)
ax.yaxis.set_major_formatter(degree_formatter)
sns.despine(ax=ax)

fig.savefig(FIG.format('aoa_1 vs gamma'), **FIGOPT)


# %% Jointplot of gamma and aoa

#sns.jointplot(gamma, aoa_1, kind='kde')

grid = sns.jointplot(gamma, aoa_1, kind='kde', color="m")
#grid.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
#grid.ax_joint.collections[0].set_alpha(0)
grid.set_axis_labels("Glide angle", "Angle of attack")
grid.ax_joint.xaxis.set_major_formatter(degree_formatter)
grid.ax_joint.yaxis.set_major_formatter(degree_formatter)
grid.ax_joint.set_xlim(0, 93)
grid.ax_joint.set_ylim(0, 93)
grid.ax_joint.set_xticks([0, 30, 60, 90])
grid.ax_joint.set_yticks([0, 30, 60, 90])
grid.ax_joint.plot([0, 93], [0, 93], 'k--')

grid.fig.savefig(FIG.format('aoa_1 vs gamma kde'), **FIGOPT)


# %% Hexbin subplots of gamma and aoa for three wings

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,
                                    figsize=(12, 4))
#ax.axvline(0, color='w', lw=.5)
#ax.axhline(0, color='w', lw=.5)
# ax.plot(0, 0, 'ro', ms=5)

cax = ax1.hexbin(gamma, aoa_1, gridsize=30, cmap='plasma',
                vmin=0, vmax=600, mincnt=1, extent=[0, 93, 0, 93],
                linewidths=0.2)
cax = ax2.hexbin(gamma, aoa_2, gridsize=30, cmap='plasma',
                vmin=0, vmax=600, mincnt=1, extent=[0, 93, 0, 93],
                linewidths=0.2)
cax = ax3.hexbin(gamma, aoa_3, gridsize=30, cmap='plasma',
                vmin=0, vmax=600, mincnt=1, extent=[0, 93, 0, 93],
                linewidths=0.2)
#cbar = fig.colorbar(cax, ax=ax, shrink=.75)
#cbar.set_label('Counts')
for ax in (ax1, ax2, ax3):
    ax.plot([0, 93], [0, 93], 'k--')
    ax.set_xlim(0, 93)
    ax.set_ylim(0, 93)
    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([0, 30, 60, 90])
    ax.xaxis.set_major_formatter(degree_formatter)
    ax.yaxis.set_major_formatter(degree_formatter)
    sns.despine(ax=ax)

ax1.set_xlabel("Glide angle")
ax1.set_ylabel("Angle of attack")
plt.setp((ax1, ax2, ax3), aspect=1.0, adjustable='box-forced')

fig.savefig(FIG.format('aoa_1 vs gamma wings'), **FIGOPT)


# %%

k = 0

fig, ax = plt.subplots()
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)

for i in np.arange(ntrials):
    ax.plot(gaps_all[i][:, k], staggers_all[i][:, k], 'o', ms=1.5)

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.set_aspect('equal', adjustable='box')
sns.despine()


# %% Gap and stagger hexbin --- ALL COUNTS

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=.5, ls='-')
ax.axhline(0, color='k', lw=.5, ls='-')
ax.plot(0, 0, 'kx', ms=8, mew=1.5)

cax = ax.hexbin(g, s, mincnt=0, gridsize=(30, 12), cmap=plt.cm.gray_r,
                linewidths=0.2, extent=(-5, 10, -2, 10))
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='small')

#counts = cax.get_array()
#ncnts = np.count_nonzero(counts)
#verts = cax.get_offsets()
#ax.plot(verts[:, 0], verts[:, 1], 'k.')  # https://stackoverflow.com/a/13754416
#for offc in np.arange(verts.shape[0]):
#    binx, biny = verts[offc][0], verts[offc][1]
#    if counts[offc]:
#        plt.plot(binx, biny, 'k.', zorder=100)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='r', mew=1.)

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xlim(10, -5)
ax.set_ylim(10, -2)

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)


#fig.savefig(FIG.format('stagger vs gap all'), **FIGOPT)


# %% Gap and stagger hexbin -- AOA FIRST SET MEAN

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=.5)
ax.axhline(0, color='k', lw=.5)

cax = ax.hexbin(g1, s1, aoa_1, gridsize=100, cmap='inferno',
                vmin=-10, vmax=90, linewidths=0.2)
cbar = fig.colorbar(cax, ax=ax, shrink=.85)
#cbar.ax.yaxis.set_major_formatter(degree_formatter)
cbar.set_ticks(np.arange(0, 91, 30))
# add degree symbol to angles
fig.canvas.draw()
ticks = cbar.ax.get_yticklabels()
newticks = []
for tick in ticks:
    text = tick.get_text()
    newticks.append(text + u'\u00B0')
cbar.ax.set_yticklabels(newticks)
cbar.set_label('Average angle of attack (deg)')

for piv_aoa in [0, 20, 30, 40, 60]:
    cbar.ax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='k', mew=1.5)

ax.set_xlabel('Gap')
ax.set_ylabel('Stagger')

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xlim(10, -5)
ax.set_ylim(10, -2)

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)

#fig.savefig(FIG.format('stagger vs gap aoa_1'), **FIGOPT)


# %% Gap and stagger hexbin -- AOA FIRST SET STD

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots()
ax.axvline(0, color='w', lw=.5)
ax.axhline(0, color='w', lw=.5)

cax = ax.hexbin(g1, s1, aoa_1, gridsize=100, cmap='inferno')#,
#                vmin=0, vmax=90)
cbar = fig.colorbar(cax, ax=ax, shrink=.75)


for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='w', mew=1.5)

ax.set_xlabel('Gap')
ax.set_ylabel('Stagger')

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xlim(10, -5)
ax.set_ylim(10, -2)

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)


# %% Gap and stagger hexbin + kde plot
#
#fig, ax = plt.subplots()
#ax.axvline(0, color='r', lw=1)
#ax.axhline(0, color='r', lw=1)
#ax.plot(0, 0, 'ro', ms=5)
#
#ax.hexbin(g, s, gridsize=100, cmap='viridis', marginals=True)
#
#cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
##sns.kdeplot(g, s, cmap=cmap, n_levels=60, shade=True, ax=ax)
##sns.kdeplot(g[~np.isnan(g)], s[~np.isnan(s)], cmap=cmap, n_levels=10, ax=ax)
#
#for gp in gaps_piv:
#    for sp in staggers_piv:
#        ax.plot(gp, sp, '+', color='w', mew=1.5)
#
#ax.set_xlabel('Gap')
#ax.set_ylabel('Stagger')
#
#ax.set_aspect('equal', adjustable='box')
#sns.despine(ax=ax)
#fig.set_tight_layout(True)


# %% COUNTS

# https://stackoverflow.com/a/38940369
# https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(13, 6))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="3%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)
axs_flat = axs.flatten()

for k in np.arange(2):
    for j in np.arange(4):
        ax = axs[k, j]
        gaps_j = np.array([])
        staggers_j = np.array([])
        aoas_j0, aoas_j1 = np.array([]), np.array([])
        betas_j0, betas_j1 = np.array([]), np.array([])
        for i in np.arange(ntrials):
            znon = znons[i]
            lb, ub = j * .25, (j + 1) * .25
            idx = np.where((znon >= lb) & (znon < ub))[0]
            gaps_j = np.r_[gaps_j, gaps_all[i][idx, k]]
            staggers_j = np.r_[staggers_j, staggers_all[i][idx, k]]
            aoas_j0 = np.r_[aoas_j0, aoa_all[i][idx, 0]]  # first foil aoa
            aoas_j1 = np.r_[aoas_j1, aoa_all[i][idx, 1]]  # second foil aoa
            betas_j0 = np.r_[betas_j0, beta_all[i][idx, 0]]  # first foil aoa
            betas_j1 = np.r_[betas_j1, beta_all[i][idx, 1]]  # second foil aoa

#            if i == 10:
#                print(lb, ub)
#                print(zdims[i][idx])
#                print()

        hax = ax.hexbin(gaps_j, staggers_j, gridsize=30, cmap='viridis',
                                extent=[10.5, -5.5, 10.5, -2.5],
                                vmin=0, vmax=50, mincnt=0,
                                linewidths=0.2)
        ax.axhline(0, color='w', lw=.5)
        ax.axvline(0, color='w', lw=.5)

        if k == 0:
            ax.set_title(r'{0} < z/h$_o$ < {1}'.format(lb, ub), fontsize='small')

        # for some reason this is not working...
        if k == 0 and j == 1:
            ax.set_xlabel('Gap (c)')
            ax.set_ylabel('Stagger (c)')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlim(10, -5)
    ax.set_ylim(10, -2)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
ax.cax.set_yticks(np.arange(0, 51, 10))
ax.cax.set_ylabel('Counts')

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('stagger vs gap z_h0'), **FIGOPT)


# %% COUNTS EMPTY

# https://stackoverflow.com/a/38940369
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(13, 6))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="3%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)
axs_flat = axs.flatten()

for k in np.arange(2):
    for j in np.arange(4):
        ax = axs[k, j]
        gaps_j = np.array([])
        staggers_j = np.array([])
        aoas_j0, aoas_j1 = np.array([]), np.array([])
        betas_j0, betas_j1 = np.array([]), np.array([])
        for i in np.arange(ntrials):
            znon = znons[i]
            lb, ub = j * .25, (j + 1) * .25
            idx = np.where((znon >= lb) & (znon < ub))[0]
            gaps_j = np.r_[gaps_j, gaps_all[i][idx, k]]
            staggers_j = np.r_[staggers_j, staggers_all[i][idx, k]]
            aoas_j0 = np.r_[aoas_j0, aoa_all[i][idx, 0]]  # first foil aoa
            aoas_j1 = np.r_[aoas_j1, aoa_all[i][idx, 1]]  # second foil aoa
            betas_j0 = np.r_[betas_j0, beta_all[i][idx, 0]]  # first foil aoa
            betas_j1 = np.r_[betas_j1, beta_all[i][idx, 1]]  # second foil aoa

        hax = ax.hexbin(gaps_j, staggers_j, gridsize=30, cmap='viridis',
                                extent=[10.5, -5.5, 10.5, -2.5],
                                vmin=0, vmax=50, mincnt=1,
                                linewidths=0.2)
        ax.axhline(0, color='r', lw=.5)
        ax.axvline(0, color='r', lw=.5)

        if k == 0:
            ax.set_title(r'{0} < z/h$_o$ < {1}'.format(lb, ub), fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

#    ax.set_xticks([-10, 5, 0, 5])
#    ax.set_yticks([-10, 5, 0])

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlim(10, -5)
    ax.set_ylim(10, -2)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
ax.cax.set_yticks(np.arange(0, 51, 10))
ax.cax.set_ylabel('Counts')

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('stagger vs gap mincnt z_h0'), **FIGOPT)


# %% ANGLE OF ATTACK

# https://stackoverflow.com/a/38940369
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(13, 6))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="3%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)
axs_flat = axs.flatten()

for k in np.arange(2):
    for j in np.arange(4):
        ax = axs[k, j]
        gaps_j = np.array([])
        staggers_j = np.array([])
        aoas_j0, aoas_j1 = np.array([]), np.array([])
        betas_j0, betas_j1 = np.array([]), np.array([])
        for i in np.arange(ntrials):
            znon = znons[i]
            lb, ub = j * .25, (j + 1) * .25
            idx = np.where((znon >= lb) & (znon < ub))[0]
            gaps_j = np.r_[gaps_j, gaps_all[i][idx, k]]
            staggers_j = np.r_[staggers_j, staggers_all[i][idx, k]]
            aoas_j0 = np.r_[aoas_j0, aoa_all[i][idx, 0]]  # first foil aoa
            aoas_j1 = np.r_[aoas_j1, aoa_all[i][idx, 1]]  # second foil aoa
            betas_j0 = np.r_[betas_j0, beta_all[i][idx, 0]]  # first foil aoa
            betas_j1 = np.r_[betas_j1, beta_all[i][idx, 1]]  # second foil aoa

        hax = ax.hexbin(gaps_j, staggers_j, aoas_j0, gridsize=30, cmap='magma',
                        extent=[10.5, -5.5, 10.5, -2.5],
                        vmin=30, vmax=90,
                        linewidths=0.2)
        ax.axhline(0, color='k', lw=.5)
        ax.axvline(0, color='k', lw=.5)

        if k == 0:
            ax.set_title(r'{0} < z/h$_o$ < {1}'.format(lb, ub), fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

#    ax.set_xticks([-10, 5, 0, 5])
#    ax.set_yticks([-10, 5, 0])

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlim(10, -5)
    ax.set_ylim(10, -2)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
ax.cax.yaxis.set_major_formatter(degree_formatter)
#ax.cax.set_yticks([-10, 0, 30, 60, 90])
ax.cax.set_yticks([30, 45, 60, 75, 90])
ax.cax.set_ylabel('Angle of attack (deg)')

#for piv_aoa in [0, 20, 30, 40, 60]:
#    # cbar.ax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)
#    ax.cax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('stagger vs gap aoa_all z_h0'), **FIGOPT)


# %% SWEEP ANGLE

# https://stackoverflow.com/a/38940369
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(13, 6))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="3%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)
axs_flat = axs.flatten()

for k in np.arange(2):
    for j in np.arange(4):
        ax = axs[k, j]
        gaps_j = np.array([])
        staggers_j = np.array([])
        aoas_j0, aoas_j1 = np.array([]), np.array([])
        betas_j0, betas_j1 = np.array([]), np.array([])
        for i in np.arange(ntrials):
            znon = znons[i]
            lb, ub = j * .25, (j + 1) * .25
            idx = np.where((znon >= lb) & (znon < ub))[0]
            gaps_j = np.r_[gaps_j, gaps_all[i][idx, k]]
            staggers_j = np.r_[staggers_j, staggers_all[i][idx, k]]
            aoas_j0 = np.r_[aoas_j0, aoa_all[i][idx, 0]]  # first foil aoa
            aoas_j1 = np.r_[aoas_j1, aoa_all[i][idx, 1]]  # second foil aoa
            betas_j0 = np.r_[betas_j0, beta_all[i][idx, 0]]  # first foil aoa
            betas_j1 = np.r_[betas_j1, beta_all[i][idx, 1]]  # second foil aoa

        hax = ax.hexbin(gaps_j, staggers_j, betas_j0, gridsize=30, cmap='coolwarm',
                        extent=[10.5, -5.5, 10.5, -2.5],
                        vmin=-90, vmax=90)
        ax.axhline(0, color='k', lw=.5)
        ax.axvline(0, color='k', lw=.5)

        if k == 0:
            ax.set_title(r'{0} < z/h$_o$ < {1}'.format(lb, ub), fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

#    ax.set_xticks([-10, 5, 0, 5])
#    ax.set_yticks([-10, 5, 0])

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlim(10, -5)
    ax.set_ylim(10, -2)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
ax.cax.yaxis.set_major_formatter(degree_formatter)
ax.cax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax.cax.set_ylabel('Sweep angle (deg)')

fig.set_tight_layout(False)

fig.canvas.draw()


# %% SWEEP ANGLE ABS

# https://stackoverflow.com/a/38940369
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(13, 6))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="3%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)
axs_flat = axs.flatten()

for k in np.arange(2):
    for j in np.arange(4):
        ax = axs[k, j]
        gaps_j = np.array([])
        staggers_j = np.array([])
        aoas_j0, aoas_j1 = np.array([]), np.array([])
        betas_j0, betas_j1 = np.array([]), np.array([])
        for i in np.arange(ntrials):
            znon = znons[i]
            lb, ub = j * .25, (j + 1) * .25
            idx = np.where((znon >= lb) & (znon < ub))[0]
            gaps_j = np.r_[gaps_j, gaps_all[i][idx, k]]
            staggers_j = np.r_[staggers_j, staggers_all[i][idx, k]]
            aoas_j0 = np.r_[aoas_j0, aoa_all[i][idx, 0]]  # first foil aoa
            aoas_j1 = np.r_[aoas_j1, aoa_all[i][idx, 1]]  # second foil aoa
            betas_j0 = np.r_[betas_j0, beta_all[i][idx, 0]]  # first foil aoa
            betas_j1 = np.r_[betas_j1, beta_all[i][idx, 1]]  # second foil aoa

        hax = ax.hexbin(gaps_j, staggers_j, np.abs(betas_j0), gridsize=30, cmap='inferno',
                        extent=[10.5, -5.5, 10.5, -2.5],
                        vmin=0, vmax=90)
        ax.axhline(0, color='k', lw=.5)
        ax.axvline(0, color='k', lw=.5)

        if k == 0:
            ax.set_title(r'{0} < z/h$_o$ < {1}'.format(lb, ub), fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

#    ax.set_xticks([-10, 5, 0, 5])
#    ax.set_yticks([-10, 5, 0])

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlim(10, -5)
    ax.set_ylim(10, -2)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
ax.cax.yaxis.set_major_formatter(degree_formatter)
ax.cax.set_yticks([0, 30, 60, 90])
ax.cax.set_ylabel('Sweep angle (deg)')

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

fig.savefig(FIG.format('stagger vs gap beta_all z_h0'), **FIGOPT)


# %% ANGLE OF ATTACK --- JOINT BY FRAC THROUGH GLIDE

# https://stackoverflow.com/a/38940369
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(13, 6))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="3%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)
axs_flat = axs.flatten()

for k in np.arange(2):
    for j in np.arange(4):
        ax = axs[k, j]
        gaps_j = np.array([])
        staggers_j = np.array([])
        aoas_j0, aoas_j1 = np.array([]), np.array([])
        betas_j0, betas_j1 = np.array([]), np.array([])
        for i in np.arange(ntrials):
            znon = znons[i]
            lb, ub = j * .25, (j + 1) * .25
            idx = np.where((znon >= lb) & (znon < ub))[0]
            gaps_j = np.r_[gaps_j, gaps_all[i][idx, k]]
            staggers_j = np.r_[staggers_j, staggers_all[i][idx, k]]
            aoas_j0 = np.r_[aoas_j0, aoa_all[i][idx, k]]  # first foil aoa
            aoas_j1 = np.r_[aoas_j1, aoa_all[i][idx, k + 1]]  # second foil aoa
            betas_j0 = np.r_[betas_j0, beta_all[i][idx, k]]  # first foil aoa
            betas_j1 = np.r_[betas_j1, beta_all[i][idx, k + 1]]  # second foil aoa

        hax = ax.hexbin(aoas_j0, aoas_j1, gridsize=30, cmap='magma',
                        extent=[-10, 90, -10, 90],
                        vmin=0, mincnt=0,
                        linewidths=0.2)
#        ax.axhline(0, color='k', lw=.5)
#        ax.axvline(0, color='k', lw=.5)

        if k == 0:
            ax.set_title(r'{0} < z/h$_o$ < {1}'.format(lb, ub), fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([0, 30, 60, 90])

#    ax.invert_xaxis()
#    ax.invert_yaxis()

    ax.set_xlim(-10, 90)
    ax.set_ylim(-10, 90)

    ax.xaxis.set_major_formatter(degree_formatter)
    ax.yaxis.set_major_formatter(degree_formatter)

    ax.grid(True)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
#ax.cax.set_yticks([-10, 0, 30, 60, 90])
#ax.cax.set_yticks([30, 45, 60, 75, 90])
ax.cax.set_ylabel('Counts', fontsize='small')

#for piv_aoa in [0, 20, 30, 40, 60]:
#    # cbar.ax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)
#    ax.cax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('stagger vs gap aoa_all z_h0'), **FIGOPT)


# %% SWEEP ANGLE ABS --- JOINT BY FRAC THROUGH GLIDE

# https://stackoverflow.com/a/38940369
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(13, 6))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 4),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="3%",
                 cbar_pad=0.25,
                 )

axs = np.array(grid.axes_row)
axs_flat = axs.flatten()

for k in np.arange(2):
    for j in np.arange(4):
        ax = axs[k, j]
        gaps_j = np.array([])
        staggers_j = np.array([])
        aoas_j0, aoas_j1 = np.array([]), np.array([])
        betas_j0, betas_j1 = np.array([]), np.array([])
        for i in np.arange(ntrials):
            znon = znons[i]
            lb, ub = j * .25, (j + 1) * .25
            idx = np.where((znon >= lb) & (znon < ub))[0]
            gaps_j = np.r_[gaps_j, gaps_all[i][idx, k]]
            staggers_j = np.r_[staggers_j, staggers_all[i][idx, k]]
            aoas_j0 = np.r_[aoas_j0, aoa_all[i][idx, k]]  # first foil aoa
            aoas_j1 = np.r_[aoas_j1, aoa_all[i][idx, k + 1]]  # second foil aoa
            betas_j0 = np.r_[betas_j0, beta_all[i][idx, k]]  # first foil aoa
            betas_j1 = np.r_[betas_j1, beta_all[i][idx, k + 1]]  # second foil aoa

        hax = ax.hexbin(np.abs(betas_j0), np.abs(betas_j1),
                        gridsize=30, cmap='inferno',
                        extent=[0, 90, 0, 90],
                        vmin=0, mincnt=0,
                        linewidths=0.2)
#        ax.axhline(0, color='k', lw=.5)
#        ax.axvline(0, color='k', lw=.5)

        if k == 0:
            # txt = '$\overline{\mathrm{z}}$'
            txt = '$\\bar \mathrm{z}$'
            ax.set_title(r'{0} < {1} < {2}'.format(lb, txt, ub), fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for ax in axs.flatten():
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([0, 30, 60, 90])

#    ax.invert_xaxis()
#    ax.invert_yaxis()

    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)

    ax.xaxis.set_major_formatter(degree_formatter)
    ax.yaxis.set_major_formatter(degree_formatter)

    ax.grid(True)

ax.cax.colorbar(hax)
ax.cax.toggle_label(True)
#ax.cax.set_yticks([-10, 0, 30, 60, 90])
#ax.cax.set_yticks([30, 45, 60, 75, 90])
ax.cax.set_ylabel('Counts', fontsize='small')

#for piv_aoa in [0, 20, 30, 40, 60]:
#    # cbar.ax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)
#    ax.cax.axhline((piv_aoa + 10) / 100, color='w', lw=1.5)

for c in ax.cax.collections:
    c.set_edgecolor("face")

fig.set_tight_layout(False)

fig.canvas.draw()

#fig.savefig(FIG.format('stagger vs gap beta_all z_h0'), **FIGOPT)


# %% Gap and stagger average for each trial




# %% Snake IDs

fnames = ret_fnames()

ids = []
for fname in fnames:
    id_i, _ = trial_info(fname)
    ids.append(id_i)
ids = np.array(ids)

idxs = {}
fnames = ret_fnames()
for sn_id in ids:
    idx = []
    for i, fname in enumerate(fnames):
        snake_id, trial_id = trial_info(fname)

        if sn_id == snake_id:
            idx.append(i)

    idxs[sn_id] = idx


# %%

snake_id = 81
g_sn_i = np.array(gaps_all)[idxs[snake_id]]
s_sn_i = np.array(staggers_all)[idxs[snake_id]]


g_sn, s_sn = np.array([]), np.array([])
g_sn_1, s_sn_1 = np.array([]), np.array([])
g_sn_2, s_sn_2 = np.array([]), np.array([])
for i in np.arange(len(g_sn_i)):
    g_sn = np.r_[g_sn, g_sn_i[i].flatten()]
    s_sn = np.r_[s_sn, s_sn_i[i].flatten()]
    g_sn_1 = np.r_[g_sn_1, g_sn_i[i][:, 0].flatten()]
    s_sn_1 = np.r_[s_sn_1, s_sn_i[i][:, 0].flatten()]
    g_sn_2 = np.r_[g_sn_2, g_sn_i[i][:, 1].flatten()]
    s_sn_2 = np.r_[s_sn_2, s_sn_i[i][:, 1].flatten()]


# %% Gap and stagger hexbin - individual snake

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots()
ax.axvline(0, color='r', lw=1)
ax.axhline(0, color='r', lw=1)
ax.plot(0, 0, 'ro', ms=5)
#cax = ax.hexbin(g_sn, s_sn, gridsize=100, cmap='viridis', marginals=True)
cax = ax.hexbin(g_sn_1, s_sn_1, gridsize=50, cmap='viridis',
                extent=[10.5, -5.5, 10.5, -2.5])
#cax = ax.hexbin(g_sn, s_sn, gridsize=100, cmap='viridis', marginals=True)

cbar = fig.colorbar(cax, ax=ax, shrink=.75)

#ax.plot(g_sn, s_sn, 'r+', mew=1)

#for gp in gaps_piv:
#    for sp in staggers_piv:
#        ax.plot(gp, sp, '+', color='w', mew=1.5)

ax.set_xlabel('Gap')
ax.set_ylabel('Stagger')

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)


# %%

def rot_foil(foil, deg):
    ang = np.deg2rad(deg)

    Rth = np.array([[np.cos(ang), -np.sin(ang)],
                    [np.sin(ang),  np.cos(ang)]])

    return np.dot(Rth, rfoil.T).T

# load in the airfoil shape
rfoil = np.genfromtxt('../Data/Foil/snake0.004.bdy.txt', skip_header=1)
rfoil = rfoil - rfoil.mean(axis=0)
rfoil -= rfoil.mean(axis=0)
rfoil /= np.ptp(rfoil[:, 0])
rfoil_rot = rot_foil(rfoil, 78)


if True:
    fig, ax = plt.subplots()
    ax.axvline(-.5, color='gray', lw=1)
    ax.axvline(.5, color='gray', lw=1)
#    ax.plot(rfoil[:, 0], rfoil[:, 1])
    #ax.plot(rfoil_rot[:, 0], rfoil_rot[:, 1])
    f1, = ax.fill(rfoil_rot[:, 0], rfoil_rot[:, 1], alpha=.5)
    ax.set_aspect('equal', adjustable='box')
    sns.despine()


# %% Select out trial 507_81

for i in np.arange(len(fnames)):
    fname  = fnames[i].split('.npz')[0][-6:]
#    if fname == '507_81':
#        print(i)
    if fname == '807_95':
        print(i)
    print(fname)


# %%

#j = 18
j = 39

gj = gaps_all[j]
sj = staggers_all[j]
aoaj = aoa_all[j]
ntime = len(gj)


fig, ax = plt.subplots()
#ax.axhline(0, color='gray', lw=1)
#ax.axvline(0, color='gray', lw=1)

#ax.set_ylim(-120, 120)
#ax.set_yticks([-120, -90, -60, -30, 0, 30, 60, 90, 120])

#ax.set_xlim(-6, 6)
#ax.set_ylim(-6, 6)
ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
sns.despine()
fig.set_tight_layout(True)


i = 0
k = 0

offset_1 = np.r_[gj[i, k], sj[i, k]]
foil_0 = rot_foil(rfoil, 180+aoaj[i, k])
foil_1 = rot_foil(rfoil, 180+aoaj[i, k + 1]) + offset_1

f0, = ax.fill(foil_0[:, 0], foil_0[:, 1], alpha=1, color=bmap[k])
f1, = ax.fill(foil_1[:, 0], foil_1[:, 1], alpha=1, color=bmap[k + 1])

line_1, = ax.plot(gj[:i + 1, k], sj[:i + 1, k], color=bmap[k + 1])


title_text = ax.set_title('{0:.2f}'.format(i / ntime))

ax.set_xlabel('gap (c)')
ax.set_ylabel('stager (c)')

def animate(i):

    offset_1 = np.r_[gj[i, k], sj[i, k]]
    foil_0 = rot_foil(rfoil, 180+aoaj[i, k])
    foil_1 = rot_foil(rfoil, 180+aoaj[i, k + 1]) + offset_1

    f0.set_xy(foil_0)
    f1.set_xy(foil_1)
    line_1.set_xdata(gj[:i + 1, k])
    line_1.set_ydata(sj[:i + 1, k])
    title_text.set_text('{0:.2f}'.format(i / ntime))

#    return theta_line, phi_line,
    return f0, f1, line_1, title_text


from matplotlib.animation import FuncAnimation

slowed = 10
dt = 1 / 179.
ani = FuncAnimation(fig, animate, frames=ntime,
                    interval=dt * 1000 * slowed,  # draw a frame every x ms
                    repeat=2, blit=True)#, init_func=init)


save_movie = False
if save_movie:
    #ani.save('../Movies/s_serp3d/5X aerial serpnoid curve.mp4',
#         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])

    movie_name = '../Movies/s_wings/{0}_{1} pair_{2} 10x bending.mp4'
    movie_name = movie_name.format('507', '81', k)
    ani.save(movie_name,
             extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])
#    ani.save(movie_name,
#             codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])


# %%

fig, ax = plt.subplots()
ax.plot(gj[:, 0], sj[:, 0])
ax.set_aspect('equal', adjustable='box')


# %% Average number of wings

idx = np.argsort(nwing_avg_all)

fig, ax = plt.subplots()
ax.plot(nwing_avg_all[idx])
#ax.plot(nwing_avg_all[idx] + nwing_std_all[idx])
#ax.plot(nwing_avg_all[idx] - nwing_std_all[idx])
ax.plot(nwing_max_all[idx])
ax.set_xlabel('Trial sorted by average number of wings')
ax.set_ylabel('Average number of wings')
ax.set_yticks(np.arange(6))
ax.set_ylim(0, 5)
sns.despine()


# %% Gap and stagger hexbin

# what Farid measured
gaps_piv = -np.r_[2, 4, 6, 8]
staggers_piv = -np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots()
ax.axvline(0, color='r', lw=1)
ax.axhline(0, color='r', lw=1)
ax.plot(0, 0, 'ro', ms=5)
cax = ax.hexbin(g, s, gridsize=100, cmap='viridis', marginals=True)

cbar = fig.colorbar(cax, ax=ax, shrink=.75)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='w', mew=1.5)

ax.set_xlabel('Gap')
ax.set_ylabel('Stagger')

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)
fig.set_tight_layout(True)


# %% Gap and stagger hexbin + kde plot

fig, ax = plt.subplots()
ax.axvline(0, color='r', lw=1)
ax.axhline(0, color='r', lw=1)
ax.plot(0, 0, 'ro', ms=5)

ax.hexbin(g, s, gridsize=100, cmap='viridis', marginals=True)

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
#sns.kdeplot(g, s, cmap=cmap, n_levels=60, shade=True, ax=ax)
sns.kdeplot(g[~np.isnan(g)], s[~np.isnan(s)], cmap=cmap, n_levels=10, ax=ax)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='w', mew=1.5)

ax.set_xlabel('Gap')
ax.set_ylabel('Stagger')

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)
fig.set_tight_layout(True)


# %%


# %% Format with degrees

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


# %% AoA_up and AoA_down hexbin --- first set

alphas_piv = np.r_[0, 20, 40, 60]
alpha0_piv = 30

fig, ax = plt.subplots()
ax.axvline(np.nanmean(aoa_1), color='gray', lw=1)
ax.axhline(np.nanmean(aoa_2), color='gray', lw=1)

cax = ax.hexbin(aoa_1, aoa_2, gridsize=50, cmap='magma',
                extent=[0, 90, 0, 90], linewidths=0.2)

cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='small')

for al in alphas_piv:
    ax.plot(alpha0_piv, al, '+', color='w', mew=1.5)
    ax.plot(al, alpha0_piv, '+', color='w', mew=1.5)

ax.set_xlim(0, 90)
ax.set_ylim(0, 90)

ax.set_xticks(np.arange(0, 91, 15))
ax.set_yticks(np.arange(0, 91, 15))

ax.xaxis.set_major_formatter(degree_formatter)
ax.yaxis.set_major_formatter(degree_formatter)

ax.set_xlabel('Upstream airfoil angle of attack')
ax.set_ylabel('Downstream airfoil angle of attack')

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)
fig.set_tight_layout(True)

fig.savefig(FIG.format('aoa_2 vs aoa_1'), **FIGOPT)


# %% AoA_up and AoA_down hexbin --- second set

alphas_piv = np.r_[0, 20, 40, 60]
alpha0_piv = 30

fig, ax = plt.subplots()
ax.axvline(np.nanmean(aoa_2), color='gray', lw=1)
ax.axhline(np.nanmean(aoa_3), color='gray', lw=1)

cax = ax.hexbin(aoa_2, aoa_3, gridsize=50, cmap='magma',
                extent=[0, 90, 0, 90], linewidths=0.2)

cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='small')

for al in alphas_piv:
    ax.plot(alpha0_piv, al, '+', color='w', mew=1.5)
    ax.plot(al, alpha0_piv, '+', color='w', mew=1.5)

ax.set_xlim(0, 90)
ax.set_ylim(0, 90)

ax.set_xticks(np.arange(0, 91, 15))
ax.set_yticks(np.arange(0, 91, 15))

ax.xaxis.set_major_formatter(degree_formatter)
ax.yaxis.set_major_formatter(degree_formatter)

ax.set_xlabel('Upstream airfoil angle of attack')
ax.set_ylabel('Downstream airfoil angle of attack')

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)
fig.set_tight_layout(True)

fig.savefig(FIG.format('aoa_3 vs aoa_2'), **FIGOPT)


# %% sweep_up and sweep_down hexbin --- first set

fig, ax = plt.subplots()
#ax.axvline(np.nanmean(beta_1), color='gray', lw=1)
#ax.axhline(np.nanmean(beta_2), color='gray', lw=1)

cax = ax.hexbin(np.abs(beta_1), np.abs(beta_2), gridsize=50, cmap='inferno',
                extent=[0, 90, 0, 90], linewidths=0.2)

cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='small')

ax.set_xlim(0, 90)
ax.set_ylim(0, 90)

ax.set_xticks(np.arange(0, 91, 15))
ax.set_yticks(np.arange(0, 91, 15))

ax.xaxis.set_major_formatter(degree_formatter)
ax.yaxis.set_major_formatter(degree_formatter)

ax.set_xlabel('Upstream sweep angle')
ax.set_ylabel('Downstream sweep angle')

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)
fig.set_tight_layout(True)

fig.savefig(FIG.format('beta_2 vs beta_1'), **FIGOPT)


# %% sweep_up and sweep_down hexbin --- second set

fig, ax = plt.subplots()
#ax.axvline(np.nanmean(beta_2), color='gray', lw=1)
#ax.axhline(np.nanmean(beta_3), color='gray', lw=1)

cax = ax.hexbin(np.abs(beta_2), np.abs(beta_3), gridsize=50, cmap='inferno',
                extent=[0, 90, 0, 90], linewidths=0.2)

cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='small')

ax.set_xlim(0, 90)
ax.set_ylim(0, 90)

ax.set_xticks(np.arange(0, 91, 15))
ax.set_yticks(np.arange(0, 91, 15))

ax.xaxis.set_major_formatter(degree_formatter)
ax.yaxis.set_major_formatter(degree_formatter)

ax.set_xlabel('Upstream sweep angle')
ax.set_ylabel('Downstream sweep angle')

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)
fig.set_tight_layout(True)

fig.savefig(FIG.format('beta_3 vs beta_2'), **FIGOPT)


# %%

## %% sweep_up and sweep_down hexbin --- first set
#
#fig, ax = plt.subplots()
#ax.axvline(np.nanmean(beta_1), color='gray', lw=1)
#ax.axhline(np.nanmean(beta_2), color='gray', lw=1)
#
#cax = ax.hexbin(np.abs(beta_1), np.abs(beta_2), gridsize=50, cmap='inferno',
#                extent=[-90, 90, -90, 90])
#
#cbar = fig.colorbar(cax, ax=ax, shrink=.75)
#
#ax.set_xlim(-90, 90)
#ax.set_ylim(-90, 90)
#
#ax.set_xticks(np.arange(-90, 91, 30))
#ax.set_yticks(np.arange(-90, 91, 30))
#
#ax.xaxis.set_major_formatter(degree_formatter)
#ax.yaxis.set_major_formatter(degree_formatter)
#
#ax.set_xlabel('Upstream sweep angle')
#ax.set_ylabel('Downstream sweep angle')
#
#ax.set_aspect('equal', adjustable='box')
#sns.despine(ax=ax)
#fig.set_tight_layout(True)


# %%

fig, ax = plt.subplots()
ax.plot(aoa_1, aoa_2, 'o', alpha=.2)
ax.set_aspect('equal', adjustable='box')
sns.despine()


# %%

fig, ax = plt.subplots()
ax.plot(beta_1, beta_2, 'o', alpha=.2)
ax.set_aspect('equal', adjustable='box')
sns.despine()


# %%

fig, ax = plt.subplots()
ax.plot(g, s, 'o', alpha=.2)
#ax.plot(gaps[:, 2], staggers[:, 2], 'r+', mew=1)
#ax.plot(gaps[:, 1], staggers[:, 1], 'y+', mew=1)
#ax.plot(gaps[:, 0], staggers[:, 0], 'w+', mew=1)
#ax.set_xlim(-12, 2)
#ax.set_ylim(-12, 2)
ax.set_aspect('equal', adjustable='box')
sns.despine()


# %%

# %% Socha Socha (2010) gap and stagger

# %%

import pandas as pd
from scipy.interpolate import UnivariateSpline


def load_5p_csv(fname):
    """Load in data copied from xls file from Jake.
    """

    d = np.genfromtxt(fname, skip_header=2, delimiter=',')
    n = d[:, 0]
    t = d[:, 1]
    d = d[:, 2:]

    n = np.arange(len(d))

    return d, n, t


def load_3p_csv(fname):
    """Load in the three-point marked snake files.
    """

    df = pd.read_csv(fname, index_col=1, skiprows=2, na_values=b'\xe2\x80\x94')
    d = df.drop(u'ID', 1).astype(np.float)
    d = d.values
    t = df.index.values
    n = np.arange(len(t))

    return d, n, t


def filler(tvec, data, k, s):
    """Fill in/extrapolate missing data based on linear interpolation.
    """

    good = np.where(~np.isnan(data[:, 0]))[0]
    x = UnivariateSpline(tvec[good], data[good, 0], k=k, s=s)
    y = UnivariateSpline(tvec[good], data[good, 1], k=k, s=s)
    z = UnivariateSpline(tvec[good], data[good, 2], k=k, s=s)

    return np.c_[x(tvec), y(tvec), z(tvec)]


def fill_raw(tvec, data, k=1, s=0):
    """Fill in the gaps in the raw data.
    """

    npts = data.shape[1] // 3
    datanew = np.zeros(data.shape)
    for i in range(npts):
        part = data[:, i * npts:(i + 1) * npts]
        datanew[:, i * npts:(i + 1) * npts] = filler(tvec, part, k, s)
    return datanew


def fix_5p(d, center=False):
    # coordinates as (x, y, z)
    head = d[:, [0, 1, 2]]
    qrtr = d[:, [3, 4, 5]]
    mids = d[:, [6, 7, 8]]
    thrq = d[:, [9, 10, 11]]
    vent = d[:, [12, 13, 14]]

    ntime = head.shape[0]
    nmark = 5
    data = np.zeros((ntime, nmark, 3))
    data[:, 0] = head
    data[:, 1] = qrtr
    data[:, 2] = mids
    data[:, 3] = thrq
    data[:, 4] = vent

    # CoM location using method from Socha (2010)
    com = (head + vent + 2 * (qrtr + mids + thrq)) / 8

    if center:
        head -= com[0]
        qrtr -= com[0]
        mids -= com[0]
        thrq -= com[0]
        vent -= com[0]
        com -= com[0]

    return data, com


def fix_3p(d, center=False):
    """Get out the different components from the x, y, z coordinates.
    """
    # coordinates as (x, y, z)
    head = d[:, [0, 1, 2]]
    mids = d[:, [3, 4, 5]]
    vent = d[:, [6, 7, 8]]

    ntime = head.shape[0]
    nmark = 3
    data = np.zeros((ntime, nmark, 3))
    data[:, 0] = head
    data[:, 1] = mids
    data[:, 2] = vent

    # CoM location using method from Socha (2010)
    com = (head + vent + 2 * mids) / 4

    if center:
        sidx = np.where((~np.isnan(d) * 1).sum(axis=1) == d.shape[1])[0][0]
        head -= com[sidx]
        mids -= com[sidx]
        vent -= com[sidx]
        com -= com[sidx]

    return data, com


def fix_3p_old(d, center=False):
    """Get out the different components from the x, y, z coordinates.
    """

    # coordinates as (x, y, z)
    head = d[:, [0, 1, 2]]
    mids = d[:, [3, 4, 5]]
    vent = d[:, [6, 7, 8]]

    # CoM location using method from Socha (2010)
    com = (head + vent + 2 * mids) / 4

    if center:
        sidx = np.where((~np.isnan(d) * 1).sum(axis=1) == d.shape[1])[0][0]
        head -= com[sidx]
        mids -= com[sidx]
        vent -= com[sidx]
        com -= com[sidx]

    x = np.c_[head[:, 0], mids[:, 0], vent[:, 0]]
    y = np.c_[head[:, 1], mids[:, 1], vent[:, 1]]
    z = np.c_[head[:, 2], mids[:, 2], vent[:, 2]]

    return x, y, z, head, mids, vent, com


def fix_5p_old(d, center=False):
    # coordinates as (x, y, z)
    head = d[:, [0, 1, 2]]
    qrtr = d[:, [3, 4, 5]]
    mids = d[:, [6, 7, 8]]
    thrq = d[:, [9, 10, 11]]
    vent = d[:, [12, 13, 14]]

    # CoM location using method from Socha (2010)
    com = (head + vent + 2 * (qrtr + mids + thrq)) / 8

    if center:
        head -= com[0]
        qrtr -= com[0]
        mids -= com[0]
        thrq -= com[0]
        vent -= com[0]
        com -= com[0]

    x = np.c_[head[:, 0], qrtr[:, 0], mids[:, 0], thrq[:, 0], vent[:, 0]]
    y = np.c_[head[:, 1], qrtr[:, 1], mids[:, 1], thrq[:, 1], vent[:, 1]]
    z = np.c_[head[:, 2], qrtr[:, 2], mids[:, 2], thrq[:, 2], vent[:, 2]]

    return x, y, z, head, qrtr, mids, thrq, vent, com


def rotate(xy, th):
    """Rotate shape given coordinate and rotation in degrees.
    """
    Rth = np.array([[np.cos(th), -np.sin(th), 0],
                    [np.sin(th),  np.cos(th), 0],
                    [0, 0, 1]])
    return np.dot(Rth, xy.T).T


def calc_com(data):
    ds = data.copy()
    npts = ds.shape[0]
    nmark = ds.shape[1] // 3

    if nmark == 3:
        weights = np.r_[1, 2, 1]
    elif nmark == 5:
        weights = np.r_[1, 2, 2, 2, 1]
    norm = weights.sum()

    com = np.zeros((npts, 3))
    for j in range(nmark):
        start, stop = j * nmark, (j + 1) * nmark
        com += ds[:, start:stop] * weights[j]

    return com / norm


def center_data(data, com):
    ds = data.copy()
    nmark = ds.shape[1] // 3
    for j in range(nmark):
        start, stop = j * nmark, (j + 1) * nmark
        ds[:, start:stop] -= com[0]
    com -= com[0]

    return ds, com


def straighten_trajectory(data):
    """Align the trajectory so we are in the 'glide-polar'
    2D projection.
    """

    # non-destructive updates
    ds = data.copy()
    npts = data.shape[0]
    nmark = data.shape[1] // 3

    cs = calc_com(data)
    ds, cs = center_data(ds, cs)

    for i in range(npts):
        th = np.arctan2(cs[i, 0], cs[i, 1])
        cs[i:] = rotate(cs[i:], th)

        for j in range(nmark):
            start, stop = j * nmark, (j + 1) * nmark
            ds[i:, start:stop] = rotate(ds[i:, start:stop], th)

    return ds, cs


def rotate_to_flow_pts(gamma, R_Sc, dRo_S):
    """Rotate from inertial to flow frame, such that CoM velocity
    is in the forward y direction. Also check that the rotation
    is correct from I2F and S2F.
    """

    # what we want to return
    ntime = R_Sc.shape[0]
    R_Fc = np.zeros_like(R_Sc)
    C_S2F = np.zeros((ntime, 3, 3))
    dRo_F = np.zeros_like(dRo_S)

    for i in np.arange(ntime):
        gamma_i = -gamma[i]

        # rotate about the x axis
        C_S2F[i] = np.array([[1, 0, 0],
                             [0, np.cos(gamma_i), np.sin(gamma_i)],
                             [0, -np.sin(gamma_i), np.cos(gamma_i)]])

        R_Fc[i] = np.dot(C_S2F[i], R_Sc[i].T).T
        dRo_F[i] = np.dot(C_S2F[i], dRo_S[i].T).T

    return R_Fc, dRo_F


# %% Run through all trials, smooth, and make a bunch of plots

fnames5 = [1099, 1102, 1105, 1108, 1109, 1112, 1114, 1117]
fnames3 = [735, 739, 776, 778, 779, 815, 847, 850, 892]

masses5 = np.r_[25.5, 42, 42, 25.5, 42, 25.5, 42, 25.5] / 1000
svl5 = np.r_[60.3, 74, 74, 60.3, 74, 60.3, 74, 60.3] / 100
chord5 = .01 * (.99 * np.log(svl5 * 100) - 1.21)  # m
sn_id5 = np.r_[1, 2, 2, 1, 2, 1, 2, 1]

masses3 = np.r_[83, 41, 27, 11, 3, 63, 26, 11, 16] / 1000
svl3 = np.r_[85, 71, 63, 47, 31, 83, 62, 47, 54] / 100
chord3 = .01 * (.99 * np.log(svl3 * 100) - 1.21)  # m
sn_id3 = np.arange(len(masses3))

# there is a lot of missing start data from the last trail
fnames3 = fnames3[:-1]
masses3 = masses3[:-1]
svl3 = svl3[:-1]

save_base = '../Data/Socha2010/{}.csv'

data5, data3 = {}, {}


# %% 3 point snake

for i, fname in enumerate(fnames3):
    data_i, pts, tvec = load_3p_csv(save_base.format(fname))

    data_if = fill_raw(tvec, data_i)
    # ds, cs = straighten_trajectory(dataf)
    pr_I, Ro_I = fix_3p(data_if, center=False)
    dt = np.diff(tvec).mean()
    mass = masses3[i]
    SVL = svl3[i]
    sn_id = sn_id3[i]

    head_vent = pr_I[:, 0] - pr_I[:, 2]
    dist_hv = np.linalg.norm(head_vent, axis=1)
    sinuosity = dist_hv / svl3[i]

    d = dict(pr_I=pr_I, Ro_I=Ro_I, pts=pts, times=tvec, dt=dt, fname=fname,
             mass=mass, SVL=SVL, sn_id=sn_id, sinuosity=sinuosity)

    data3[fname] = d


# %% 5 point snake

# shift the Z component up so it is always positive
Zo = 8  # m; trial 1105 Z[-1] = -7.408m

for i, fname in enumerate(fnames5):
    data_i, pts, tvec = load_5p_csv(save_base.format(fname))
    pr_S, Ro_S = fix_5p(data_i, center=False)
    dt = np.diff(tvec).mean()
    fs = 1 / dt  # 30 Hz
    mass = masses5[i]
    SVL = svl5[i]
    chord = chord5[i]
    sn_id = sn_id5[i]
    mg = mass * 9.81

    # swap x and y to match Cube convention
    x_tmp, y_tmp = pr_S[:, :, 0].copy(), pr_S[:, :, 1].copy()
    xo_tmp, yo_tmp = Ro_S[:, 0].copy(), Ro_S[:, 1].copy()
    pr_S[:, :, 1] = x_tmp
    pr_S[:, :, 0] = y_tmp
    Ro_S[:, 1] = xo_tmp
    Ro_S[:, 0] = yo_tmp

    # move up the z
    pr_S[:, :, 2] += Zo
    Ro_S[:, 2] += Zo

    # position relative to CoM
    pr_Sc = np.zeros_like(pr_S)
    for j in np.arange(5):
        pr_Sc[:, j] = pr_S[:, j] - Ro_S

    # center of mass velocity
    dRo_S, ddRo_S = findiff(Ro_S, dt)

    aa = dRo_S.copy()

    # glide angle
    gamma = -np.arctan2(dRo_S[:, 2], dRo_S[:, 1])

    # rotate to flow frame
    pr_Fc, dRo_F = rotate_to_flow_pts(gamma, pr_Sc, dRo_S)


    # sinuosity
    head_vent = pr_S[:, 0] - pr_S[:, 4]
    dist_hv = np.linalg.norm(head_vent, axis=1)
    sinuosity = dist_hv / svl5[i]

    key = '{}_{}'.format(sn_id, fname)
    d = dict(pr_S=pr_S, Ro_S=Ro_S, pts=pts, times=tvec, dt=dt, fname=fname,
             mass=mass, SVL=SVL, chord=chord, sn_id=sn_id, sinuosity=sinuosity,
             pr_Sc=pr_Sc, dRo_S=dRo_S, ddRo_S=ddRo_S, mg=mg,
             gamma=gamma, pr_Fc=pr_Fc, dRo_F=dRo_F)

    data5[key] = d


# %% CoM side view

fig, ax = plt.subplots()

for key in sorted(data5.keys()):
    d = data5[key]
    ax.plot(d['Ro_S'][:, 1], d['Ro_S'][:, 2], label=key)

ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box')
sns.despine()


# %% Glide angle

fig, ax = plt.subplots()

for key in sorted(data5.keys()):
    d = data5[key]
    ax.plot(d['times'], np.rad2deg(d['gamma']), label=key)

ax.set_ylim(0, 90)
ax.legend(loc='best')
sns.despine()


# %% VPD

fig, ax = plt.subplots()

for key in sorted(data5.keys()):
    d = data5[key]
    ax.plot(d['dRo_S'][:, 1], d['dRo_S'][:, 2], label=key)

ax.set_ylim(-10, 0)
ax.set_xlim(0, 10)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.legend(loc='best')
ax.set_aspect('equal', adjustable='box')
sns.despine(bottom=True, top=False)


# %% CoM velocity

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False)

for key in sorted(data5.keys()):
    d = data5[key]
    ax1.plot(d['times'], d['dRo_S'][:, 0], label=key)
    ax2.plot(d['times'], d['dRo_S'][:, 1], label=key)
    ax3.plot(d['times'], d['dRo_S'][:, 2], label=key)

ax.legend(loc='best')
#ax.set_aspect('equal', adjustable='box')
sns.despine()


# %% CoM acceleration

fig, ax = plt.subplots()

for key in sorted(data5.keys()):
    d = data5[key]
    ax.plot(d['ddRo_S'] / mg, label=key)

ax.legend(loc='best')
#ax.set_aspect('equal', adjustable='box')
sns.despine()


# %%

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

for key in sorted(data5.keys()):
    d = data5[key]
    ax1.plot(d['times'], d['sinuosity'], label=key)

for key in sorted(data3.keys()):
    d = data3[key]
    ax2.plot(d['times'], d['sinuosity'], label=key)

ax1.legend(loc='best', fontsize='xx-small')
ax2.legend(loc='best', fontsize='xx-small')
ax1.set_ylim(0, 1)
sns.despine()


# %%

#keys = [1105, 1108, 1109, 1112, 1114, 1099, 1117, 1102]
keys = ['1_1099', '1_1108', '1_1112', '1_1117', '2_1102', '2_1105',
        '2_1109', '2_1114']
key = keys[1]

# 7

d = data5[key]


X, Y, Z = d['Ro_S'].T
X = np.arange(len(X))
X = Y
I = np.arange(len(X))
xx = Y

#TODO check the peak here
chord = .03 * d['SVL']

pr_Sc = d['pr_Sc'].copy()
pr_Sc = d['pr_Sc'].copy() / d['SVL']
pr_Sc = d['pr_Sc'].copy() / chord

pr_Sc_x = pr_Sc[:, :, 0]
pr_Sc_y = pr_Sc[:, :, 1]
pr_Sc_z = pr_Sc[:, :, 2]


pr_Fc = d['pr_Fc'].copy()
pr_Fc_x = pr_Fc[:, :, 0]
pr_Fc_y = pr_Fc[:, :, 1]
pr_Fc_z = pr_Fc[:, :, 2]

# where the time horizontal component of the time series intersect
# https://stackoverflow.com/a/28766902
idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1


# %% Side view of when we have intersections

fig, ax = plt.subplots()
ax.plot(Y, Z, 'k')
ax.plot(Y[idx_13], Z[idx_13], 'bo')
ax.plot(Y[idx_24], Z[idx_24], 'ro')
ax.plot(Y[idx_35], Z[idx_35], 'go')
ax.set_aspect('equal', adjustable='box')
sns.despine()


# %% Horizontal component --- flow frame

c = sns.color_palette('husl', n_colors=5)

fig, axs = plt.subplots(7, 1, sharex=True, sharey=True,
                        figsize=(7, 13))
for ax in axs:
    ax.axhline(0, color='gray', lw=1)

axs[0].plot(xx, pr_Fc_x[:, 0], 'o-', c=c[0])
axs[0].plot(xx, pr_Fc_x[:, 1], 'o-', c=c[1])
axs[0].text(0, .1, '1-2', fontsize='x-small')

axs[1].plot(xx, pr_Fc_x[:, 0], 'o-', c=c[0])
axs[1].plot(xx, pr_Fc_x[:, 2], 'o-', c=c[2])
axs[1].text(0, .1, '1-3', fontsize='x-small')
axs[1].plot(xx[idx_13], pr_Fc_x[idx_13, 0], 'ko')

axs[2].plot(xx, pr_Fc_x[:, 1], 'o-', c=c[1])
axs[2].plot(xx, pr_Fc_x[:, 2], 'o-', c=c[2])
axs[2].text(0, .1, '2-3', fontsize='x-small')

axs[3].plot(xx, pr_Fc_x[:, 1], 'o-', c=c[1])
axs[3].plot(xx, pr_Fc_x[:, 3], 'o-', c=c[3])
axs[3].text(0, .1, '2-4', fontsize='x-small')
axs[3].plot(xx[idx_24], pr_Fc_x[idx_24, 1], 'ko')

axs[4].plot(xx, pr_Fc_x[:, 2], 'o-', c=c[2])
axs[4].plot(xx, pr_Fc_x[:, 3], 'o-', c=c[3])
axs[4].text(0, .1, '3-4', fontsize='x-small')

axs[5].plot(xx, pr_Fc_x[:, 2], 'o-', c=c[2])
axs[5].plot(xx, pr_Fc_x[:, 4], 'o-', c=c[4])
axs[5].text(0, .1, '3-5', fontsize='x-small')
axs[5].plot(xx[idx_35], pr_Fc_x[idx_35, 2], 'ko')

axs[6].plot(xx, pr_Fc_x[:, 3], 'o-', c=c[3])
axs[6].plot(xx, pr_Fc_x[:, 4], 'o-', c=c[4])
axs[6].text(0, .1, '4-5', fontsize='x-small')

sns.despine()


# %% Vertical component --- flow frame

fig, axs = plt.subplots(7, 1, sharex=True, sharey=True,
                        figsize=(7, 13))
for ax in axs:
    ax.axhline(0, color='gray', lw=1)

axs[0].plot(X, pr_Fc_z[:, 0], 'o-', c=c[0])
axs[0].plot(X, pr_Fc_z[:, 1], 'o-', c=c[1])
axs[0].text(0, .1, '1-2', fontsize='x-small')

axs[1].plot(X, pr_Fc_z[:, 0], 'o-', c=c[0])
axs[1].plot(X, pr_Fc_z[:, 2], 'o-', c=c[2])
axs[1].text(0, .1, '1-3', fontsize='x-small')
axs[1].plot(X[idx_13], pr_Fc_z[idx_13, 0], 'ko')

axs[2].plot(X, pr_Fc_z[:, 1], 'o-', c=c[1])
axs[2].plot(X, pr_Fc_z[:, 2], 'o-', c=c[2])
axs[2].text(0, .1, '2-3', fontsize='x-small')

axs[3].plot(X, pr_Fc_z[:, 1], 'o-', c=c[1])
axs[3].plot(X, pr_Fc_z[:, 3], 'o-', c=c[3])
axs[3].text(0, .1, '2-4', fontsize='x-small')
axs[3].plot(X[idx_24], pr_Fc_z[idx_24, 1], 'ko')

axs[4].plot(X, pr_Fc_z[:, 2], 'o-', c=c[2])
axs[4].plot(X, pr_Fc_z[:, 3], 'o-', c=c[3])
axs[4].text(0, .1, '3-4', fontsize='x-small')

axs[5].plot(X, pr_Fc_z[:, 2], 'o-', c=c[2])
axs[5].plot(X, pr_Fc_z[:, 4], 'o-', c=c[4])
axs[5].text(0, .1, '3-5', fontsize='x-small')
axs[5].plot(X[idx_35], pr_Fc_z[idx_35, 2], 'ko')

axs[6].plot(X, pr_Fc_z[:, 3], 'o-', c=c[3])
axs[6].plot(X, pr_Fc_z[:, 4], 'o-', c=c[4])
axs[6].text(0, .1, '4-5', fontsize='x-small')

sns.despine()


# %% Top view

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                        figsize=(12, 5))
ax1, ax2, ax3 = axs

for ax in axs:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(pr_Fc_x[idx_13].T, pr_Fc_y[idx_13].T, '-')
ax1.set_color_cycle(None)
ax1.plot(pr_Fc_x[:, [0, 2]][idx_13].T, pr_Fc_y[:, [0, 2]][idx_13].T, 'o')

ax2.plot(pr_Fc_x[idx_24].T, pr_Fc_y[idx_24].T, '-')
ax2.set_color_cycle(None)
ax2.plot(pr_Fc_x[:, [1, 3]][idx_24].T, pr_Fc_y[:, [1, 3]][idx_24].T, 'o')

ax3.plot(pr_Fc_x[idx_35].T, pr_Fc_y[idx_35].T, '-')
ax3.set_color_cycle(None)
ax3.plot(pr_Fc_x[:, [2, 4]][idx_35].T, pr_Fc_y[:, [2, 4]][idx_35].T, 'o')

ax1.set_title('1-3', fontsize='small')
ax2.set_title('2-4', fontsize='small')
ax3.set_title('3-5', fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')
sns.despine()


# %% Side view

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                        figsize=(12, 5))
ax1, ax2, ax3 = axs

for ax in axs:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax1.plot(pr_Fc_y[idx_13].T, pr_Fc_z[idx_13].T, '-')
ax1.set_color_cycle(None)
ax1.plot(pr_Fc_y[:, [0, 2]][idx_13].T, pr_Fc_z[:, [0, 2]][idx_13].T, 'o')

ax2.plot(pr_Fc_y[idx_24].T, pr_Fc_z[idx_24].T, '-')
ax2.set_color_cycle(None)
ax2.plot(pr_Fc_y[:, [1, 3]][idx_24].T, pr_Fc_z[:, [1, 3]][idx_24].T, 'o')

ax3.plot(pr_Fc_y[idx_35].T, pr_Fc_z[idx_35].T, '-')
ax3.set_color_cycle(None)
ax3.plot(pr_Fc_y[:, [2, 4]][idx_35].T, pr_Fc_z[:, [2, 4]][idx_35].T, 'o')

ax1.set_title('1-3', fontsize='small')
ax2.set_title('2-4', fontsize='small')
ax3.set_title('3-5', fontsize='small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')
sns.despine()


# %% Time series for each trial --- horizontal component

fig, axs_all = plt.subplots(len(keys), 3, sharex=True, sharey=True,
                        figsize=(12, 14))
for ax in axs_all.flatten():
    ax.axhline(0, color='gray', lw=1)
    # ax.axvline(0, color='gray', lw=1)


for i, key in enumerate(keys):
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy() / d['SVL']
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    X, Y, Z = d['Ro_S'].T
    xx = Y

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    axs = axs_all[i]
    axs[0].plot(xx, pr_Fc_x[:, 0], 'o-', c=c[0])
    axs[0].plot(xx, pr_Fc_x[:, 2], 'o-', c=c[2])
    axs[0].plot(xx[idx_13], pr_Fc_x[idx_13, 0], 'ko', ms=8)

    axs[1].plot(xx, pr_Fc_x[:, 1], 'o-', c=c[1])
    axs[1].plot(xx, pr_Fc_x[:, 3], 'o-', c=c[3])
    axs[1].plot(xx[idx_24], pr_Fc_x[idx_24, 1], 'ko', ms=8)

    axs[2].plot(xx, pr_Fc_x[:, 2], 'o-', c=c[2])
    axs[2].plot(xx, pr_Fc_x[:, 4], 'o-', c=c[4])
    axs[2].plot(xx[idx_35], pr_Fc_x[idx_35, 2], 'ko', ms=8)

    _key = key.split('_')
    label = 'Snake {0}, trial {1}'.format(_key[0], _key[1])
    axs[0].text(1, .25, label, fontsize='x-small')

axs_all[-1, 0].set_xlabel('Y (m)', fontsize='small')
axs_all[-1, 0].set_ylabel('X (SVL)', fontsize='small')

axs_all[0, 0].set_title('1-3', fontsize='small')
axs_all[0, 1].set_title('2-4', fontsize='small')
axs_all[0, 2].set_title('3-5', fontsize='small')

sns.despine()

#fig.savefig(FIG.format('socha2010 flow horizontal'), **FIGOPT)


# %% Time series for each trial --- vertical component

fig, axs_all = plt.subplots(len(keys), 3, sharex=True, sharey=True,
                        figsize=(12, 14))
for ax in axs_all.flatten():
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)


for i, key in enumerate(keys):
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy() / d['SVL']
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    X, Y, Z = d['Ro_S'].T
    xx = Y

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    axs = axs_all[i]

    axs[0].plot(xx, pr_Fc_z[:, 0], 'o-', c=c[0])
    axs[0].plot(xx, pr_Fc_z[:, 2], 'o-', c=c[2])
    axs[0].plot(xx[idx_13], pr_Fc_z[idx_13, 0], 'ko')

    axs[1].plot(xx, pr_Fc_z[:, 1], 'o-', c=c[1])
    axs[1].plot(xx, pr_Fc_z[:, 3], 'o-', c=c[3])
    axs[1].plot(xx[idx_24], pr_Fc_z[idx_24, 1], 'ko')

    axs[2].plot(xx, pr_Fc_z[:, 2], 'o-', c=c[2])
    axs[2].plot(xx, pr_Fc_z[:, 4], 'o-', c=c[4])
    axs[2].plot(xx[idx_35], pr_Fc_z[idx_35, 2], 'ko')

sns.despine()


# %% Time series for each trial --- fore-aft component

fig, axs_all = plt.subplots(len(keys), 3, sharex=True, sharey=True,
                        figsize=(12, 14))
for ax in axs_all.flatten():
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)


for i, key in enumerate(keys):
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy() / d['chord']
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    X, Y, Z = d['Ro_S'].T
    xx = Y

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    axs = axs_all[i]

    axs[0].plot(xx, pr_Fc_y[:, 0], 'o-', c=c[0])
    axs[0].plot(xx, pr_Fc_y[:, 2], 'o-', c=c[2])
    axs[0].plot(xx[idx_13], pr_Fc_y[idx_13, 0], 'ko')

    axs[1].plot(xx, pr_Fc_y[:, 1], 'o-', c=c[1])
    axs[1].plot(xx, pr_Fc_y[:, 3], 'o-', c=c[3])
    axs[1].plot(xx[idx_24], pr_Fc_y[idx_24, 1], 'ko')

    axs[2].plot(xx, pr_Fc_y[:, 2], 'o-', c=c[2])
    axs[2].plot(xx, pr_Fc_y[:, 4], 'o-', c=c[4])
    axs[2].plot(xx[idx_35], pr_Fc_y[idx_35, 2], 'ko')

sns.despine()


# %% CoM side view

c = sns.color_palette('husl', n_colors=len(keys))

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                        figsize=(14.75, 4))

for i, key in enumerate(keys):
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy() / d['chord']
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    X, Y, Z = d['Ro_S'].T
    xx = Y

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    sn_id, trial_id = key.split('_')
    label = 'Snake {0}, T{1}'.format(sn_id, trial_id)
    axs[0].plot(Y, Z, c=c[i], label=label)
    axs[0].plot(Y[idx_13], Z[idx_13], 'o', c=c[i])

    axs[1].plot(Y, Z, c=c[i])
    axs[1].plot(Y[idx_24], Z[idx_24], 'o', c=c[i])

    axs[2].plot(Y, Z, c=c[i])
    axs[2].plot(Y[idx_35], Z[idx_35], 'o', c=c[i])

axs[0].legend(loc='upper right', ncol=2, fontsize='xx-small',
              handlelength=1, columnspacing=2)
plt.setp(axs, aspect=1.0, adjustable='box-forced')
axs[0].set_ylim(-.5, 9)
axs[0].set_xlim(-.5, 13)
axs[0].set_xticks([0, 3, 6, 9, 12])
axs[0].set_xlabel('Y (m)', fontsize='small')
axs[0].set_ylabel('Z (m)', fontsize='small')
axs[0].set_title('1-3', fontsize='small')
axs[1].set_title('2-4', fontsize='small')
axs[2].set_title('3-5', fontsize='small')
sns.despine()

#fig.savefig(FIG.format('socha2010 Z vs Y locs'), **FIGOPT)


# %% Glide angle

c = sns.color_palette('husl', n_colors=len(keys))

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                        figsize=(14.75, 4))

for i, key in enumerate(keys):
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy() / d['chord']
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    X, Y, Z = d['Ro_S'].T
    gamma = np.rad2deg(d['gamma'])
    xx = Y

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    sn_id, trial_id = key.split('_')
    label = 'Snake {0}, T{1}'.format(sn_id, trial_id)
    axs[0].plot(Y, gamma, c=c[i], label=label)
    axs[0].plot(Y[idx_13], gamma[idx_13], 'o', c=c[i])

    axs[1].plot(Y, gamma, c=c[i])
    axs[1].plot(Y[idx_24], gamma[idx_24], 'o', c=c[i])

    axs[2].plot(Y, gamma, c=c[i])
    axs[2].plot(Y[idx_35], gamma[idx_35], 'o', c=c[i])

axs[0].legend(loc='upper right', ncol=2, fontsize='xx-small',
              handlelength=1, columnspacing=2)

axs[0].set_ylim(15, 45)
axs[0].set_xlim(-.5, 13)
axs[0].set_xticks([0, 3, 6, 9, 12])
axs[0].set_xlabel('Y (m)', fontsize='small')
axs[0].set_ylabel('Glide angle', fontsize='small')
axs[0].set_title('1-3', fontsize='small')
axs[1].set_title('2-4', fontsize='small')
axs[2].set_title('3-5', fontsize='small')
axs[0].yaxis.set_major_formatter(degree_formatter)
axs[1].yaxis.set_major_formatter(degree_formatter)
axs[2].yaxis.set_major_formatter(degree_formatter)
sns.despine()

#fig.savefig(FIG.format('socha2010 gamma vs Y locs'), **FIGOPT)


# %% Glide angle for each trial

c = sns.color_palette('husl', n_colors=len(keys))

fig, axs = plt.subplots(len(keys), 3, sharex=True, sharey=True,
                        figsize=(10, 14))

for i, key in enumerate(keys):
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy() / d['chord']
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    X, Y, Z = d['Ro_S'].T
    gamma = np.rad2deg(d['gamma'])
    xx = Y

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    sn_id, trial_id = key.split('_')
    label = 'Snake {0}, trial {1}'.format(sn_id, trial_id)
    axs[i, 0].text(3, 40, label, fontsize='xx-small')

    axs[i, 0].plot(Y, gamma, c=c[i])
    axs[i, 0].plot(Y[idx_13], gamma[idx_13], 'o', mec=c[i], mfc='none', mew=1.5)

    axs[i, 1].plot(Y, gamma, c=c[i])
    axs[i, 1].plot(Y[idx_24], gamma[idx_24], 'o', mec=c[i], mfc='none', mew=1.5)

    axs[i, 2].plot(Y, gamma, c=c[i])
    axs[i, 2].plot(Y[idx_35], gamma[idx_35], 'o', mec=c[i], mfc='none', mew=1.5)

#axs[0].legend(loc='upper right', ncol=2, fontsize='xx-small',
#              handlelength=1, columnspacing=2)

ax = axs[-1, 0]
ax.set_ylim(15, 45)
ax.set_xlim(-.5, 13)
ax.set_xticks([0, 3, 6, 9, 12])
ax.set_xlabel('Y (m)', fontsize='small')
ax.set_ylabel('Glide angle', fontsize='small')
axs[0, 0].set_title('1-3', fontsize='small')
axs[0, 1].set_title('2-4', fontsize='small')
axs[0, 2].set_title('3-5', fontsize='small')
ax.yaxis.set_major_formatter(degree_formatter)
ax.yaxis.set_major_formatter(degree_formatter)
ax.yaxis.set_major_formatter(degree_formatter)
sns.despine()

#fig.savefig(FIG.format('socha2010 gamma vs Y locs by trial'), **FIGOPT)


# %% dGlide angle/dt

c = sns.color_palette('husl', n_colors=len(keys))

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                        figsize=(14.75, 4))

for i, key in enumerate(keys):
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy() / d['chord']
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    X, Y, Z = d['Ro_S'].T
    gamma = np.rad2deg(d['gamma'])
    dgamma_dt = np.gradient(gamma, d['dt'], edge_order=2)
    xx = Y

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    sn_id, trial_id = key.split('_')
    label = 'Snake {0}, T{1}'.format(sn_id, trial_id)
    axs[0].plot(Y, dgamma_dt, c=c[i], label=label)
    axs[0].plot(Y[idx_13], dgamma_dt[idx_13], 'o', c=c[i])

    axs[1].plot(Y, dgamma_dt, c=c[i])
    axs[1].plot(Y[idx_24], dgamma_dt[idx_24], 'o', c=c[i])

    axs[2].plot(Y, dgamma_dt, c=c[i])
    axs[2].plot(Y[idx_35], dgamma_dt[idx_35], 'o', c=c[i])

axs[0].legend(loc='upper right', ncol=2, fontsize='xx-small',
              handlelength=1, columnspacing=2)

#axs[0].set_ylim(15, 45)
axs[0].set_xlim(-.5, 13)
axs[0].set_xticks([0, 3, 6, 9, 12])
axs[0].set_xlabel('Y (m)', fontsize='small')
axs[0].set_ylabel(r'$\dot{\gamma}$', fontsize='small')
axs[0].set_title('1-3', fontsize='small')
axs[1].set_title('2-4', fontsize='small')
axs[2].set_title('3-5', fontsize='small')
axs[0].yaxis.set_major_formatter(degree_formatter)
axs[1].yaxis.set_major_formatter(degree_formatter)
axs[2].yaxis.set_major_formatter(degree_formatter)
sns.despine()

fig.savefig(FIG.format('socha2010 dgamma_dt vs Y locs'), **FIGOPT)


# %% dGlide angle/dt for each trial

c = sns.color_palette('husl', n_colors=len(keys))

fig, axs = plt.subplots(len(keys), 3, sharex=True, sharey=True,
                        figsize=(10, 14))

for ax in axs.flatten():
    ax.axhline(0, color='gray', lw=1)

for i, key in enumerate(keys):
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy() / d['chord']
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    X, Y, Z = d['Ro_S'].T
    gamma = np.rad2deg(d['gamma'])
    dgamma_dt = np.gradient(gamma, d['dt'], edge_order=2)
    xx = Y

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    sn_id, trial_id = key.split('_')
    label = 'Snake {0}, trial {1}'.format(sn_id, trial_id)
    axs[i, 0].text(3, 40, label, fontsize='xx-small')

    axs[i, 0].plot(Y, dgamma_dt, c=c[i])
    axs[i, 0].plot(Y[idx_13], dgamma_dt[idx_13], 'o', mec=c[i], mfc='none', mew=1.5)

    axs[i, 1].plot(Y, dgamma_dt, c=c[i])
    axs[i, 1].plot(Y[idx_24], dgamma_dt[idx_24], 'o', mec=c[i], mfc='none', mew=1.5)

    axs[i, 2].plot(Y, dgamma_dt, c=c[i])
    axs[i, 2].plot(Y[idx_35], dgamma_dt[idx_35], 'o', mec=c[i], mfc='none', mew=1.5)

#axs[0].legend(loc='upper right', ncol=2, fontsize='xx-small',
#              handlelength=1, columnspacing=2)

ax = axs[-1, 0]
ax.set_ylim(-50, 50)
ax.set_xlim(-.5, 13)
ax.set_xticks([0, 3, 6, 9, 12])
ax.set_xlabel('Y (m)', fontsize='small')
ax.set_ylabel(r'$\dot{\gamma}$', fontsize='small')
axs[0, 0].set_title('1-3', fontsize='small')
axs[0, 1].set_title('2-4', fontsize='small')
axs[0, 2].set_title('3-5', fontsize='small')
ax.yaxis.set_major_formatter(degree_formatter)
ax.yaxis.set_major_formatter(degree_formatter)
ax.yaxis.set_major_formatter(degree_formatter)
sns.despine()

fig.savefig(FIG.format('socha2010 dgamma_dt vs Y locs by trial'), **FIGOPT)


# %% Top and side view for all glides

# top view
fig_t, axs_t = plt.subplots(4, 6, sharex=True, sharey=True,
                        figsize=(16, 14))
for ax in axs_t.flatten():
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)


# side view
fig_s, axs_s = plt.subplots(len(keys), 3, sharex=True, sharey=True,
                        figsize=(12, 14))
for ax in axs_s.flatten():
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)


for i, key in enumerate(keys):
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy() / d['chord']
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    # top
    if i >= 4:
        row = i - 4
        col = 3
    else:
        row = i
        col = 0
    ax1 = axs_t[row, col + 0]
    ax1.plot(pr_Fc_x[idx_13].T, pr_Fc_y[idx_13].T, '-')
    ax1.set_prop_cycle(None)
    ax1.plot(pr_Fc_x[:, [0, 2]][idx_13].T, pr_Fc_y[:, [0, 2]][idx_13].T, 'o')
    ax2 = axs_t[row, col + 1]
    ax2.plot(pr_Fc_x[idx_24].T, pr_Fc_y[idx_24].T, '-')
    ax2.set_prop_cycle(None)
    ax2.plot(pr_Fc_x[:, [1, 3]][idx_24].T, pr_Fc_y[:, [1, 3]][idx_24].T, 'o')
    ax3 = axs_t[row, col + 2]
    ax3.plot(pr_Fc_x[idx_35].T, pr_Fc_y[idx_35].T, '-')
    ax3.set_prop_cycle(None)
    ax3.plot(pr_Fc_x[:, [2, 4]][idx_35].T, pr_Fc_y[:, [2, 4]][idx_35].T, 'o')

    _key = key.split('_')
    label = 'Snake {1}, trial {2}\n{0}'.format('1-3', _key[0], _key[1])
    ax1.text(-5, 8, label, fontsize='x-small')
    ax2.text(-5, 8, '\n2-4', fontsize='x-small')
    ax3.text(-5, 8, '\n3-5', fontsize='x-small')

    # side
    ax1 = axs_s[i, 0]
    ax1.plot(pr_Fc_y[idx_13].T, pr_Fc_z[idx_13].T, '-')
    ax1.set_prop_cycle(None)
    ax1.plot(pr_Fc_y[:, [0, 2]][idx_13].T, pr_Fc_z[:, [0, 2]][idx_13].T, 'o')
    ax2 = axs_s[i, 1]
    ax2.plot(pr_Fc_y[idx_24].T, pr_Fc_z[idx_24].T, '-')
    ax2.set_prop_cycle(None)
    ax2.plot(pr_Fc_y[:, [1, 3]][idx_24].T, pr_Fc_z[:, [1, 3]][idx_24].T, 'o')
    ax3 = axs_s[i, 2]
    ax3.plot(pr_Fc_y[idx_35].T, pr_Fc_z[idx_35].T, '-')
    ax3.set_prop_cycle(None)
    ax3.plot(pr_Fc_y[:, [2, 4]][idx_35].T, pr_Fc_z[:, [2, 4]][idx_35].T, 'o')

    _key = key.split('_')
    label = 'Snake {1}, trial {2}\n{0}'.format('1-3', _key[0], _key[1])
    ax1.text(-12, 2, label, fontsize='x-small')
    ax2.text(-12, 2, '\n2-4', fontsize='x-small')
    ax3.text(-12, 2, '\n3-5', fontsize='x-small')


axs_t[-1, 0].set_xlabel('X (c)', fontsize='x-small')
axs_t[-1, 0].set_ylabel('Y (c)', fontsize='x-small')

axs_s[-1, 0].set_xlabel('Y (c)', fontsize='x-small')
axs_s[-1, 0].set_ylabel('Z (c)', fontsize='x-small')

plt.setp(axs_t, aspect=1.0, adjustable='box-forced')
plt.setp(axs_s, aspect=1.0, adjustable='box-forced')
sns.despine(fig=fig_t)
sns.despine(fig=fig_s)

#fig_t.savefig(FIG.format('socha2010 top'), **FIGOPT)
#fig_s.savefig(FIG.format('socha2010 side'), **FIGOPT)


# %% Gap and stagger calculation for all glides

g_13, s_13, g_24, s_24, g_35, s_35 = [], [], [], [], [], []
Y_13, Z_13, X_13 = np.array([]), np.array([]), np.array([])
Y_24, Z_24, X_24 = np.array([]), np.array([]), np.array([])
Y_35, Z_35, X_35 = np.array([]), np.array([]), np.array([])

for key in keys:
    d = data5[key]
    pr_Fc = d['pr_Fc'].copy()
    pr_Fc_x = pr_Fc[:, :, 0]
    pr_Fc_y = pr_Fc[:, :, 1]
    pr_Fc_z = pr_Fc[:, :, 2]

    idx_13 = np.where(np.diff(np.sign(pr_Fc_x[:, 0] - pr_Fc_x[:, 2])) != 0)[0] + 1
    idx_24 = np.where(np.diff(np.sign(pr_Fc_x[:, 1] - pr_Fc_x[:, 3])) != 0)[0] + 1
    idx_35 = np.where(np.diff(np.sign(pr_Fc_x[:, 2] - pr_Fc_x[:, 4])) != 0)[0] + 1

    #TODO check the peak here
    chord = .03 * d['SVL']

    # gap and stagger of identified points
    g_13.append((pr_Fc_y[idx_13, 0] - pr_Fc_y[idx_13, 2]) / chord)
    s_13.append((pr_Fc_z[idx_13, 0] - pr_Fc_z[idx_13, 2]) / chord)
    g_24.append((pr_Fc_y[idx_24, 1] - pr_Fc_y[idx_24, 3]) / chord)
    s_24.append((pr_Fc_z[idx_24, 1] - pr_Fc_z[idx_24, 3]) / chord)
    g_35.append((pr_Fc_y[idx_35, 2] - pr_Fc_y[idx_35, 4]) / chord)
    s_35.append((pr_Fc_z[idx_35, 2] - pr_Fc_z[idx_35, 4]) / chord)

    # separations for histograms to match Socha (2010), figure 7e-g
    Y_13 = np.r_[Y_13, (pr_Fc_y[:, 0] - pr_Fc_y[:, 2]) / chord]
    Z_13 = np.r_[Z_13, (pr_Fc_z[:, 0] - pr_Fc_z[:, 2]) / chord]
    X_13 = np.r_[X_13, (pr_Fc_x[:, 0] - pr_Fc_x[:, 2]) / chord]

    Y_24 = np.r_[Y_24, (pr_Fc_y[:, 1] - pr_Fc_y[:, 3]) / chord]
    Z_24 = np.r_[Z_24, (pr_Fc_z[:, 1] - pr_Fc_z[:, 3]) / chord]
    X_24 = np.r_[X_24, (pr_Fc_x[:, 1] - pr_Fc_x[:, 3]) / chord]

    Y_35 = np.r_[Y_35, (pr_Fc_y[:, 2] - pr_Fc_y[:, 4]) / chord]
    Z_35 = np.r_[Z_35, (pr_Fc_z[:, 2] - pr_Fc_z[:, 4]) / chord]
    X_35 = np.r_[X_35, (pr_Fc_x[:, 2] - pr_Fc_x[:, 4]) / chord]


# %% Histograms of spacings

weights = np.ones_like(X_13) / len(X_13)  # http://stackoverflow.com/a/16399202

fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)

ax1, ax2, ax3 = axs

ax = ax1
bins = bins = np.arange(-8.5, 12.6, 1)
ax.hist(Y_13, bins=bins, weights=weights, ec='r', histtype='step')
ax.hist(Y_24, bins=bins, weights=weights, ec='g', histtype='step')
ax.hist(Y_35, bins=bins, weights=weights, ec='b', histtype='step')
ax.set_xticks(bins[::4] + .5)
ax.set_xlabel('Y separation (c)')
ax.set_ylabel('Probability')

ax = ax2
bins = bins = np.arange(-22.5, 22.6, 2)
ax.hist(X_13, bins=bins, weights=weights, ec='r', histtype='step')
ax.hist(X_24, bins=bins, weights=weights, ec='g', histtype='step')
ax.hist(X_35, bins=bins, weights=weights, ec='b', histtype='step')
ax.set_xticks(np.r_[-20, -10, 0, 10 , 20])
ax.set_xlabel('X separation (c)')

ax = ax3
bins = bins = np.arange(-8.5, 12.6, 1)
ax.hist(Z_13, bins=bins, weights=weights, ec='r', histtype='step')
ax.hist(Z_24, bins=bins, weights=weights, ec='g', histtype='step')
ax.hist(Z_35, bins=bins, weights=weights, ec='b', histtype='step')
ax.set_xticks(bins[::4] + .5)
ax.set_xlabel('Z separation (c)')

sns.despine()

#fig.savefig(FIG.format('socha2010 hist'), **FIGOPT)


# %% KDE plots of spacing

fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)

ax1, ax2, ax3 = axs

ax = ax1
bins = bins = np.arange(-8.5, 12.6, 1)
#sns.distplot(G_13, bins=bins, hist=True, kde=True, color='r', ax=ax,
#             hist_kws=dict(histtype='step', weights=weights))
sns.distplot(Y_13, bins=bins, hist=False, kde=True, color='r', ax=ax)
sns.distplot(Y_24, bins=bins, hist=False, kde=True, color='g', ax=ax)
sns.distplot(Y_35, bins=bins, hist=False, kde=True, color='b', ax=ax)
ax.set_xlim(bins[0], bins[-1])
ax.set_xticks(bins[::4] + .5)
ax.set_yticklabels([])
ax.set_xlabel('Y separation (c)')
ax.set_ylabel('Probability')

ax = ax2
bins = bins = np.arange(-22.5, 22.6, 2)
sns.distplot(X_13, bins=bins, hist=False, kde=True, color='r', ax=ax)
sns.distplot(X_24, bins=bins, hist=False, kde=True, color='g', ax=ax)
sns.distplot(X_35, bins=bins, hist=False, kde=True, color='b', ax=ax)
ax.set_xlim(bins[0], bins[-1])
ax.set_xticks(np.r_[-20, -10, 0, 10 , 20])
ax.set_yticklabels([])
ax.set_xlabel('X separation (c)')

ax = ax3
bins = bins = np.arange(-8.5, 12.6, 1)
sns.distplot(Z_13, bins=bins, hist=False, kde=True, color='r', ax=ax)
sns.distplot(Z_24, bins=bins, hist=False, kde=True, color='g', ax=ax)
sns.distplot(Z_35, bins=bins, hist=False, kde=True, color='b', ax=ax)
ax.set_xlim(bins[0], bins[-1])
ax.set_xticks(bins[::4] + .5)
ax.set_yticklabels([])
ax.set_xlabel('Z separation (c)')

sns.despine()

#fig.savefig(FIG.format('socha2010 kde'), **FIGOPT)


# %%

c = sns.color_palette('husl', n_colors=len(keys))

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                        figsize=(12, 3.4))

ax1, ax2, ax3 = axs

for ax in axs:
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

# invert axis, propogates through because shared
ax = axs[0]
ax.invert_xaxis()
ax.invert_yaxis()

for i in np.arange(len(keys)):
    axs[0].plot(g_13[i], s_13[i], 'o', c=c[i], label=keys[i])
    axs[1].plot(g_24[i], s_24[i], 'o', c=c[i])
    axs[2].plot(g_35[i], s_35[i], 'o', c=c[i])

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

for ax in axs:
    for gp in gaps_piv:
        for sp in staggers_piv:
            ax.plot(gp, sp, '+', color='gray', mew=1.)

plt.setp(axs, aspect=1.0, adjustable='box-forced')
# ax1.legend(loc='best', fontsize='xx-small')
ax1.set_xlabel('Gap (c)')
ax1.set_ylabel('Stagger (c)')
ax1.set_title('1-3', fontsize='small')
ax2.set_title('2-4', fontsize='small')
ax3.set_title('3-5', fontsize='small')

sns.despine()

#fig.savefig(FIG.format('socha2010 stagger vs gap'), **FIGOPT)


# %%

fig, axs = plt.subplots(len(keys), 3, sharex=True, sharey=True,
                        figsize=(10, 14.375))

for ax in axs.flatten():
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

ax = axs[0, 0]
ax.invert_xaxis()
ax.invert_yaxis()

for i in np.arange(len(keys)):
    axs[i, 0].plot(g_13[i], s_13[i], 'o', c=c[i])
    axs[i, 1].plot(g_24[i], s_24[i], 'o', c=c[i])
    axs[i, 2].plot(g_35[i], s_35[i], 'o', c=c[i])

    _key = key.split('_')
    label = 'Snake {0}, trial {1}'.format(_key[0], _key[1])
    axs[i, 0].text(17, 10, label, fontsize='x-small')

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

for ax in axs.flatten():
    for gp in gaps_piv:
        for sp in staggers_piv:
            ax.plot(gp, sp, '+', color='gray', mew=1., ms=5)

plt.setp(axs, aspect=1.0, adjustable='box-forced')

axs[0, 0].set_title('1-3', fontsize='small')
axs[0, 1].set_title('2-4', fontsize='small')
axs[0, 2].set_title('3-5', fontsize='small')

axs[-1, 0].set_xlabel('Gap (c)', fontsize='small')
axs[-1, 0].set_ylabel('Stagger (c)', fontsize='small')

sns.despine()

#fig.savefig(FIG.format('socha2010 stagger vs gap by trials'), **FIGOPT)


# %%

# %% Gap and stagger hexbin --- ALL COUNTS --- with Socha (2010)

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots(figsize=(8.7, 4.8))
ax.axvline(0, color='w', lw=.5, ls='--')
ax.axhline(0, color='w', lw=.5, ls='--')
ax.plot(0, 0, 'wx', ms=8, mew=1.5)

cax = ax.hexbin(g, s, mincnt=0, gridsize=100, cmap='viridis',
                linewidths=0.2)
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='x-small')

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='w', mew=1.5)

for i in np.arange(len(keys)):
    # ax.plot(g_13[i], s_13[i], 'o', mfc='none', mec='r', mew=1.5)
    # ax.plot(g_35[i], s_35[i], '^', mfc='none', mec='c', mew=1.5)
    ax.plot(g_24[i], s_24[i], 's', mfc='none', mec='y', mew=1.5)

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.invert_xaxis()
ax.invert_yaxis()

#ax.set_xlim(10, -5)
#ax.set_ylim(10, -2)
ax.set_xlim(12, -5)
ax.set_ylim(8, -2)

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)

#fig.savefig(FIG.format('0 stagger vs gap summary socha2010'), **FIGOPT)


# %% Gap and stagger hexbin --- ALL COUNTS --- with Socha (2010)

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots(figsize=(8.7, 4.8))
ax.axvline(0, color='w', lw=.5, ls='--')
ax.axhline(0, color='w', lw=.5, ls='--')
ax.plot(0, 0, 'wx', ms=8, mew=1.5)

cax = ax.hexbin(g, s, mincnt=0, gridsize=100, cmap='viridis',
                linewidths=0.2)
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='x-small')

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='w', mew=1.5)

for i in np.arange(len(keys)):
    ax.plot(g_13[i], s_13[i], 'o', mfc='none', mec='r', mew=1.5)
for i in np.arange(len(keys)):
    ax.plot(g_35[i], s_35[i], '^', mfc='none', mec='c', mew=1.5)
for i in np.arange(len(keys)):
    ax.plot(g_24[i], s_24[i], 's', mfc='none', mec='y', mew=1.5)

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.invert_xaxis()
ax.invert_yaxis()

#ax.set_xlim(10, -5)
#ax.set_ylim(10, -2)
ax.set_xlim(12, -5)
ax.set_ylim(8, -2)

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)

#fig.savefig(FIG.format('0 stagger vs gap summary all wings socha2010'), **FIGOPT)



# %% Gap and stagger hexbin --- ALL COUNTS

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=.5, ls='-')
ax.axhline(0, color='k', lw=.5, ls='-')
#ax.plot(0, 0, 'kx', ms=8, mew=1.5)

#cax = ax.hexbin(gaps_height[3], stag_height[3], mincnt=0, gridsize=(30, 12),
#                cmap=plt.cm.gray_r,
#                linewidths=0.2, extent=(-5, 10, -2, 10))
cax = ax.hexbin(g, s, mincnt=0, gridsize=(30, 12),
                cmap=plt.cm.gray_r,
                linewidths=0.2, extent=(-5, 10, -2, 10))
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='small')

#counts = cax.get_array()
#ncnts = np.count_nonzero(counts)
#verts = cax.get_offsets()
#ax.plot(verts[:, 0], verts[:, 1], 'k.')  # https://stackoverflow.com/a/13754416

#for offc in np.arange(verts.shape[0]):
#    binx, biny = verts[offc][0], verts[offc][1]
#    if counts[offc]:
#        plt.plot(binx, biny, 'k.', zorder=100)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='r', mew=1.)

for i in np.arange(len(keys)):
    # ax.plot(g_13[i], s_13[i], 'o', mfc='none', mec='r', mew=1.5)
    # ax.plot(g_35[i], s_35[i], '^', mfc='none', mec='c', mew=1.5)
    ax.plot(g_24[i], s_24[i], 's', mfc='none', mec='y', mew=1.5)

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xlim(10, -5)
ax.set_ylim(10, -2)

ax.set_yticks([10, 8, 6, 4, 2, 0, -2])
ax.set_xticks([10, 8, 6, 4, 2, 0, -2, -4])

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)


#fig.savefig(FIG.format('stagger vs gap all end of glide'), **FIGOPT)


# %% SOCHA (2010) AS A HEXBIN PLOT

g_s2010 = np.array([])
s_s2010 = np.array([])
for i in np.arange(len(keys)):
   g_s2010 = np.r_[g_s2010, g_13[i], g_24[i], g_35[i]]
   s_s2010 = np.r_[s_s2010, s_13[i], s_24[i], s_35[i]]

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots()
ax.axvline(0, color='k', lw=.5, ls='-')
ax.axhline(0, color='k', lw=.5, ls='-')
#ax.plot(0, 0, 'kx', ms=8, mew=1.5)

#cax = ax.hexbin(g, s, mincnt=0, gridsize=(30, 12),
#                cmap=plt.cm.gray_r, vmax=5,
#                linewidths=0.2, extent=(-5, 10, -2, 10))
#cax = ax.hexbin(gaps_height[3], stag_height[3], mincnt=0, gridsize=(30, 12),
#                cmap=plt.cm.gray_r, vmax=5,
#                linewidths=0.2, extent=(-5, 10, -2, 10))
cax = ax.hexbin(g_s2010, s_s2010, mincnt=0, gridsize=(30, 12),
                cmap=plt.cm.gray_r,
                linewidths=0.2, extent=(-5, 10, -2, 10))
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='small')

#counts = cax.get_array()
#ncnts = np.count_nonzero(counts)
#verts = cax.get_offsets()
#ax.plot(verts[:, 0], verts[:, 1], 'k.')  # https://stackoverflow.com/a/13754416

#for offc in np.arange(verts.shape[0]):
#    binx, biny = verts[offc][0], verts[offc][1]
#    if counts[offc]:
#        plt.plot(binx, biny, 'k.', zorder=100)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='r', mew=1.)

#for i in np.arange(len(keys)):
#    # ax.plot(g_13[i], s_13[i], 'o', mfc='none', mec='r', mew=1.5)
#    # ax.plot(g_35[i], s_35[i], '^', mfc='none', mec='c', mew=1.5)
#    ax.plot(g_24[i], s_24[i], 's', mfc='none', mec='y', mew=1.5)

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xlim(10, -5)
ax.set_ylim(10, -2)

ax.set_yticks([10, 8, 6, 4, 2, 0, -2])
ax.set_xticks([10, 8, 6, 4, 2, 0, -2, -4])

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)


# %%

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True,
                        figsize=(8.5, 4.75))

(ax1, ax2) = axs

for ax in axs:
    ax.axvline(0, color='k', lw=.5, ls='-')
    ax.axhline(0, color='k', lw=.5, ls='-')
#ax.plot(0, 0, 'kx', ms=8, mew=1.5)

cax1 = ax1.hexbin(g, s, mincnt=0, gridsize=(30, 12),
                  cmap=plt.cm.gray_r, vmax=5,
                  linewidths=0.2, extent=(-5, 10, -2, 10))
#cax1 = ax1.hexbin(gaps_height[3], stag_height[3], mincnt=0, gridsize=(30, 12),
#                cmap=plt.cm.gray_r, vmax=5,
#                linewidths=0.2, extent=(-5, 10, -2, 10))
cax2 = ax2.hexbin(g_s2010, s_s2010, mincnt=0, gridsize=(30, 12),
                  cmap=plt.cm.gray_r,
                  linewidths=0.2, extent=(-5, 10, -2, 10))

cbar1 = fig.colorbar(cax1, ax=ax1, shrink=.75, orientation='horizontal')
cbar1.set_label('Counts', fontsize='x-small')

cbar2 = fig.colorbar(cax2, ax=ax2, shrink=.75, orientation='horizontal')
cbar2.set_label('Counts', fontsize='x-small')

#counts = cax.get_array()
#ncnts = np.count_nonzero(counts)
#verts = cax.get_offsets()
#ax.plot(verts[:, 0], verts[:, 1], 'k.')  # https://stackoverflow.com/a/13754416

#for offc in np.arange(verts.shape[0]):
#    binx, biny = verts[offc][0], verts[offc][1]
#    if counts[offc]:
#        plt.plot(binx, biny, 'k.', zorder=100)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax1.plot(gp, sp, '+', color='r', mew=1.)
        ax2.plot(gp, sp, '+', color='r', mew=1.)

#for i in np.arange(len(keys)):
#    # ax.plot(g_13[i], s_13[i], 'o', mfc='none', mec='r', mew=1.5)
#    # ax.plot(g_35[i], s_35[i], '^', mfc='none', mec='c', mew=1.5)
#    ax.plot(g_24[i], s_24[i], 's', mfc='none', mec='y', mew=1.5)

#ax1.set_xlabel('Gap (c)', fontsize='x-small')
#ax1.set_ylabel('Stagger (c)', fontsize='x-small')

ax1.invert_xaxis()
ax1.invert_yaxis()

ax1.set_xlim(10, -5)
ax1.set_ylim(10, -2)

ax1.set_yticks([10, 5, 0])
ax1.set_xticks([10, 5, 0, -5])

#ax1.set_yticks([10, 8, 6, 4, 2, 0, -2])
#ax1.set_xticks([10, 8, 6, 4, 2, 0, -2, -4])

#ax.set_aspect('equal', adjustable='box')
plt.setp([ax1, ax2], aspect=1.0, adjustable='box-forced')
sns.despine(ax=ax1)
sns.despine(ax=ax2)

#fig.savefig(FIG.format('0 gap and stagger vs socha2010 1x2'), **FIGOPT)


# %%

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                        figsize=(10, 4))

(ax1, ax2, ax3) = axs

for ax in axs:
    ax.axvline(0, color='k', lw=.5, ls='-')
    ax.axhline(0, color='k', lw=.5, ls='-')
#ax.plot(0, 0, 'kx', ms=8, mew=1.5)

cax1 = ax1.hexbin(g, s, mincnt=0, gridsize=(30, 12),
                  cmap=plt.cm.gray_r, # vmax=5,
                  linewidths=0.2, extent=(-5, 10, -2, 10))

cax2 = ax2.hexbin(g_s2010, s_s2010, mincnt=0, gridsize=(30, 12),
                  cmap=plt.cm.gray_r,
                  linewidths=0.2, extent=(-5, 10, -2, 10))

cax3 = ax3.hexbin(g, s, mincnt=0, gridsize=(30, 12),
                  cmap=plt.cm.gray_r, vmax=5,
                  linewidths=0.2, extent=(-5, 10, -2, 10))

#cax3 = ax3.hexbin(gaps_height[3], stag_height[3], mincnt=0, gridsize=(30, 12),
#                cmap=plt.cm.gray_r, vmax=5,
#                linewidths=0.2, extent=(-5, 10, -2, 10))

cbar1 = fig.colorbar(cax1, ax=ax1, shrink=.75, orientation='horizontal')
cbar1.set_label('Counts', fontsize='x-small')

cbar2 = fig.colorbar(cax2, ax=ax2, shrink=.75, orientation='horizontal')
cbar2.set_label('Counts', fontsize='x-small')

cbar3 = fig.colorbar(cax3, ax=ax3, shrink=.75, orientation='horizontal')
cbar3.set_label('Counts', fontsize='x-small')

#counts = cax.get_array()
#ncnts = np.count_nonzero(counts)
#verts = cax.get_offsets()
#ax.plot(verts[:, 0], verts[:, 1], 'k.')  # https://stackoverflow.com/a/13754416

#for offc in np.arange(verts.shape[0]):
#    binx, biny = verts[offc][0], verts[offc][1]
#    if counts[offc]:
#        plt.plot(binx, biny, 'k.', zorder=100)

for gp in gaps_piv:
    for sp in staggers_piv:
        ax1.plot(gp, sp, '+', ms=5, color='r', mew=1.)
        ax2.plot(gp, sp, '+', ms=5, color='r', mew=1.)
        ax3.plot(gp, sp, '+', ms=5, color='r', mew=1.)

#for i in np.arange(len(keys)):
#    # ax.plot(g_13[i], s_13[i], 'o', mfc='none', mec='r', mew=1.5)
#    # ax.plot(g_35[i], s_35[i], '^', mfc='none', mec='c', mew=1.5)
#    ax.plot(g_24[i], s_24[i], 's', mfc='none', mec='y', mew=1.5)

#ax1.set_xlabel('Gap (c)', fontsize='x-small')
#ax1.set_ylabel('Stagger (c)', fontsize='x-small')

ax1.invert_xaxis()
ax1.invert_yaxis()

ax1.set_xlim(10, -5)
ax1.set_ylim(10, -2)

ax1.set_yticks([10, 5, 0])
ax1.set_xticks([10, 5, 0, -5])

#ax1.set_yticks([10, 8, 6, 4, 2, 0, -2])
#ax1.set_xticks([10, 8, 6, 4, 2, 0, -2, -4])

#ax.set_aspect('equal', adjustable='box')
plt.setp([ax1, ax2, ax3], aspect=1.0, adjustable='box-forced')
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.despine(ax=ax3)

#fig.savefig(FIG.format('0 gap and stagger vs socha2010 1x3'), **FIGOPT)



# %%

# %% Gap and stagger hexbin --- ALL COUNTS --- with Socha (2010)

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots(figsize=(8.7, 4.8))
ax.axvline(0, color='w', lw=.5, ls='--')
ax.axhline(0, color='w', lw=.5, ls='--')
ax.plot(0, 0, 'wx', ms=8, mew=1.5)

# d  mincnt=1e6
cax = ax.hexbin(g, s, mincnt=0, gridsize=100, cmap='viridis',
                linewidths=0.2, vmin=0, vmax=88)
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='x-small')

# b
for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='w', mew=1.5)

# c
for i in np.arange(len(keys)):
    ax.plot(g_13[i], s_13[i], 'o', mfc='none', mec='r', mew=1.5)
for i in np.arange(len(keys)):
    ax.plot(g_35[i], s_35[i], '^', mfc='none', mec='c', mew=1.5)
for i in np.arange(len(keys)):
    ax.plot(g_24[i], s_24[i], 's', mfc='none', mec='y', mew=1.5)

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.invert_xaxis()
ax.invert_yaxis()

#ax.set_xlim(10, -5)
#ax.set_ylim(10, -2)
ax.set_xlim(12, -5)
ax.set_ylim(8, -2)

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)

#fig.savefig(FIG.format('0a stagger vs gap'), **FIGOPT)
#fig.savefig(FIG.format('0b stagger vs gap'), **FIGOPT)
#fig.savefig(FIG.format('0c stagger vs gap'), **FIGOPT)
#fig.savefig(FIG.format('0d stagger vs gap'), **FIGOPT)




# %% Gap and stagger hexbin --- mincnt --- ALL COUNTS --- with Socha (2010)

# what Farid measured
gaps_piv = np.r_[2, 4, 6, 8]
staggers_piv = np.r_[0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots(figsize=(8.7, 4.8))
ax.axvline(0, color='gray', lw=.5, ls='--')
ax.axhline(0, color='gray', lw=.5, ls='--')
ax.plot(0, 0, 'x', c='gray', ms=8, mew=1.5)

cax = ax.hexbin(g, s, mincnt=1, gridsize=100, cmap='viridis',
                linewidths=0.2)
cbar = fig.colorbar(cax, ax=ax, shrink=.75)
cbar.set_label('Counts', fontsize='x-small')

for gp in gaps_piv:
    for sp in staggers_piv:
        ax.plot(gp, sp, '+', color='gray', mew=1.5)

for i in np.arange(len(keys)):
    ax.plot(g_24[i], s_24[i], 's', mfc='none', mec='y', mew=1.5)

ax.set_xlabel('Gap (c)')
ax.set_ylabel('Stagger (c)')

ax.invert_xaxis()
ax.invert_yaxis()

#ax.set_xlim(10, -5)
#ax.set_ylim(10, -2)
ax.set_xlim(12, -5)
ax.set_ylim(8, -2)

ax.set_aspect('equal', adjustable='box')
sns.despine(ax=ax)

fig.savefig(FIG.format('0 stagger vs gap mincnt summary socha2010'), **FIGOPT)


# %% Sinuosity --- Socha (2010) and Cube

# C. paradisi to analyze
snake_ids = [81, 91, 95, 88, 90, 86, 94]
nsnakes = len(snake_ids)

# colors for plots
colors = sns.color_palette('husl', nsnakes)

# make the plot
fig, axs = plt.subplots(nsnakes + 2, 1, sharex=True, sharey=True,
                        figsize=(5, 13))

for key in keys:
    snake_id, trial_id = key.split('_')
    if snake_id == '1':
        row = 0
        color = 'b'
    else:
        row = 1
        color = 'm'

    d = data5[key]
    Y = data5[key]['Ro_S'][:, 1]
    sinuosity = data5[key]['sinuosity']
    axs[row].plot(Y, sinuosity, color=color)

axs[0].text(6, .75, 'Snake 1', fontsize='x-small')
axs[1].text(6, .75, 'Snake 2', fontsize='x-small')

for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = colors[row]
    ax = axs[row + 2]

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        dd = np.load(fname)

        pf_I, vent_idx, SVL = dd['pf_I'], dd['vent_idx'], dd['SVL']
        Y = dd['Ro_S'][:, 1] / 1000 - Yo

        head_vent = pf_I[:, 0] - pf_I[:, vent_idx]
        dist_hv = np.linalg.norm(head_vent, axis=1)
        sinuosity = dist_hv / SVL

        ax.plot(Y, sinuosity, c=colors_trial_id)
        label = 'Snake {0}'.format(snake_id)
        if i == 0:
            ax.text(2, .75, label, fontsize='x-small')

axs[-1].set_xlabel('Y (m)', fontsize='small')
axs[-1].set_ylabel('Sinuosity', fontsize='small')

ax.set_ylim(0, 1)
sns.despine()

fig.savefig(FIG.format('sinuosity Cube socha2010'), **FIGOPT)


# %% Sinuosity --- Socha (2010) and Cube

# C. paradisi to analyze
snake_ids = [81, 91, 95, 88, 90, 86, 94]
nsnakes = len(snake_ids)

# colors for plots
colors = sns.color_palette('husl', nsnakes)

# make the plot
fig, axs = plt.subplots(nsnakes, 1, sharex=True, sharey=True,
                        figsize=(5, 12))

for row, snake_id in enumerate(snake_ids):
    fn_names = ret_fnames(snake_id)
    colors_trial_id = colors[row]
    ax = axs[row]

    for i, fname in enumerate(fn_names):
        snake_id, trial_id = trial_info(fname)
        dd = np.load(fname)

        pf_I, vent_idx, SVL = dd['pf_I'], dd['vent_idx'], dd['SVL']
        Y = dd['Ro_S'][:, 1] / 1000 - Yo

        head_vent = pf_I[:, 0] - pf_I[:, vent_idx]
        dist_hv = np.linalg.norm(head_vent, axis=1)
        sinuosity = dist_hv / SVL

        ax.plot(Y, sinuosity, c=colors_trial_id)
        label = 'Snake {0}'.format(snake_id)
        if i == 0:
            ax.text(2, .75, label, fontsize='x-small')

axs[-1].set_xlabel('Y (m)', fontsize='small')
axs[-1].set_ylabel('Sinuosity', fontsize='small')
axs[-1].set_xlim(-.7, 5)

ax.set_ylim(0, 1)
sns.despine()

fig.savefig(FIG.format('sinuosity Cube'), **FIGOPT)

