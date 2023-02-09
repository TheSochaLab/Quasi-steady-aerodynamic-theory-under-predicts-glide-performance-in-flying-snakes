#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:46:04 2017

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

import seaborn as sns
from mayavi import mlab

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Helvetica'}
sns.set('notebook', 'ticks', font='Arial',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_force_dist/{}.pdf'
FIGPNG = '../Figures/s_force_dist/{}.png'
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


from matplotlib.ticker import FuncFormatter

# http://stackoverflow.com/a/8555837
def _formatter_remove_zeros(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str


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


def _formatter_percent(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}%'.format(x)
    return val_str


decimal_formatter = FuncFormatter(_formatter_remove_zeros)
degree_formatter = FuncFormatter(_formatter_degree)
percent_formatter = FuncFormatter(_formatter_percent)


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


# %%

snake_id, trial_id = 95, 618  # best performance
#snake_id, trial_id = 88, 505  # 2nd best performance

fname = ret_fnames(snake_id, trial_id)[0]

d = np.load(fname)


# %%

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


foils_Ic = d['foils_Ic']
foil_color = d['foil_color']
YZ_S = d['YZ_S']

R_Fc, foils_Fc, dRo_F = rotate_to_flow(d)


_nmesh = 21

# extents of the mesh
dx, dy, dz = 75, 75, 75
dx, dy, dz = 100, 100, 100
xx = np.linspace(-dx, dx, _nmesh)
yy = np.linspace(-dy, dy, _nmesh)
zz = np.linspace(-dz, dz, _nmesh)
YZ_y, YZ_z = np.meshgrid(yy, zz)
XZ_x, XZ_z = np.meshgrid(xx, zz)
XY_x, XY_y = np.meshgrid(xx, yy)



# %% 3D plot of the body showing gap and stagger

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 750))

#Nframe = np.eye(3)
#frame_c = [bmap[2], bmap[1], bmap[0]]
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=50,
#                  color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

i = 130
i = 220

#fc_i = foil_color.copy()
#jp = np.where(R_Fc[i, :, 0] >= 0)[0]
#jn = np.where(R_Fc[i, :, 0] < 0)[0]


# inertial axies
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
_args = dict(opacity=.75, tube_radius=1.5)
mlab.plot3d([-dx, dx], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-dy, dy], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-dz, dz], color=frame_c[2], **_args)

#mlab.plot3d([0, 400], [0, 0], [0, 0], color=frame_c[0], **_args)
#mlab.plot3d([200, 200], [-200, 200], [0, 0], color=frame_c[1],**_args)
#mlab.plot3d([200, 200], [0, 0], [-75, 75], color=frame_c[2], **_args)

#pts = mlab.points3d(pfe_Ic[i, :, 0], pfe_Ic[i, :, 1], pfe_Ic[i, :, 2],
#                    color=(.85, .85, .85), scale_factor=10, resolution=64)

body = mlab.mesh(foils_Fc[i, :, :, 0],
                 foils_Fc[i, :, :, 1],
                 foils_Fc[i, :, :, 2],
                 scalars=foil_color[i],
                 colormap='YlGn', opacity=1,
                 vmin=0, vmax=1)

#body = mlab.mesh(foils_Fc[i, jp, :, 0],
#                 foils_Fc[i, jp, :, 1],
#                 foils_Fc[i, jp, :, 2],
#                 scalars=foil_color[i, jp],
#                 colormap='YlGn', opacity=1,
#                 vmin=0, vmax=1)

#body = mlab.mesh(foils_Fc[i, jn, :, 0],
#                 foils_Fc[i, jn, :, 1],
#                 foils_Fc[i, jn, :, 2],
#                 scalars=foil_color[i, jn],
#                 colormap='RdBu', opacity=1,
#                 vmin=0, vmax=1)

_c = (.6, .6, .6)
#_c = bmap[2]
YZ_mesh = mlab.mesh(YZ_y * 0, YZ_y, YZ_z,
                    color=_c, opacity=.3)

qd = mlab.quiver3d(0, 0, 0,
                   dRo_F[i, 0], dRo_F[i, 1], dRo_F[i, 2],
                   scale_factor=.005, color=(0, 0, 0), mode='arrow',
                   opacity=1)


#mlab.orientation_axes()
fig.scene.isometric_view()
fig.scene.parallel_projection = True

if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)


mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)

view_i220 = (52.284433783963983,
 73.25496212766717,
 999.03709305725897,
 np.array([ 73.19377827,  28.21498176,   3.69076595]))


# %%

mlab.savefig(FIGPNG.format('{}_{} i{} flow side'.format(snake_id, trial_id, i)),
                           size=(5 * 750, 5 * 708), figure=fig)


# %%

from scipy.optimize import minimize


# indices to use (determined from the ball drop experiment)
tstart = 8
tstop = -10

g = 9.81


def to_min(boosts, args):
    boost_Fl, boost_Fd = boosts
    Fl, Fd, ddRo = args

    Fa_b = (boost_Fl * Fl + boost_Fd * Fd).sum(axis=0)

    error = np.linalg.norm(ddRo - Fa_b)
    return error


times = d['times'][tstart:tstop]
Z = d['Ro_S'][tstart:tstop, 2] / 1000
mass_kg = float(d['mass']) / 1000
mg = mass_kg * g

# forces and accelerations in STRAIGHTENED frame
ddRo = d['ddRo_S'][tstart:tstop] / 1000
Fl = d['Fl_S'][tstart:tstop]
Fd = d['Fd_S'][tstart:tstop]
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


# %%

#times = d['times'] - d['times'][0]
dt = float(d['dt'])
times = d['times'][tstart:tstop]
ntime = len(times)
vent_loc = d['vent_idx_spl'] + 1
SVL = d['SVL_avg']
start = d['idx_pts'][1]  # 0 is the virtual marker
#start = 0
snon = d['t_coord'][0, start:vent_loc] / SVL
snonf = d['t_coord'][0] / SVL
s_plot = np.arange(vent_loc)
nbody = len(snon)


# body position
R = d['R_Sc'][tstart:tstop]

x, y, z = R[:, 0:vent_loc].T  # TODO
xf, yf, zf = R.T

x, y, z = x.T, y.T, z.T
xf, yf, zf = xf.T, yf.T, zf.T


# bending angles
dRds = d['Tdir_S'][tstart:tstop]
dRds = d['Tdir_I'][tstart:tstop]  #TODO was using _S before 2017-02-13

psi = np.arcsin(dRds[:, start:vent_loc, 2])
psi_f = np.arcsin(dRds[:, :, 2])

theta = np.arctan2(dRds[:, start:vent_loc, 0], -dRds[:, start:vent_loc, 1])
#theta = np.arctan(dRds[:, start:vent_loc, 0] / -dRds[:, start:vent_loc, 1])
theta_f = np.arctan2(dRds[:, :, 0], -dRds[:, :, 1])

## IJY 2017-08-23
## why do we have the negative on the dRds_Y??
#theta = np.arctan2(dRds[:, start:vent_loc, 0], dRds[:, start:vent_loc, 1])
#theta_f = np.arctan2(dRds[:, :, 0], dRds[:, :, 1])

#theta_diff = np.diff(theta)

# 2017-02-22 Maybe don't actually ave to unwrap the angles
psi = np.unwrap(psi, axis=1)
psi_f = np.unwrap(psi_f, axis=1)

theta = np.unwrap(theta, axis=1)
theta_f = np.unwrap(theta_f, axis=1)

# mean remove
psi_mean = psi.mean(axis=1)
theta_mean = theta.mean(axis=1)
psi = (psi.T - psi_mean).T
theta = (theta.T - theta_mean).T
psi_f = (psi_f.T - psi_mean).T
theta_f = (theta_f.T - theta_mean).T

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

    psi_f[i] = psi_f[i] - np.polyval(pp, snonf)

    pp = np.polyfit(snon, theta[i], 1)
    y_lin = np.polyval(pp, snon)
    y_fit = theta[i] - y_lin
    d_theta_pp[i] = pp
    d_theta_fit[i] = y_lin
    theta_detrend[i] = y_fit

## store values with a trend
#theta_trend = theta.copy()
#theta = theta_detrend.copy()

# only remove trend on vertical wave
psi_trend = psi.copy()
psi = psi_detrend.copy()

# find zero crossings of the lateral wave
#idx_zero = []
#for i in np.arange(ntime):
#    zero_crossings = np.where(np.diff(np.signbit(theta[i])))[0]
#    idx_zero.append(zero_crossings)

#idx_zero = []
snon_zr = []
snon_zr_f = []
diff_snon_zr = []
theta_zr, psi_zr = [], []
x_zr, y_zr, z_zr = [], [], []
for i in np.arange(ntime):
    ti = theta[i]

    i0 = np.where(np.diff(np.signbit(theta[i])))[0]
    i1 = i0 + 1
    i0_f, i1_f = i0 + start, i1 + start
    frac = np.abs(ti[i0] / (ti[i1] - ti[i0]))

    zrs_i = snon[i0] + frac * (snon[i1] - snon[i0])
    snon_zr.append(zrs_i)
    snon_zr_f.append(zrs_i + start)
    diff_snon_zr.append(np.diff(zrs_i))

    theta_zr.append(ti[i0] + frac * (ti[i1] - ti[i0]))
    psi_zr.append(psi[i][i0] + frac * (psi[i][i1] - psi[i][i0]))

    x_zr.append(x[i][i0_f] + frac * (x[i][i1_f] - x[i][i0_f]))
    y_zr.append(y[i][i0_f] + frac * (y[i][i1_f] - y[i][i0_f]))
    z_zr.append(z[i][i0_f] + frac * (z[i][i1_f] - z[i][i0_f]))
#    x_zr.append(x[i][i0] + frac * (x[i][i1] - x[i][i0]))
#    y_zr.append(y[i][i0] + frac * (y[i][i1] - y[i][i0]))
#    z_zr.append(z[i][i0] + frac * (z[i][i1] - z[i][i0]))


# %% Yaw and angle rotate position by

yaw = np.rad2deg(d['yaw'][tstart:tstop])
mus = np.rad2deg(d['mus'][tstart:tstop])

fig, ax = plt.subplots()
ax.plot(times, yaw)
ax.plot(times, mus)
ax.plot(times, mus.cumsum())
sns.despine()





# %%







# %% Heat map of bending --- no tail

#i = 170

i = 100

Ss, Tt = np.meshgrid(100 * snon, times)
#Ss, Tt = np.meshgrid(100 * snonf, np.arange(len(times)))

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                               figsize=(8, 4))

cax = ax1.pcolormesh(Tt, Ss, np.rad2deg(theta), cmap=plt.cm.coolwarm,
              vmin=-120, vmax=120, linewidth=0, rasterized=True)
ax1.plot(Tt[i, start:vent_loc + 1], Ss[i, start:vent_loc + 1],
        c='b', lw=3)
cax.set_edgecolor('face')

cax = ax2.pcolormesh(Tt, Ss, np.rad2deg(psi), cmap=plt.cm.coolwarm,
              vmin=-45, vmax=45, linewidth=0, rasterized=True)
ax2.plot(Tt[i, start:vent_loc + 1], Ss[i, start:vent_loc + 1],
        c='g', lw=3)
cax.set_edgecolor('face')

#cbar = fig.colorbar(cax, ax=[ax1, ax2])
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(cax, cax=cbar_ax)

ax1.plot(len(snon_zr[i]) * [Tt[i, 0]], 100 * snon_zr[i],
         'ro', mfc='none', mew=2, mec='r', zorder=1000)
ax2.plot(len(snon_zr[i]) * [Tt[i, 0]], 100 * snon_zr[i],
         'ko', mfc='none', mew=2, mec='k', zorder=1000)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Body coordinate (%SVL)')
ax2.set_xlabel('Time (s)')

ax1.set_xlim(Tt.min(), Tt.max())
ax1.set_ylim(0, 135)
ax1.set_yticks([0, 25, 50, 75, 100, 135])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
fig.set_tight_layout(True)


# %%

#times2D, t_coord = d['times2D'], d['t_coord']
#SVL = d['SVL']
#s_coord = 100 * t_coord / SVL

Ss, Tt = np.meshgrid(100 * snon, times)
Sf, Tf = np.meshgrid(100 * snonf, times)

g = 9.81
mass_kg = float(d['mass']) / 1000
mg = mass_kg * g

Re = d['Re'][tstart:tstop]
aoa = np.rad2deg(d['aoa'][tstart:tstop])
beta = np.abs(np.rad2deg(d['beta'][tstart:tstop]))
#beta = np.rad2deg(d['beta'][tstart:tstop])

dynP_frac = d['dynP_frac'][tstart:tstop]  # , start:vent_loc]

Fl_S = d['Fl_S'][tstart:tstop] / mg
Fd_S = d['Fd_S'][tstart:tstop] / mg
Fa_S = d['Fa_S'][tstart:tstop] / mg

Fl_S_p = (boost[:, 0] * d['Fl_S'][tstart:tstop].T / mg).T
Fd_S_p = (boost[:, 1] * d['Fd_S'][tstart:tstop].T / mg).T
Fa_S_p = Fl_S_p + Fd_S_p

Fl_S_mag = np.linalg.norm(Fl_S, axis=2)
Fd_S_mag = np.linalg.norm(Fd_S, axis=2)
Fa_S_mag = np.linalg.norm(Fa_S, axis=2)

Fl_S_p_mag = np.linalg.norm(Fl_S_p, axis=2)
Fd_S_p_mag = np.linalg.norm(Fd_S_p, axis=2)
Fa_S_p_mag = np.linalg.norm(Fa_S_p, axis=2)

Fl_S_y, Fl_S_z = Fl_S[:, :, 1], Fl_S[:, :, 2]
Fd_S_y, Fd_S_z = Fd_S[:, :, 1], Fd_S[:, :, 2]
Fa_S_y, Fa_S_z = Fa_S[:, :, 1], Fa_S[:, :, 2]

Fl_S_p_y, Fl_S_p_z = Fl_S_p[:, :, 1], Fl_S_p[:, :, 2]
Fd_S_p_y, Fd_S_p_z = Fd_S_p[:, :, 1], Fd_S_p[:, :, 2]
Fa_S_p_y, Fa_S_p_z = Fa_S_p[:, :, 1], Fa_S_p[:, :, 2]


# %%

r = d['R_Sc'][tstart:tstop]
foils = d['foils_Sc'][tstart:tstop]
foil_color = d['foil_color'][tstart:tstop]

Ro = d['Ro_S'][tstart:tstop] / 1000
dRo = d['dRo_S'][tstart:tstop]

Ro_non = Ro / Ro[0]

fracs = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
idx_glide = np.zeros(len(fracs), dtype=np.int)
for i, f in enumerate(fracs):
    idx_glide[i] = np.argmin(np.abs(Ro_non[:, 2] - f))

#idx_glide = np.searchsorted(Ro_non[:, 2], np.r_[.9:.09:-.1])

nbody = Fl_S.shape[1]

L = np.zeros((ntime, nbody, 3, 2))
D = np.zeros((ntime, nbody, 3, 2))
A = np.zeros((ntime, nbody, 3, 2))

Lp = np.zeros((ntime, nbody, 3, 2))
Dp = np.zeros((ntime, nbody, 3, 2))
Ap = np.zeros((ntime, nbody, 3, 2))

#Utot = np.zeros((ntime, nbody, 3, 2))
#U_BC = np.zeros((ntime, nbody, 3, 2))
#U_T = np.zeros((ntime, nbody, 3, 2))

scale_velocities = 1  # .01  # 1/100th
scale_forces = 10000  # 10

for i in np.arange(ntime):
    for j in np.arange(nbody):
        # in inertial frame
        L[i, j, :, 0] = r[i, j]
        L[i, j, :, 1] = r[i, j] + scale_forces * Fl_S[i, j]
        D[i, j, :, 0] = r[i, j]
        D[i, j, :, 1] = r[i, j] + scale_forces * Fd_S[i, j]
        A[i, j, :, 0] = r[i, j]
        A[i, j, :, 1] = r[i, j] + scale_forces * Fa_S[i, j]

        Lp[i, j, :, 0] = r[i, j]
        Lp[i, j, :, 1] = r[i, j] + scale_forces * Fl_S_p[i, j]
        Dp[i, j, :, 0] = r[i, j]
        Dp[i, j, :, 1] = r[i, j] + scale_forces * Fd_S_p[i, j]
        Ap[i, j, :, 0] = r[i, j]
        Ap[i, j, :, 1] = r[i, j] + scale_forces * Fa_S_p[i, j]

#        Utot[i, j, :, 0] = r[i, j]
#        Utot[i, j, :, 1] = r[i, j] + scale_velocities * dR[i, j]
#
#        U_BC[i, j, :, 0] = r[i, j]
#        U_BC[i, j, :, 1] = r[i, j] + scale_velocities * dR_BC[i, j]
#
#        U_T[i, j, :, 0] = r[i, j]
#        U_T[i, j, :, 1] = r[i, j] + scale_velocities * dR_T[i, j]


# %% MOVIE OF FULL GLIDE

i = 10

i = idx_glide[-1]

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

# inertial axies
# inertial axies
frame_c = [bmap[2], bmap[1], bmap[0]]  # x = red, y = green, z = blue
_args = dict(opacity=.75, tube_radius=1.5)
dx, dy, dz = 100, 100, 100
mlab.plot3d([-dx, dx], [0, 0], [0, 0], color=frame_c[0], **_args)
mlab.plot3d([0, 0], [-dy, dy], [0, 0], color=frame_c[1],**_args)
mlab.plot3d([0, 0], [0, 0], [-dz, dz], color=frame_c[2], **_args)


## ensure each figure is the same size (accomdate the entire body)
#ii = 0
#mlab.plot3d(r[ii, :, 0], r[ii, :, 1], r[ii, :, 2], opacity=.1)
#ii = -1
#mlab.plot3d(r[ii, :, 0], r[ii, :, 1], r[ii, :, 2], opacity=.1)

body = mlab.mesh(foils[i, :, :, 0], foils[i, :, :, 1], foils[i, :, :, 2],
                 scalars=foil_color[i], colormap='YlGn', vmin=0, vmax=1)

# CoM velocity
vcom = mlab.quiver3d([dRo[i, 0]], [dRo[i, 1]], [dRo[i, 2]], scale_factor=.01,
                     color=(0, 0, 0), mode='arrow', resolution=64)


#sk = 2
#mlab.quiver3d(r[i, ::sk, 0], r[i, ::sk, 1], r[i, ::sk, 2],
#          Cv[i, ::sk, 0], Cv[i, ::sk, 1], Cv[i, ::sk, 2],
#          color=bmap[1], mode='arrow', resolution=64, scale_factor=.025)
## bhat
#mlab.quiver3d(r[i, ::sk, 0], r[i, ::sk, 1], r[i, ::sk, 2],
#          Bv[i, ::sk, 0], Bv[i, ::sk, 1], Bv[i, ::sk, 2],
#          color=bmap[0], mode='arrow', resolution=64, scale_factor=.025)

#op = .6
#ml = mlab.mesh(L[i, :, 0], L[i, :, 1], L[i, :, 2], color=bmap[0], opacity=op)
#md = mlab.mesh(D[i, :, 0], D[i, :, 1], D[i, :, 2], color=bmap[4], opacity=op)
#ma = mlab.mesh(A[i, :, 0], A[i, :, 1], A[i, :, 2], color=bmap[2], opacity=op)


op = .6
mlp = mlab.mesh(Lp[i, :, 0], Lp[i, :, 1], Lp[i, :, 2], color=bmap[0], opacity=op)
mdp = mlab.mesh(Dp[i, :, 0], Dp[i, :, 1], Dp[i, :, 2], color=bmap[4], opacity=op)
#map = mlab.mesh(Ap[i, :, 0], Ap[i, :, 1], Ap[i, :, 2], color=bmap[2], opacity=op)


#U_Bt = np.zeros_like(Utot)
#U_Bt[:, :, :, 0] = Utot[:, :, :, 0]
#assert(np.allclose(U_Bt[:, :, :, 0], U_BC[:, :, :, 0]))
#U_Bt[:, :, :, 1] =U_Bt[:, :, :, 0] + ( Utot[:, :, :, 1] - U_BC[:, :, :, 1])
#mlab.mesh(Utot[i, :, 0], Utot[i, :, 1], Utot[i, :, 2], color=bmap[2], opacity=.8)
#mlab.mesh(U_BC[i, :, 0], U_BC[i, :, 1], U_BC[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(U_Bt[i, :, 0], U_Bt[i, :, 1], U_Bt[i, :, 2], color=bmap[3], opacity=.8)
#mlab.mesh(U_T[i, :, 0], U_T[i, :, 1], U_T[i, :, 2], color=bmap[0], opacity=.8)

fig.scene.isometric_view()
fig.scene.parallel_projection = True
#mlab.orientation_axes()


if False:
    mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
    mlab.view(azimuth=-90, elevation=90, distance='auto')  # back view (x-z)
    mlab.view(azimuth=-90, elevation=-90, distance='auto')  # front view (x-z)
    mlab.view(azimuth=0, elevation=0, distance='auto')  # top view (x-y)
    mlab.view(azimuth=0, elevation=0, roll=-90, distance='auto')  # top view (x-y) to right
    mlab.view(azimuth=-90, elevation=0, distance='auto')  # top, head to R


# %%

#ffmpeg -f image2 -r 10 -i iso_forces_%03d.png -pix_fmt yuv420p iso_forces_slowed10x.mp4

@mlab.animate(delay=500)
def anim():
    for cnt, i in enumerate(idx_glide):
        print(i)
        print(cnt)
        body.mlab_source.set(x=foils[i, :, :, 0],
                             y=foils[i, :, :, 1],
                             z=foils[i, :, :, 2],
                             scalars=foil_color[i])
#        ml.mlab_source.set(x=L[i, :, 0],
#                           y=L[i, :, 1],
#                           z=L[i, :, 2])
#        md.mlab_source.set(x=D[i, :, 0],
#                           y=D[i, :, 1],
#                           z=D[i, :, 2])
        mlp.mlab_source.set(x=Lp[i, :, 0],
                            y=Lp[i, :, 1],
                            z=Lp[i, :, 2])
        mdp.mlab_source.set(x=Dp[i, :, 0],
                            y=Dp[i, :, 1],
                            z=Dp[i, :, 2])
        vcom.mlab_source.set(u=[dRo[i, 0]],
                             v=[dRo[i, 1]],
                             w=[dRo[i, 2]])

        mlab.view(azimuth=0, elevation=90, distance='auto')  # side view (y-z)
        save_name = '{}_{} forces side i{} f{}'.format(snake_id, trial_id, i, fracs[cnt])
        mlab.savefig(FIGPNG.format(save_name), size=(4 * 750, 4 * 708), figure=fig)

        mlab.view(azimuth=0, elevation=0, roll=-90, distance='auto')  # top view (x-y) to right
        save_name = '{}_{} forces top i{} f{}'.format(snake_id, trial_id, i, fracs[cnt])
        mlab.savefig(FIGPNG.format(save_name), size=(4 * 750, 4 * 708), figure=fig)
        print(save_name)
        yield
manim = anim()
mlab.show()


# %% Re, aoa, beta heat maps for 618_95 and 88_505


snake_id, trial_id = 95, 618  # best performance
snake_id, trial_id = 88, 505  # 2nd best performance


fig, axs = plt.subplots(3, 2, sharey=True, figsize=(9, 10))

axf = axs.flatten()
caxs = np.zeros_like(axs)
cbars = np.zeros_like(axs)


snake_ids, trial_ids = [95, 88], [618, 505]
for i in np.arange(2):
    snake_id, trial_id = snake_ids[i], trial_ids[i]

    fname = ret_fnames(snake_id, trial_id)[0]
    d = np.load(fname)

    times = d['times'][tstart:tstop]
    SVL = d['SVL_avg']
    snonf = d['t_coord'][0] / SVL
    start = d['idx_pts'][1]  # 0 is the virtual marker
    vent_loc = d['vent_idx_spl'] + 1
    snon = d['t_coord'][0, start:vent_loc] / SVL

    Ss, Tt = np.meshgrid(100 * snon, times)
    Sf, Tf = np.meshgrid(100 * snonf, times)

    Re = d['Re'][tstart:tstop]
    aoa = np.rad2deg(d['aoa'][tstart:tstop])
    beta = np.abs(np.rad2deg(d['beta'][tstart:tstop]))

    dRds = d['Tdir_I'][tstart:tstop]  #TODO was using _S before 2017-02-13

    theta = np.arctan2(dRds[:, start:vent_loc, 0], -dRds[:, start:vent_loc, 1])
    theta = np.unwrap(theta, axis=1)
    theta_mean = theta.mean(axis=1)
    theta = (theta.T - theta_mean).T

    caxs[0, i] = axs[0, i].pcolormesh(Tf, Sf, Re, vmin=3000, vmax=13000,
        cmap=plt.cm.Blues, rasterized=True)
    caxs[1, i] = axs[1, i].pcolormesh(Tf, Sf, aoa, vmin=30, vmax=90,
        cmap=plt.cm.Greens, rasterized=True)
    caxs[2, i] = axs[2, i].pcolormesh(Tf, Sf, beta, vmin=0, vmax=90,
        cmap=plt.cm.Reds, rasterized=True)

#    for ax in axs[:, i]:
#        ax.contour(Tt, Ss, theta, [0], colors='gray', linestyles='dashed')
    axs[0, i].contour(Tt, Ss, theta, [0], colors='gray', linestyles='dashed')

    Ro = d['Ro_S'][tstart:tstop] / 1000
    Ro_non = Ro / Ro[0]
    fracs = [.9, .8, .7, .6, .5, .4, .3]  # , .2, .1]
    idx_glide = np.zeros(len(fracs), dtype=np.int)
    for j, f in enumerate(fracs):
        idx_glide[j] = np.argmin(np.abs(Ro_non[:, 2] - f))

    for idx_g in idx_glide:
        tg = Tf[idx_g, 0]
        for ax in axs[:, i]:
            # ax.axvline(tg, color='gray', lw=1)
            ax.plot([tg, tg], [0, 5], 'k', lw=2)


    _format = [None, degree_formatter, degree_formatter]
    for j in np.arange(3):
        cbars[j, i] = fig.colorbar(caxs[j, i], ax=axs[j, i], shrink=.75,
                                   orientation='vertical', format=_format[j])
        cbars[j, i].ax.tick_params(labelsize='x-small')
    cbars[0, i].set_ticks([3000, 5000, 7000, 9000, 11000, 13000])
    cbars[1, i].set_ticks([30, 45, 60, 75, 90])
    cbars[2, i].set_ticks([0, 15, 30, 45, 60, 75, 90])

for ax in axs[:2].flatten():
    ax.set_xticklabels([])

for ax in axf:
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 25, 50, 75, 100, 130])

caxf = caxs.flatten()

for i in np.arange(len(axf)):
    sns.despine(ax=axf[i])

axs[2, 0].set_xlabel('Time (s)', fontsize='small')
axs[2, 1].set_xlabel('Time (s)', fontsize='small')
axs[2, 0].set_ylabel('Distance along body (%SVL)', fontsize='small')


#fig.savefig(FIG.format('618_95 505_88 Re aoa beta contour'), **FIGOPT)


# %% Re, aoa, beta heat maps for 618_95 and 88_505

#import matplotlib.ticker as mticker
#
#percent_formatter = mticker.PercentFormatter(100)

snake_id, trial_id = 95, 618  # best performance
snake_id, trial_id = 88, 505  # 2nd best performance


fig, axs = plt.subplots(4, 2, sharey=True, figsize=(9, 12))

axf = axs.flatten()
caxs = np.zeros_like(axs)
cbars = np.zeros_like(axs)


snake_ids, trial_ids = [95, 88], [618, 505]
for i in np.arange(2):
    snake_id, trial_id = snake_ids[i], trial_ids[i]

    fname = ret_fnames(snake_id, trial_id)[0]
    d = np.load(fname)

    times = d['times'][tstart:tstop]
    SVL = d['SVL_avg']
    snonf = d['t_coord'][0] / SVL
    start = d['idx_pts'][1]  # 0 is the virtual marker
    vent_loc = d['vent_idx_spl'] + 1
    snon = d['t_coord'][0, start:vent_loc] / SVL

    Ss, Tt = np.meshgrid(100 * snon, times)
    Sf, Tf = np.meshgrid(100 * snonf, times)

    Re = d['Re'][tstart:tstop]
    aoa = np.rad2deg(d['aoa'][tstart:tstop])
    beta = np.abs(np.rad2deg(d['beta'][tstart:tstop]))
    dynP_frac = d['dynP_frac'][tstart:tstop]

    dRds = d['Tdir_I'][tstart:tstop]  #TODO was using _S before 2017-02-13

    theta = np.arctan2(dRds[:, start:vent_loc, 0], -dRds[:, start:vent_loc, 1])
    theta = np.unwrap(theta, axis=1)
    theta_mean = theta.mean(axis=1)
    theta = (theta.T - theta_mean).T

    caxs[0, i] = axs[0, i].pcolormesh(Tf, Sf, Re, vmin=3000, vmax=13000,
        cmap=plt.cm.Blues, rasterized=True)
    caxs[1, i] = axs[1, i].pcolormesh(Tf, Sf, aoa, vmin=30, vmax=90,
        cmap=plt.cm.Greens, rasterized=True)
    caxs[2, i] = axs[2, i].pcolormesh(Tf, Sf, beta, vmin=0, vmax=90,
        cmap=plt.cm.Reds, rasterized=True)
    caxs[3, i] = axs[3, i].pcolormesh(Tf, Sf, 100 * dynP_frac, vmin=0, vmax=100,
        cmap=plt.cm.Purples, rasterized=True)

    dynP_cont = .75
    axs[3, i].contour(Tf, Sf, dynP_frac, [dynP_cont], colors='w', linewidths=1)

    axs[0, i].contour(Tt, Ss, theta, [0], colors='gray', linestyles='dashed')

    Ro = d['Ro_S'][tstart:tstop] / 1000
    Ro_non = Ro / Ro[0]
    fracs = [.9, .8, .7, .6, .5, .4, .3]  # , .2, .1]
    idx_glide = np.zeros(len(fracs), dtype=np.int)
    for j, f in enumerate(fracs):
        idx_glide[j] = np.argmin(np.abs(Ro_non[:, 2] - f))

    for idx_g in idx_glide:
        tg = Tf[idx_g, 0]
        for ax in axs[:, i]:
            # ax.axvline(tg, color='gray', lw=1)
            ax.plot([tg, tg], [0, 5], 'k', lw=2)


    _format = [None, degree_formatter, degree_formatter, percent_formatter]
    for j in np.arange(4):
        cbars[j, i] = fig.colorbar(caxs[j, i], ax=axs[j, i], shrink=.75,
                                   orientation='vertical', format=_format[j])
        cbars[j, i].ax.tick_params(labelsize='x-small')
    cbars[0, i].set_ticks([3000, 5000, 7000, 9000, 11000, 13000])
    cbars[1, i].set_ticks([30, 45, 60, 75, 90])
    cbars[2, i].set_ticks([0, 15, 30, 45, 60, 75, 90])
    cbars[3, i].set_ticks([0, 25, 50, 75, 100])
    cbars[3, i].ax.axhline(dynP_cont, color='w', lw=2)

for ax in axs[:3].flatten():
    ax.set_xticklabels([])

for ax in axf:
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 25, 50, 75, 100, 130])

caxf = caxs.flatten()

for i in np.arange(len(axf)):
    sns.despine(ax=axf[i])

axs[3, 0].set_xlabel('Time (s)', fontsize='small')
axs[3, 1].set_xlabel('Time (s)', fontsize='small')
axs[3, 0].set_ylabel('Length (%SVL)', fontsize='small')


#fig.savefig(FIG.format('618_95 505_88 Re aoa beta dynP_frac'), **FIGOPT)
#fig.savefig(FIG.format('618_95 505_88 Re aoa beta dynP_frac_contour'), **FIGOPT)


# %%

mm2m = .001

Tdir_I = d['Tdir_I'][tstart:tstop]

dR_I = d['dR_I'][tstart:tstop] * mm2m
dR_BC_I = d['dR_BC_I'][tstart:tstop]  # m/s
dR_T_I = ((dR_I * Tdir_I).sum(axis=2).T * Tdir_I.T).T

assert(np.allclose(dR_BC_I, dR_I - dR_T_I))

U_BC = d['U_BC_I'][tstart:tstop]
assert(np.allclose(U_BC, np.linalg.norm(dR_BC_I, axis=2)))
U_T = np.linalg.norm(dR_T_I, axis=2)
Umag = np.linalg.norm(dR_I, axis=2)

assert(np.allclose(np.sqrt(U_BC**2 + U_T**2), Umag))

Jinv = U_T / U_BC

dR_Ic = d['dR_Ic'] * mm2m
dR_Sc = d['dR_Sc'] * mm2m


# %%

fig, ax = plt.subplots()

#cax = ax.pcolormesh(Tf, Sf, Umag, cmap=plt.cm.Reds, vmin=0, vmax=10)
#cax = ax.pcolormesh(Tf, Sf, U_BC, cmap=plt.cm.Reds, vmin=0, vmax=10)
#cax = ax.pcolormesh(Tf, Sf, U_T, cmap=plt.cm.Reds, vmin=0, vmax=10)
cax = ax.pcolormesh(Tf, Sf, Jinv, cmap=plt.cm.Reds, vmin=0, vmax=2)

ax.contour(Tt, Ss, theta, [0], colors='w')
#ax.contour(Tt, Ss, psi, [0], colors='w')
cbar = fig.colorbar(cax, ax=ax, shrink=.5)
ax.set_yticks([0, 25, 50, 75, 100, 135])
ax.axhline(100, color='gray', ls='--')
sns.despine(ax=ax)


# %%

fig, axs = plt.subplots(2, 6, sharex=True, sharey=True,
                        figsize=(12, 6))

axf = axs.flatten()
caxs = np.zeros_like(axs)

cbars = np.zeros_like(axf)

j = 0
args = dict(cmap='PiYG', vmin=-.016, vmax=.016)
caxs[0, j] = axs[0, j].pcolormesh(Tf, Sf, Fl_S_mag, **args)
caxs[1, j] = axs[1, j].pcolormesh(Tf, Sf, Fl_S_p_mag, **args)

j = 1
args = dict(cmap='Spectral', vmin=-.016, vmax=.016)
caxs[0, j] = axs[0, j].pcolormesh(Tf, Sf, Fd_S_mag, **args)
caxs[1, j] = axs[1, j].pcolormesh(Tf, Sf, Fd_S_p_mag, **args)

j = 2
caxs[0, j] = axs[0, j].pcolormesh(Tf, Sf, Fa_S_y)
caxs[1, j] = axs[1, j].pcolormesh(Tf, Sf, Fa_S_p_y)

j = 3
caxs[0, j] = axs[0, j].pcolormesh(Tf, Sf, Fa_S_z)
caxs[1, j] = axs[1, j].pcolormesh(Tf, Sf, Fa_S_p_z)

j = 4
caxs[0, j] = axs[0, j].pcolormesh(Tf, Sf, aoa)
caxs[1, j] = axs[1, j].pcolormesh(Tf, Sf, beta)

j = 5
caxs[0, j] = axs[0, j].pcolormesh(Tf, Sf, Re)
caxs[1, j] = axs[1, j].pcolormesh(Tf, Sf, dynP_frac)

caxf = caxs.flatten()

for i in np.arange(len(axf)):
    try:
        cbars[i] = fig.colorbar(caxf[i], ax=axf[i], shrink=.5,
                                orientation='vertical')
    except:
        pass
    sns.despine(ax=axf[i])


# %%

dynP_frac = d['dynP_frac'][:, start:vent_loc]
aoa = np.rad2deg(d['aoa'][:, start:vent_loc])
beta = np.rad2deg(d['beta'][:, start:vent_loc])

i = 100

fig, ax = plt.subplots()
ax.axhline(-90, color='gray', lw=1)
ax.axhline(90, color='gray', lw=1)
ax.axhline(0, color='gray', lw=1)

ax.plot(snon, np.rad2deg(theta)[i])
ax.plot(snon, np.rad2deg(psi)[i])
ax.plot(snon, aoa[i], 'r', lw=2)
ax.plot(snon, beta[i], 'r--')
ax.plot(snon, 100 * dynP_frac[i], 'm')

sns.despine()


# %% Movie of the angle space (psi vs. theta)


fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
sns.despine()

for i in np.arange(0, ntime):
    ax.cla()
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 60)
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.plot(np.rad2deg(theta_f[i]), np.rad2deg(psi_f[i]))
    ax.plot(np.rad2deg(theta_f[i, 0]), np.rad2deg(psi_f[i, 0]), 'ro')
    ax.scatter(np.rad2deg(theta[i]), np.rad2deg(psi[i]), c=snon,
               cmap=plt.cm.viridis, zorder=0)
    fig.canvas.draw()
    plt.pause(.01)



# %% Sweep angle - different axes

aoa = np.rad2deg(d['aoa'][:, start:vent_loc])
beta = np.rad2deg(d['beta'][:, start:vent_loc])

i = 50

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False,
                                    figsize=(4, 6))

ax1.axhline(0, color='gray', lw=1)
ax1.plot(snon, np.rad2deg(theta)[i])
ax1.plot(snon, np.rad2deg(psi)[i])
ax1.set_ylabel('body angles', fontsize='xx-small')

ax2.plot(snon, aoa[i])
ax2.plot(snon, beta[i])
ax2.set_ylabel(r'$\alpha$ and $\beta$', fontsize='xx-small')

ax2.axhline(-90, color='gray', lw=1)
ax2.axhline(90, color='gray', lw=1)

ax3.plot(snon, 100 * dynP_frac[i], 'm')
ax3.set_ylabel('dyn press % (SST)', fontsize='xx-small')

ax3.set_ylim(0, 103)
ax1.set_xlim(0, 1)

sns.despine()

#fig.savefig(FIG.format('body coord i50 618_95'), **FIGOPT)


# %% Check that calculating forces in the different frames are the same as
# calculating in the inertal frame and then rotating

import m_aerodynamics as aerodynamics
aero_interp = aerodynamics.extend_wind_tunnel_data(plot=False)

from m_asd import aero_forces

spl_ds, chord_spl = d['spl_ds'], d['chord_spl']

R_Ic, dR_I = d['R_Ic'], d['dR_I']
Tdir_I, Cdir_I, Bdir_I = d['Tdir_I'], d['Cdir_I'], d['Bdir_I']

R_Sc, dR_S = d['R_Sc'], d['dR_S']
Tdir_S, Cdir_S, Bdir_S = d['Tdir_S'], d['Cdir_S'], d['Bdir_S']

rho = 1.17  # kg/m^3
mm2m = .001

ntime, nspl, _ = R_Ic.shape

Fl_I = np.zeros((ntime, nspl, 3))
Fd_I = np.zeros((ntime, nspl, 3))
Fa_I = np.zeros((ntime, nspl, 3))

Fl_S = np.zeros((ntime, nspl, 3))
Fd_S = np.zeros((ntime, nspl, 3))
Fa_S = np.zeros((ntime, nspl, 3))
Ml_S = np.zeros((ntime, nspl, 3))
Md_S = np.zeros((ntime, nspl, 3))
Ma_S = np.zeros((ntime, nspl, 3))

Re_I = np.zeros((ntime, nspl))
aoa_I = np.zeros((ntime, nspl))
beta_I = np.zeros((ntime, nspl))

Re_S = np.zeros((ntime, nspl))
aoa_S = np.zeros((ntime, nspl))
beta_S = np.zeros((ntime, nspl))

for i in np.arange(ntime):
    # aerodynamic forces, angles
    out = aero_forces(Tdir_I[i], Cdir_I[i], Bdir_I[i], dR_I[i],
                      spl_ds[i], chord_spl[i], rho, aero_interp,
                      full_out=True)

    # store the values
    Fl_I[i] = out['Fl']
    Fd_I[i] = out['Fd']
    Fa_I[i] = out['Fa']
    Re_I[i] = out['Re']
    aoa_I[i] = out['aoa']
    beta_I[i] = out['beta']


    # aerodynamic forces, angles
    out = aero_forces(Tdir_S[i], Cdir_S[i], Bdir_S[i], dR_S[i],
                      spl_ds[i], chord_spl[i], rho, aero_interp,
                      full_out=True)

    # store the values
    Fl_S[i] = out['Fl']
    Fd_S[i] = out['Fd']
    Fa_S[i] = out['Fa']
    Ml_S[i] = np.cross(R_Sc[i], Fl_S[i])  # Nmm
    Md_S[i] = np.cross(R_Sc[i], Fd_S[i])  # Nmm
    Ma_S[i] = np.cross(R_Sc[i], Fa_S[i])  # Nmm
    Re_S[i] = out['Re']
    aoa_S[i] = out['aoa']
    beta_S[i] = out['beta']


# assertions
assert(np.allclose(d['Re'], Re_I))
assert(np.allclose(Re_I, Re_S))

assert(np.allclose(d['aoa'], aoa_I))
assert(np.allclose(aoa_I, aoa_S))

assert(np.allclose(d['beta'], beta_I))
assert(np.allclose(beta_I, beta_S))

assert(np.allclose(d['Fl_I'], Fl_I))
assert(np.allclose(d['Fd_I'], Fd_I))
assert(np.allclose(d['Fa_I'], Fa_I))

assert(np.allclose(d['Fl_S'], Fl_S))
assert(np.allclose(d['Fd_S'], Fd_S))
assert(np.allclose(d['Fa_S'], Fa_S))

assert(np.allclose(d['Ml_S'], Ml_S))
assert(np.allclose(d['Md_S'], Md_S))
assert(np.allclose(d['Ma_S'], Ma_S))


# %% Try fitting the TCB coordinate system in straightned and see how
# it compares

import m_asd as asd

C_I2S = d['C_I2S']
dRds_I, ddRds_I, dddRds_I = d['dRds_I'], d['ddRds_I'], d['dddRds_I']
dRds_S, ddRds_S, dddRds_S = d['dRds_S'], d['ddRds_S'], d['dddRds_S']

out = asd.apply_body_cs(dRds_I, ddRds_I, dddRds_I, C_I2S)
Tdir_I, Cdir_I, Bdir_I = out['Tdir_I'], out['Cdir_I'], out['Bdir_I']

out = asd.apply_body_cs(dRds_S, ddRds_S, dddRds_S, C_I2S)
Tdir_S, Cdir_S, Bdir_S = out['Tdir_I'], out['Cdir_I'], out['Bdir_I']


# %%

assert(np.allclose(d['Tdir_I'], Tdir_I))
assert(np.allclose(d['Cdir_I'], Cdir_I))
assert(np.allclose(d['Bdir_I'], Bdir_I))


assert(np.allclose(d['Tdir_S'], Tdir_S))
assert(np.allclose(d['Cdir_S'], Cdir_S))
assert(np.allclose(d['Bdir_S'], Bdir_S))






