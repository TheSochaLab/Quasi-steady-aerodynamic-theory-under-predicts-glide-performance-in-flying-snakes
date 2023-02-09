#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:26:41 2017

@author: isaac
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from mayavi import mlab

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Helvetica'}
sns.set('notebook', 'ticks', font='Helvetica',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_all_proc_plots/{}.pdf'
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


# %%

fname = ret_fnames(91, 413)[0]
d = np.load(fname)

R_Sc = d['R_Sc']
pf_Sc = d['pf_Sc']
pfe_Sc = d['pfe_Sc']
mass_spl = d['mass_spl']
foils_Sc = d['foils_Sc']
foil_color = d['foil_color']
Fl_S, Fd_S, Fa_S = d['Fl_S'], d['Fd_S'], d['Fa_S']
Tdir_S, Cdir_S, Bdir_S = d['Tdir_S'], d['Cdir_S'], d['Bdir_S']

# a visually plesaing time point
i = 110

# a good camera angle and distance
#view_save = (55.826409254334664, 58.922421240924997, 670.61545304290667,
#             np.array([-50.66807202, -48.27570399,  -2.28887983]))
view_save = (55.826409254334671, 58.922421240924997, 711.59716535398661,
             np.array([-57.70447829, -52.89882712,  10.61504041]))
azimuth = view_save[0]
elevation = view_save[1]
distance = view_save[2]
focalpoint = view_save[3]

# color of the markers (look like IR tape color)
gc = 200
gray = tuple(np.r_[gc, gc, gc] / 255)

nbody = R_Sc.shape[1]

L = np.zeros((nbody, 3, 2))
D = np.zeros((nbody, 3, 2))
A = np.zeros((nbody, 3, 2))

scale_velocities = .01  # 1/100th
scale_forces = 7500

for j in np.arange(nbody):
    L[j, :, 0] = R_Sc[i, j]
    L[j, :, 1] = R_Sc[i, j] + scale_forces * Fl_S[i, j]
    D[j, :, 0] = R_Sc[i, j]
    D[j, :, 1] = R_Sc[i, j] + scale_forces * Fd_S[i, j]
    A[j, :, 0] = R_Sc[i, j]
    A[j, :, 1] = R_Sc[i, j] + scale_forces * Fa_S[i, j]

#Nframe = np.eye(3)
#frame_c = [bmap[2], bmap[1], bmap[0]]
#for ii in np.arange(3):
#    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=40,
#                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)


# %%

# size (750, 751) becomes (750, 708) when plotted
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

Nframe = np.eye(3)
frame_c = [bmap[2], bmap[1], bmap[0]]
for ii in np.arange(3):
    mlab.quiver3d(Nframe[ii, 0], Nframe[ii, 1], Nframe[ii, 2], scale_factor=40,
                      color=frame_c[ii], mode='arrow', opacity=1, resolution=64)

# recorded markers
mlab.points3d(pf_Sc[i, :, 0], pf_Sc[i, :, 1], pf_Sc[i, :, 2],
              color=gray, scale_factor=12, resolution=64)

# virtual marker
mlab.points3d(pfe_Sc[i, 1, 0], pfe_Sc[i, 1, 1], pfe_Sc[i, 1, 2],
              color=gray, scale_factor=8, resolution=64)

# spline fit
mlab.plot3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2],
            color=bmap[1], tube_radius=3)

# mass distribution
mlab.points3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2], mass_spl[i],
              color=bmap[1], scale_factor=15)

# cdir, bdir
mlab.quiver3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2],
              Cdir_S[i, :, 0], Cdir_S[i, :, 1], Cdir_S[i, :, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)
mlab.quiver3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2],
              Bdir_S[i, :, 0], Bdir_S[i, :, 1], Bdir_S[i, :, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)

#skip = 3
#mlab.quiver3d(R_Sc[i, ::skip, 0], R_Sc[i, ::skip, 1], R_Sc[i, ::skip, 2],
#              Cdir_S[i, ::skip, 0], Cdir_S[i, ::skip, 1], Cdir_S[i, ::skip, 2],
#              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)
#mlab.quiver3d(R_Sc[i, ::skip, 0], R_Sc[i, ::skip, 1], R_Sc[i, ::skip, 2],
#              Bdir_S[i, ::skip, 0], Bdir_S[i, ::skip, 1], Bdir_S[i, ::skip, 2],
#              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)

## tdir
#mlab.quiver3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2],
#              Tdir_S[i, :, 0], Tdir_S[i, :, 1], Tdir_S[i, :, 2],
#              color=bmap[1], mode='arrow', resolution=64, scale_factor=25)

# foil
mlab.mesh(foils_Sc[i, :, :, 0], foils_Sc[i, :, :, 1], foils_Sc[i, :, :, 2],
          scalars=foil_color[i], colormap='YlGn', vmin=0, vmax=1)

# aerodynamic forces
op = .6
mlab.mesh(L[:, 0], L[:, 1], L[:, 2], color=bmap[0], opacity=op)
mlab.mesh(D[:, 0], D[:, 1], D[:, 2], color=bmap[4], opacity=op)
# mlab.mesh(A[:, 0], A[:, 1], A[:, 2], color=bmap[2], opacity=op)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)

#save_name = '0_points_413_91_i110'
#mlab.savefig('../Figures/data_buildup/{0}.png'.format(save_name),
#             size=(4 * 750, 4 * 708), figure=fig)


# %%

# size (750, 751) becomes (750, 708) when plotted
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

# recorded markers
mlab.points3d(pf_Sc[i, :, 0], pf_Sc[i, :, 1], pf_Sc[i, :, 2],
              color=gray, scale_factor=12, resolution=64)

# virtual marker
mlab.points3d(pfe_Sc[i, 1, 0], pfe_Sc[i, 1, 1], pfe_Sc[i, 1, 2],
              color=gray, scale_factor=8, resolution=64)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)

#save_name = '0_points_413_91_i110'
#mlab.savefig('../Figures/data_buildup/{0}.png'.format(save_name),
#             size=(4 * 750, 4 * 708), figure=fig)


# %%

# size (750, 751) becomes (750, 708) when plotted
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

# recorded markers
mlab.points3d(pf_Sc[i, :, 0], pf_Sc[i, :, 1], pf_Sc[i, :, 2],
              color=gray, scale_factor=12, resolution=64)

# virtual marker
mlab.points3d(pfe_Sc[i, 1, 0], pfe_Sc[i, 1, 1], pfe_Sc[i, 1, 2],
              color=gray, scale_factor=8, resolution=64)

# spline fit
mlab.plot3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2],
            color=bmap[1], tube_radius=3)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)

save_name = '1_spline_413_91_i110'
mlab.savefig('../Figures/data_buildup/{0}.png'.format(save_name),
             size=(4 * 750, 4 * 708), figure=fig)


# %%

# size (750, 751) becomes (750, 708) when plotted
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

# recorded markers
mlab.points3d(pf_Sc[i, :, 0], pf_Sc[i, :, 1], pf_Sc[i, :, 2],
              color=gray, scale_factor=12, resolution=64)

# virtual marker
mlab.points3d(pfe_Sc[i, 1, 0], pfe_Sc[i, 1, 1], pfe_Sc[i, 1, 2],
              color=gray, scale_factor=8, resolution=64)

# spline fit
mlab.plot3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2],
            color=bmap[1], tube_radius=3)

# cdir, bdir
mlab.quiver3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2],
              Cdir_S[i, :, 0], Cdir_S[i, :, 1], Cdir_S[i, :, 2],
              color=bmap[2], mode='arrow', resolution=64, scale_factor=25)
mlab.quiver3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2],
              Bdir_S[i, :, 0], Bdir_S[i, :, 1], Bdir_S[i, :, 2],
              color=bmap[0], mode='arrow', resolution=64, scale_factor=25)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)

save_name = '2_cdir_bdir_413_91_i110'
mlab.savefig('../Figures/data_buildup/{0}.png'.format(save_name),
             size=(4 * 750, 4 * 708), figure=fig)


# %%

# size (750, 751) becomes (750, 708) when plotted
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

## recorded markers
#mlab.points3d(pf_Sc[i, :, 0], pf_Sc[i, :, 1], pf_Sc[i, :, 2],
#              color=gray, scale_factor=12, resolution=64)
#
## virtual marker
#mlab.points3d(pfe_Sc[i, 1, 0], pfe_Sc[i, 1, 1], pfe_Sc[i, 1, 2],
#              color=gray, scale_factor=8, resolution=64)

# tdir
skip = 3
mlab.quiver3d(R_Sc[i, ::skip, 0], R_Sc[i, ::skip, 1], R_Sc[i, ::skip, 2],
              Tdir_S[i, ::skip, 0], Tdir_S[i, ::skip, 1], Tdir_S[i, ::skip, 2],
              color=bmap[1], mode='arrow', resolution=64, scale_factor=25)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)

save_name = '3b_tdir_413_91_i110'
mlab.savefig('../Figures/data_buildup/{0}.png'.format(save_name),
             size=(4 * 750, 4 * 708), figure=fig)


# %%

# size (750, 751) becomes (750, 708) when plotted
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

# mass distribution
mlab.points3d(R_Sc[i, :, 0], R_Sc[i, :, 1], R_Sc[i, :, 2], mass_spl[i],
              color=bmap[1], scale_factor=15)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)

save_name = '4_mass_distribution_413_91_i110'
mlab.savefig('../Figures/data_buildup/{0}.png'.format(save_name),
             size=(4 * 750, 4 * 708), figure=fig)


# %%

# size (750, 751) becomes (750, 708) when plotted
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

# recorded markers
mlab.points3d(pf_Sc[i, 1:, 0], pf_Sc[i, 1:, 1], pf_Sc[i, 1:, 2],
              color=gray, scale_factor=12, resolution=64)

# virtual marker
mlab.points3d(pfe_Sc[i, 1, 0], pfe_Sc[i, 1, 1], pfe_Sc[i, 1, 2],
              color=gray, scale_factor=8, resolution=64)

# foil
mlab.mesh(foils_Sc[i, :, :, 0], foils_Sc[i, :, :, 1], foils_Sc[i, :, :, 2],
          scalars=foil_color[i], colormap='YlGn', vmin=0, vmax=1)

## aerodynamic forces
#op = .6
#mlab.mesh(L[:, 0], L[:, 1], L[:, 2], color=bmap[0], opacity=op)
#mlab.mesh(D[:, 0], D[:, 1], D[:, 2], color=bmap[4], opacity=op)
# mlab.mesh(A[:, 0], A[:, 1], A[:, 2], color=bmap[2], opacity=op)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)

save_name = '5_foil_413_91_i110'
mlab.savefig('../Figures/data_buildup/{0}.png'.format(save_name),
             size=(4 * 750, 4 * 708), figure=fig)


# %%

# size (750, 751) becomes (750, 708) when plotted
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(750, 751))

## recorded markers
#mlab.points3d(pf_Sc[i, 1:, 0], pf_Sc[i, 1:, 1], pf_Sc[i, 1:, 2],
#              color=gray, scale_factor=12, resolution=64)
#
## virtual marker
#mlab.points3d(pfe_Sc[i, 1, 0], pfe_Sc[i, 1, 1], pfe_Sc[i, 1, 2],
#              color=gray, scale_factor=8, resolution=64)

# foil
mlab.mesh(foils_Sc[i, :, :, 0], foils_Sc[i, :, :, 1], foils_Sc[i, :, :, 2],
          scalars=foil_color[i], colormap='YlGn', vmin=0, vmax=1)

# aerodynamic forces
op = .8
mlab.mesh(L[:, 0], L[:, 1], L[:, 2], color=bmap[0], opacity=op)
mlab.mesh(D[:, 0], D[:, 1], D[:, 2], color=bmap[4], opacity=op)
# mlab.mesh(A[:, 0], A[:, 1], A[:, 2], color=bmap[2], opacity=op)

fig.scene.isometric_view()

mlab.view(azimuth, elevation, distance, focalpoint)

save_name = '6b_forces_413_91_i110'
mlab.savefig('../Figures/data_buildup/{0}.png'.format(save_name),
             size=(4 * 750, 4 * 708), figure=fig)
