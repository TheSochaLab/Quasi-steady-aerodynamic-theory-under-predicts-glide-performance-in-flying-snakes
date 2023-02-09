#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:44:18 2017

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
from scipy.io import loadmat
import pandas as pd

import seaborn as sns
# from mayavi import mlab

rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'cm',
      'font.sans-serif': 'Arial'}
sns.set('notebook', 'ticks', font='Arial',
        font_scale=1.5, color_codes=True, rc=rc)
bmap = sns.color_palette()

# where to save plots
FIG = '../Figures/s_tandem/{}.pdf'
FIGOPT = {'transparent': True, 'bbox_inches': 'tight'}


# %%

# Re = 13,000 (p. 84 of thesis)
LD_single = loadmat('../Data/Aerodynamics/LD_single.mat')
single = {}
single['aoa'] = LD_single['aoa'].flatten()[2::4]
single['Re'] = np.arange(3000, 15001, 2000)[5]  # Re = 13,000
single['drag'] = LD_single['drag'][2::4, 5]
single['lift'] = LD_single['lift'][2::4, 5]

single['aoa_all'] = LD_single['aoa'].flatten()
single['drag_all'] = LD_single['drag'][:, 5]
single['lift_all'] = LD_single['lift'][:, 5]

# Select aoa = 30 deg; LD_single['aoa'].flatten()[8] == 30
single['drag30'] = LD_single['drag'][8, 5]
single['lift30'] = LD_single['lift'][8, 5]

#fig, ax = plt.subplots()
#ax.plot(single['aoa'], single['drag'], single['aoa'], single['lift'])


# %%
# Re = 13,000 (p. 84 of thesis)
LD_tandem = loadmat('../Data/Aerodynamics/LD_tandem.mat')

gaps = np.r_[2, 4, 6, 8]
staggers = np.r_[0, 1, 2, 3, 4, 5]
alphas = np.r_[0, 20, 40, 60]
alpha0 = 30  # for the fixed upstream or downstream airfoil

# 6 x 4 x 4 arrays of (stagger, gap, varying angle of attack)
# _1 alpha_u = 30, downstream varies
# _2 alpha_d = 30, upstream varies
#drag_d_1 = LD_tandem['drag_d_1']
#drag_d_2 = LD_tandem['drag_d_2']
#drag_u_1 = LD_tandem['drag_u_1']
#drag_u_2 = LD_tandem['drag_u_2']
#lift_d_1 = LD_tandem['lift_d_1']
#lift_d_2 = LD_tandem['lift_d_2']
#lift_u_1 = LD_tandem['lift_u_1']
#lift_u_2 = LD_tandem['lift_u_2']

data = {}

data[1] = {}
data[1]['drag_d'] = LD_tandem['drag_d_1']
data[1]['drag_u'] = LD_tandem['drag_u_1']
data[1]['lift_d'] = LD_tandem['lift_d_1']
data[1]['lift_u'] = LD_tandem['lift_u_1']

#data[2] = {}
#data[2]['drag_d'] = LD_tandem['drag_d_2']
#data[2]['drag_u'] = LD_tandem['drag_u_2']
#data[2]['lift_d'] = LD_tandem['lift_d_2']
#data[2]['lift_u'] = LD_tandem['lift_u_2']

#TODO 2017-04-29 I think u and d are switched in the original data
data[2] = {}
data[2]['drag_u'] = LD_tandem['drag_d_2']
data[2]['drag_d'] = LD_tandem['drag_u_2']
data[2]['lift_u'] = LD_tandem['lift_d_2']
data[2]['lift_d'] = LD_tandem['lift_u_2']


# %% Figure 6 from Jafari

#https://matplotlib.org/users/colormapnorms.html
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


G, S = np.meshgrid(gaps, staggers)
#S, G = np.meshgrid(staggers, gaps)


fig, axs = plt.subplots(4, 2, sharex=True, sharey=True,
                        figsize=(5, 8),
                        gridspec_kw=dict(wspace=.15, hspace=0.05,
                        left=.15, right=.95, bottom=.05, top=.95))

# upstream airfoil at 30 deg
d = data[1]
for i in np.arange(4):
    ax = axs[i, 0]
    L = d['lift_u'][:, :, i] + d['lift_d'][:, :, i]
    D = d['drag_u'][:, :, i] + d['drag_d'][:, :, i]
    LD = L / D
    norm = MidpointNormalize(midpoint=1)
    cs = ax.contour(G, S, LD, levels=np.arange(0, 2.6, .1),
                    cmap=plt.cm.Spectral_r,
                    norm=norm)
    ax.clabel(cs, inline=0, fontsize=8, fmt='%1.1f', colors='k')
    # ax.plot(G.flat, S.flat, '+', mec='gray', mew=1)

# downstream airfoil at 30 deg
d = data[2]
for i in np.arange(4):
    ax = axs[i, 1]
    L = d['lift_u'][:, :, i] + d['lift_d'][:, :, i]
    D = d['drag_u'][:, :, i] + d['drag_d'][:, :, i]
    LD = L / D
    norm = MidpointNormalize(midpoint=1)
    cs = ax.contour(G, S, LD, levels=np.arange(0, 2.6, .1),
                    cmap=plt.cm.Spectral_r,
                    norm=norm)
    ax.clabel(cs, inline=0, fontsize=8, fmt='%1.1f', colors='k')
    # ax.plot(G.flat, S.flat, '+', mec='gray', mew=1)


plt.setp(axs, aspect=1.0, adjustable='box-forced')
sns.despine()
ax.set_xticks(gaps)
ax.set_yticks(staggers)


# %% Figure 6 from Jafari, with interpolation

from scipy.interpolate import griddata

#https://matplotlib.org/users/colormapnorms.html
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# combine the data and interpolated it on a grid
gint = np.linspace(gaps[0], gaps[-1], 61)
sint = np.linspace(staggers[0], staggers[-1], 51)

G, S = np.meshgrid(gaps, staggers)
Gf, Sf = G.flat, S.flat


fig, axs = plt.subplots(4, 2, sharex=True, sharey=True,
                        figsize=(5, 8),
                        gridspec_kw=dict(wspace=.15, hspace=0.05,
                        left=.15, right=.95, bottom=.05, top=.95))

levels=np.arange(0.4, 2.6, .05)
# upstream airfoil at 30 deg
d = data[1]
L = d['lift_u'] + d['lift_d']
D = d['drag_u'] + d['drag_d']
LD = L / D
for i in np.arange(4):
    ax = axs[i, 0]
    sns.despine(ax=ax)

    # interpolate
    LDint = griddata((Gf, Sf), LD[:, :, i].flat,
                    (gint[None, :], sint[:, None]),
                    method='cubic')

    norm = MidpointNormalize(midpoint=1)
    cs = ax.contourf(gint, sint, LDint, levels, cmap=plt.cm.RdYlBu_r,
                    norm=norm, rasterized=True)
    ax.clabel(cs, levels[2::5], inline=0, fontsize=8, fmt='%g', colors='k')
#    ax.plot(Gf, Sf, '+', mec='gray', mew=1, ms=7)

# downstream airfoil at 30 deg
d = data[2]
L = d['lift_u'] + d['lift_d']
D = d['drag_u'] + d['drag_d']
LD = L / D
for i in np.arange(4):
    ax = axs[i, 1]
    sns.despine(ax=ax)

    # interpolate
    LDint = griddata((Gf, Sf), LD[:, :, i].flat,
                    (gint[None, :], sint[:, None]),
                    method='cubic')

    norm = MidpointNormalize(midpoint=1)
#    cs = ax.contour(G, S, LD, levels=np.arange(0, 2.4, .1), cmap=plt.cm.Reds)
#    cs = ax.contour(gint, sint, LDint, levels=np.arange(0, 2.4, .1), cmap=plt.cm.Reds)
    cs = ax.contourf(gint, sint, LDint, levels, cmap=plt.cm.RdYlBu_r,
                    norm=norm, rasterized=True)

#    cs = ax.contour(gint, sint, LDint, levels=levels, cmap=plt.cm.RdYlBu_r)
    ax.clabel(cs, levels[2::5], inline=0, fontsize=8, fmt='%g', colors='k')

#    cbar = fig.colorbar(cs, ax=ax, shrink=1, ticks=levels[2::10], format='%g')

#    cs = ax.contour(gint, sint, LDint, levels[7::5], colors=['k'],
#                norm=norm, rasterized=True)
#    ax.plot(Gf, Sf, '+', mec='gray', mew=1, ms=7)


plt.setp(axs, aspect=1.0, adjustable='box-forced')
ax.set_xticks(gaps)
ax.set_yticks(staggers)


# %% Figure 6 from Jafari - dots - SENT TO FARID

#https://matplotlib.org/users/colormapnorms.html
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


G, S = np.meshgrid(gaps, staggers)
#S, G = np.meshgrid(staggers, gaps)


fig, axs = plt.subplots(4, 2, sharex=True, sharey=True,
                        figsize=(5, 8),
                        gridspec_kw=dict(wspace=.15, hspace=0.05,
                        left=.15, right=.95, bottom=.05, top=.95))

# upstream airfoil at 30 deg
d = data[1]
L = d['lift_u'] + d['lift_d']
D = d['drag_u'] + d['drag_d']
LD = L / D
for i in np.arange(4):
    ax = axs[i, 0]
    sns.despine(ax=ax)

    # interpolate
    LDint = griddata((Gf, Sf), LD[:, :, i].flat,
                    (gint[None, :], sint[:, None]),
                    method='cubic')

    norm = MidpointNormalize(midpoint=1)

    ax.scatter(G.flat, S.flat, s=120, c=LD[:, :, i].flat,
               norm=norm, cmap=plt.cm.RdYlBu_r,
               vmin=0.4, vmax=2.5)

#    ax.contour(gint, sint, LDint, [1], colors='k', linestyles='-')
    cs = ax.contour(gint, sint, LDint, 4, cmap=plt.cm.RdYlBu_r,
                    norm=norm, vmin=0.4, vmax=2.5, rasterized=False)

    ax.clabel(cs, inline=0, fontsize=8, fmt='%g', colors='k')


# downstream airfoil at 30 deg
d = data[2]
L = d['lift_u'] + d['lift_d']
D = d['drag_u'] + d['drag_d']
LD = L / D
for i in np.arange(4):
    ax = axs[i, 1]
    sns.despine(ax=ax)

    # interpolate
    LDint = griddata((Gf, Sf), LD[:, :, i].flat,
                    (gint[None, :], sint[:, None]),
                    method='cubic')
    norm = MidpointNormalize(midpoint=1)

#    ax.scatter(G.flat, S.flat, s=120,  c=LD[:, :, i].flat, norm=norm, cmap=plt.cm.RdYlBu_r)
    ax.scatter(G.flat, S.flat, s=120,  c=LD[:, :, i].flat,
               norm=norm, cmap=plt.cm.RdYlBu_r,
               vmin=0.4, vmax=2.5)
#    ax.contour(gint, sint, LDint, [1], colors='k')

#    cs = ax.contour(gint, sint, LDint, [.5, 1, 1.5, 2], cmap=plt.cm.RdYlBu_r,
#                    norm=norm, rasterized=True)

    cs = ax.contour(gint, sint, LDint, 4, cmap=plt.cm.RdYlBu_r,
                    norm=norm, vmin=0.4, vmax=2.5, rasterized=False)

    fmt = '%g'
    ax.clabel(cs, inline=0, fontsize=8, fmt=fmt, colors='k')


plt.setp(axs, aspect=1.0, adjustable='box-forced')
sns.despine()
ax.set_xticks(gaps)
ax.set_yticks(staggers)


# %% Figure 6 from Jafari - dots - DEFENSE TALK

#https://matplotlib.org/users/colormapnorms.html
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


G, S = np.meshgrid(gaps, staggers)
#S, G = np.meshgrid(staggers, gaps)


fig, axs = plt.subplots(4, 2, sharex=True, sharey=True,
                        figsize=(5.5, 9.5),
                        gridspec_kw=dict(wspace=.15, hspace=0.05,
                        left=.15, right=.95, bottom=.05, top=.95))

# upstream airfoil at 30 deg
d = data[1]
L = d['lift_u'] + d['lift_d']
D = d['drag_u'] + d['drag_d']
LD = L / D
for i in np.arange(4):
    ax = axs[i, 0]
    sns.despine(ax=ax)

    # interpolate
    LDint = griddata((Gf, Sf), LD[:, :, i].flat,
                    (gint[None, :], sint[:, None]),
                    method='cubic')

    norm = MidpointNormalize(midpoint=1)

    ax.scatter(G.flat, S.flat, s=120, c=LD[:, :, i].flat,
               norm=norm, cmap=plt.cm.RdYlBu_r,
               vmin=0.4, vmax=2.5)

#    cs = ax.contour(gint, sint, LDint, [.5, 1, 1.5, 2, 2.5], cmap=plt.cm.RdYlBu_r,
#                    norm=norm, rasterized=True, vmin=0.4, vmax=2.5)

#    ax.contour(gint, sint, LDint, [1], colors='k', linestyles='-')
    cs = ax.contour(gint, sint, LDint, 4, cmap=plt.cm.RdYlBu_r,
                    norm=norm, vmin=0.4, vmax=2.5, rasterized=False)

    ax.clabel(cs, inline=0, fontsize=8, fmt='%g', colors='k')


# downstream airfoil at 30 deg
d = data[2]
L = d['lift_u'] + d['lift_d']
D = d['drag_u'] + d['drag_d']
LD = L / D
for i in np.arange(4):
    ax = axs[i, 1]
    sns.despine(ax=ax)

    # interpolate
    LDint = griddata((Gf, Sf), LD[:, :, i].flat,
                    (gint[None, :], sint[:, None]),
                    method='cubic')
    norm = MidpointNormalize(midpoint=1)

#    ax.scatter(G.flat, S.flat, s=120,  c=LD[:, :, i].flat, norm=norm, cmap=plt.cm.RdYlBu_r)
    ax.scatter(G.flat, S.flat, s=120,  c=LD[:, :, i].flat,
               norm=norm, cmap=plt.cm.RdYlBu_r,
               vmin=0.4, vmax=2.5)
#    ax.contour(gint, sint, LDint, [1], colors='k')

#    cs = ax.contour(gint, sint, LDint, [.5, 1, 1.5, 2, 2.5], cmap=plt.cm.RdYlBu_r,
#                    norm=norm, rasterized=True, vmin=0.4, vmax=2.5)

    cs = ax.contour(gint, sint, LDint, 4, cmap=plt.cm.RdYlBu_r,
                    norm=norm, vmin=0.4, vmax=2.5, rasterized=False)

    fmt = '%g'
    ax.clabel(cs, inline=0, fontsize=8, fmt=fmt, colors='k')


#ax.set_xlim(xmin=-1)
#ax.set_xticks(np.arange(0, 10, 2))
ax.set_xticks(gaps)
ax.set_yticks(staggers)

ax.invert_xaxis()
ax.invert_yaxis()

#for ax in axs.flatten():
#    ax.plot(0, 0, 'ko', ms=12)
#    ax.axhline(0, color='gray', lw=1)
#    ax.axvline(0, color='gray', lw=1)

axs[3, 0].set_xlabel('Stagger (c)', fontsize='x-small')
axs[3, 0].set_ylabel('Gap (c)', fontsize='x-small')

plt.setp(axs, aspect=1.0, adjustable='box-forced')
sns.despine()

fig.set_tight_layout(True)

#ax.set_xticks(gaps)
#ax.set_yticks(staggers)

#fig.savefig(FIG.format('Dot matrix contours'), **FIGOPT)


# %% Load in crosssection shape and rotate it

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
rfoil_rot = rot_foil(rfoil, 30)

if False:
    fig, ax = plt.subplots()
    ax.axvline(-.5, color='gray', lw=1)
    ax.axvline(.5, color='gray', lw=1)
    ax.plot(rfoil[:, 0], rfoil[:, 1])
    #ax.plot(rfoil_rot[:, 0], rfoil_rot[:, 1])
    ax.fill_between(rfoil_rot[:, 0], rfoil_rot[:, 1], alpha=.5)
    ax.set_aspect('equal', adjustable='box')
    sns.despine()


# %% Testing locations

fig, ax = plt.subplots()

ax.plot(0, 0, 'ro')
for gap in gaps:
    for stagger in staggers:
        ax.plot(gap, stagger, 'o', c='gray')

ax.axis('equal', adjustable='box')
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel(r'gap, $\Delta$y (chord)')
ax.set_ylabel(r'stagger, $\Delta$z (chord)')
sns.despine()
fig.set_tight_layout(True)


# %% Configuration 1: upstream fixed

f30 = rot_foil(rfoil, 30)

f0 = rfoil
f20 = rot_foil(rfoil, 20)
f40 = rot_foil(rfoil, 40)
f60 = rot_foil(rfoil, 60)

fig, ax = plt.subplots()

#ax.plot(0, 0, 'ro')
ax.fill(-f30[:, 0], -f30[:, 1], 'gray')
for gap in gaps:
    for stagger in staggers:
        #ax.plot(gap, stagger, 'o', c='gray')
        alpha = .75
        ax.fill(-f0[:, 0] + gap, -f0[:, 1] + stagger, color=bmap[0], alpha=alpha)
        ax.fill(-f20[:, 0] + gap, -f20[:, 1] + stagger, color=bmap[1], alpha=alpha)
        ax.fill(-f40[:, 0] + gap, -f40[:, 1] + stagger, color=bmap[2], alpha=alpha)
        ax.fill(-f60[:, 0] + gap, -f60[:, 1] + stagger, color=bmap[3], alpha=alpha)
        # ax.plot(gap, stagger, 'ko')

ax.axis('equal', adjustable='box')
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel(r'gap, $\Delta$y (chord)')
ax.set_ylabel(r'stagger, $\Delta$z (chord)')
ax.axis('off')
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('testing_loc_1'), **FIGOPT)


# %% Configuration 2: downstream fixed

f30 = rot_foil(rfoil, 30)

f0 = rfoil
f20 = rot_foil(rfoil, 20)
f40 = rot_foil(rfoil, 40)
f60 = rot_foil(rfoil, 60)

fig, ax = plt.subplots()

alpha = .75
ax.fill(-f0[:, 0], -f0[:, 1], color=bmap[0], alpha=alpha)
ax.fill(-f20[:, 0], -f20[:, 1], color=bmap[1], alpha=alpha)
ax.fill(-f40[:, 0], -f40[:, 1], color=bmap[2], alpha=alpha)
ax.fill(-f60[:, 0], -f60[:, 1], color=bmap[3], alpha=alpha)
for gap in gaps:
    for stagger in staggers:
        ax.fill(-f30[:, 0] + gap, -f30[:, 1] + stagger, color='gray')

ax.axis('equal', adjustable='box')
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel(r'gap, $\Delta$y (chord)')
ax.set_ylabel(r'stagger, $\Delta$z (chord)')
ax.axis('off')
sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('testing_loc_2'), **FIGOPT)


# %%

N = 4  # len(alphas)
ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars
#colors = flatui = ["#9b59b6", "#3498db", "#e74c3c"]
#colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
#colors = sns.xkcd_palette(colors)
#colors = sns.color_palette('husl', 3)
colors = sns.husl_palette(3, l=.65)[::-1]

show_title = False

for trial in [1, 2]:
    fig, axs = plt.subplots(6, 4, sharex=True, sharey=True, figsize=(9, 11))

    for k, aoa in enumerate(alphas):
        for i, stagger in enumerate(staggers):
            for j, gap in enumerate(gaps):
            # for j in np.arange(len(gaps))[::-1]:
            #     j = 4 - j
                gap = gaps[j]
                cl_d = data[trial]['lift_d'][i, j, k]
                cd_d = data[trial]['drag_d'][i, j, k]
                clcd_d = cl_d / cd_d

                cl_u = data[trial]['lift_u'][i, j, k]
                cd_u = data[trial]['drag_u'][i, j, k]
                clcd_u = cl_u / cd_u

                cl = (cl_d + cl_u) / 2
                cd = (cd_d + cd_u) / 2
                clcd = cl / cd

                cl_s = single['lift'][k]
                cd_s = single['drag'][k]
                clcd_s = cl_s / cd_s

                cl_s30 = single['lift30']
                cd_s30 = single['drag30']
                clcd_s30 = cl_s30 / cd_s30

                # percent change in force coefficients
                # only interested on effect on downstream airfoil
                if trial == 1:
                    cl_orig = cl_s
                    cd_orig = cd_s
                    clcd_orig = clcd_s
                elif trial == 2:
                    cl_orig = cl_s30
                    cd_orig = cd_s30
                    clcd_orig = clcd_s30
                dcl_d = (cl_d - cl_orig) / cl_orig * 100
                dcd_d = (cd_d - cd_orig) / cd_orig * 100
                dclcd_d = (clcd_d - clcd_orig) / clcd_orig * 100

                #TODO change _d
                # of the tandem system together
#                dcl_d = (cl - cl_orig) / cl_orig * 100
#                dcd_d = (cd - cd_orig) / cd_orig * 100
#                dclcd_d = (clcd - clcd_orig) / clcd_orig * 100

                ax = axs[i, 3 - j]
                if show_title and k == 3:
                    ax.set_title('g{0} s{1}'.format(gap, stagger),
                                 fontsize='xx-small')

#                ax.bar(ind[k], dcl_d, width, color=bmap[k])
#                ax.bar(ind[k] + width, dcd_d, width, color=bmap[k])
#                ax.bar(ind[k] + 2 * width, dclcd_d, width, color=bmap[k])

                ax.bar(ind[k], dcl_d, width, color=colors[0])
                ax.bar(ind[k] + width, dcd_d, width, color=colors[1])
                ax.bar(ind[k] + 2 * width, dclcd_d, width, color=colors[2])

                # ax.set_xticks(ind + 3 * width / 2)
                ax.set_xticks(ind + width)
                ax.set_xticklabels((0, 20, 40, 60))

        sns.despine(bottom=True)
        for ax in axs.flatten():
            ax.axhline(0, color='gray', lw=1)
            # ax.set_yticks([-300, -200, -100, 0, 100, 200, 300])
#            ax.set_ylim([-100, 100])
#            ax.set_yticks([-100, -50, 0, 50, 100])
            ax.set_ylim([-90, 90])
            ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
            ax.grid(True, axis='y')
            ax.xaxis.set_tick_params(length=0, width=.5)
            ax.yaxis.set_tick_params(length=0, width=.5)  # length=5
            [tick.label.set_fontsize('xx-small') for tick in ax.xaxis.get_major_ticks()]
            [tick.label.set_fontsize('xx-small') for tick in ax.yaxis.get_major_ticks()]

    # add degree symbol to angles
    fig.canvas.draw()
    for ax in axs[-1].flatten():
        ticks = ax.get_xticklabels()
        newticks = []
        for tick in ticks:
            tick.set_rotation(0)
            text = tick.get_text()  # remove float
            newticks.append(text + u'\u00B0')
        ax.set_xticklabels(newticks)

    for ax in axs[:, 0].flatten():
        ticks = ax.get_yticklabels()
        newticks = []
        for tick in ticks:
            tick.set_rotation(0)
            text = tick.get_text()  # remove float
            newticks.append(text + '%')
        ax.set_yticklabels(newticks)

#    fig.savefig(FIG.format('per_change_{0}'.format(trial)), **FIGOPT)


# %%

N = 4  # len(alphas)
ind = np.arange(N)  # the x locations for the groups
colors = sns.husl_palette(3, l=.65)[::-1]

show_title = False

for trial in [1, 2]:
    fig, axs = plt.subplots(6, 4, sharex=True, sharey=True, figsize=(9, 11))

    for k, aoa in enumerate(alphas):
        for i, stagger in enumerate(staggers):
            for j, gap in enumerate(gaps):
                cl_d = data[trial]['lift_d'][i, j, k]
                cd_d = data[trial]['drag_d'][i, j, k]
                clcd_d = cl_d / cd_d

                cl_u = data[trial]['lift_u'][i, j, k]
                cd_u = data[trial]['drag_u'][i, j, k]
                clcd_u = cl_u / cd_u

                cl = (cl_d + cl_u) / 2
                cd = (cd_d + cd_u) / 2
                clcd = cl / cd

                cl_s = single['lift'][k]
                cd_s = single['drag'][k]
                clcd_s = cl_s / cd_s

                cl_s30 = single['lift30']
                cd_s30 = single['drag30']
                clcd_s30 = cl_s30 / cd_s30

                # only interested on effect on downstream airfoil
                if trial == 1:  # the downstream foil rotates
                    cl_orig = cl_s
                    cd_orig = cd_s
                    clcd_orig = clcd_s
                elif trial == 2:  # the downstream foil is fixed
                    cl_orig = cl_s30
                    cd_orig = cd_s30
                    clcd_orig = clcd_s30

                ax = axs[i, 3 - j]
                if show_title and k == 3:
                    ax.set_title('g{0} s{1}'.format(gap, stagger),
                                 fontsize='xx-small')

                x1 = [ind[k], ind[k]]
                x2 = [ind[k] + width, ind[k] + width]
                x3 = [ind[k] + 2 * width, ind[k] + 2 * width]
                al1 = .5 if cl_orig > cl_d else 1
                al2 = .5 if cd_orig < cd_d else 1
                al3 = .5 if clcd_orig > clcd_d else 1
                ax.plot(x1, [cl_orig, cl_d], color=colors[0], alpha=al1)
                ax.plot(x2, [cd_orig, cd_d], color=colors[1], alpha=al2)
                ax.plot(x3, [clcd_orig, clcd_d], color=colors[2], alpha=al3)

                ax.plot(ind[k], cl_d, 'o', ms=4, color=colors[0])
                ax.plot(ind[k] + width, cd_d, 'o', ms=4,  color=colors[1])
                ax.plot(ind[k] + 2 * width, clcd_d, 'o', ms=4, color=colors[2])
#                ax.plot(ind[k], cl_d, 'o', ms=4, mfc='none', mew=2.5, mec=colors[0])
#                ax.plot(ind[k] + width, cd_d, 'o', ms=4, mfc='none', mew=1.5, mec=colors[1])
#                ax.plot(ind[k] + 2 * width, clcd_d, 'o', ms=4, mfc='none', mew=1.5, mec=colors[2])

                ax.set_xticks(ind + width)
                ax.set_xticklabels((0, 20, 40, 60))

        sns.despine(bottom=True)
        for ax in axs.flatten():
            ax.axhline(0, color='gray', lw=1)
            ax.set_ylim(-2.5, 3)
            ax.set_yticks([-2, -1, 0, 1, 2, 3])
            ax.xaxis.set_tick_params(length=0, width=.5)
            ax.yaxis.set_tick_params(length=4, width=.5)  # length=5
            [tick.label.set_fontsize('xx-small') for tick in ax.xaxis.get_major_ticks()]
            [tick.label.set_fontsize('xx-small') for tick in ax.yaxis.get_major_ticks()]

    # add degree symbol to angles
    fig.canvas.draw()
    for ax in axs[-1].flatten():
        ticks = ax.get_xticklabels()
        newticks = []
        for tick in ticks:
            tick.set_rotation(0)
            text = tick.get_text()  # remove float
            newticks.append(text + u'\u00B0')
        ax.set_xticklabels(newticks)

    for ax in axs[:, 0].flatten():
        ticks = ax.get_yticklabels()
        newticks = []
        for tick in ticks:
            tick.set_rotation(0)
            text = tick.get_text()  # remove float
            newticks.append(text)
        ax.set_yticklabels(newticks)

#    fig.savefig(FIG.format('new_coeffs_{0}'.format(trial)), **FIGOPT)


# %% Cl vs Cd for up and downstream airfoils, 4 plot

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                        figsize=(7, 7))

plt.setp(axs, aspect=1.0, adjustable='box-forced')

for trial in [1, 2]:
    for k in np.arange(4):
        for i, stagger in enumerate(staggers):
            for j, gap in enumerate(gaps):
                ax1, ax2 = axs[trial - 1]
                cl_d = data[trial]['lift_d'][i, j, k]
                cd_d = data[trial]['drag_d'][i, j, k]
                cl_u = data[trial]['lift_u'][i, j, k]
                cd_u = data[trial]['drag_u'][i, j, k]

                if trial == 1:  # downstream varies
                    cl_d_s = single['lift'][k]
                    cd_d_s = single['drag'][k]
                    cl_u_s = single['lift30']
                    cd_u_s = single['drag30']
                    mc_1 = bmap[k]
                    mc_2 = 'y'

                if trial == 2:  # upstream varies
                    cl_d_s = single['lift30']
                    cd_d_s = single['drag30']
                    cl_u_s = single['lift'][k]
                    cd_u_s = single['drag'][k]
                    mc_1 = 'y'
                    mc_2 = bmap[k]

                ax1.plot(cd_d, cl_d, 'o', c='none', mew=1.5, mec=bmap[k])
                ax2.plot(cd_u, cl_u, 'o', c='none', mew=1.5, mec=bmap[k])

                ax1.plot(cd_d_s, cl_d_s, 's', color=mc_1, mew=1, mec='k',
                         zorder=100)
                ax2.plot(cd_u_s, cl_u_s, 's', color=mc_2, mew=1, mec='k',
                         zorder=100)

for ax in axs.flatten():
    ax.axvline(0, color='gray', lw=.5)
    ax.axhline(0, color='gray', lw=.5)
    ax.plot(single['drag_all'], single['lift_all'], 'k', zorder=0)
    ax.set_xlim(-.25, 2.25)
    ax.set_ylim(-1, 2)
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_xticks([0, 1, 2])

#axs[1, 0].set_xlabel('drag coefficient')
#axs[1, 0].set_ylabel('lift coefficient')
axs[1, 0].set_xlabel(r'$C_D$')
axs[1, 0].set_ylabel(r'$C_L$')
axs[0, 0].set_title('Downstream', fontsize='small')
axs[0, 1].set_title('Upstream', fontsize='small')

sns.despine()
fig.set_tight_layout(True)

#fig.savefig(FIG.format('polar_diagram'), **FIGOPT)


# %% Combined lift and drag of the system

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

for ax in axs.flatten():
    ax.axvline(0, color='gray', lw=.5)
    ax.axhline(0, color='gray', lw=.5)
    ax.plot(single['drag_all'], single['lift_all'], 'k')

for trial in [1, 2]:
    for k in np.arange(4):
        for i, stagger in enumerate(staggers):
            for j, gap in enumerate(gaps):
                ax = axs[trial - 1]
                cl_d = data[trial]['lift_d'][i, j, k]
                cd_d = data[trial]['drag_d'][i, j, k]
                cl_u = data[trial]['lift_u'][i, j, k]
                cd_u = data[trial]['drag_u'][i, j, k]

                cl = (cl_d + cl_u) / 2
                cd = (cd_d + cd_u) / 2

                cl_s = single['lift'][k]
                cd_s = single['drag'][k]

                ax.plot(cd, cl, 'o', c='none', mew=1.5, mec=bmap[k])
                ax.plot(cd_s, cl_s, 's', color=bmap[k], mew=1, mec='k',
                         zorder=100)

plt.setp(axs, aspect=1.0, adjustable='box-forced')
ax = axs[0]
ax.set_xlim(-.1, 2)
ax.set_ylim(-1, 2)
sns.despine()
fig.set_tight_layout(True)


# %% Contour plot of values (aoa_d vs. aoa_u)

from scipy.interpolate import griddata

i = 0  # 1c stagger
j = 1  # 2c gap

trial = 1
cl_d = data[trial]['lift_d'][i, j]
cd_d = data[trial]['drag_d'][i, j]
cl_u = data[trial]['lift_u'][i, j]
cd_u = data[trial]['drag_u'][i, j]
cl = (cl_d + cl_u) / 2
cd = (cd_d + cd_u) / 2
clcd = cl / cd
#clcd_d = cl_d / cd_d
aoa_d = alphas
aoa_u = np.r_[30, 30, 30, 30]
#dd_1 = np.c_[aoa_u, aoa_d, cl_d]
dd_1 = np.c_[aoa_u, aoa_d, cl]
dd_1 = np.c_[aoa_u, aoa_d, cd]
dd_1 = np.c_[aoa_u, aoa_d, clcd]

trial = 2
cl_d = data[trial]['lift_d'][i, j]
cd_d = data[trial]['drag_d'][i, j]
cl_u = data[trial]['lift_u'][i, j]
cd_u = data[trial]['drag_u'][i, j]
clcd_d = cl_d / cd_d
cl = (cl_d + cl_u) / 2
cd = (cd_d + cd_u) / 2
clcd = cl / cd
#clcd_d = cl_d / cd_d
aoa_d = np.r_[30, 30, 30, 30]
aoa_u = alphas
#dd_2 = np.c_[aoa_u, aoa_d, cl_d]
dd_2 = np.c_[aoa_u, aoa_d, cl]
dd_2 = np.c_[aoa_u, aoa_d, cd]
dd_2 = np.c_[aoa_u, aoa_d, clcd]


# combine the data and interpolated it on a grid
dd = np.r_[dd_1, dd_2]
xint = np.linspace(0, 60, 61)
yint = np.linspace(0, 60, 61)
zint = griddata((dd[:, 0], dd[:, 1]), dd[:, 2],
                 (xint[None, :], yint[:, None]),
                 method='cubic')

fig, ax = plt.subplots()
ax.contour(xint, yint, zint, 20, linewidths=0.5, colors='k')
cax = ax.contourf(xint, yint, zint, 20, cmap=plt.cm.viridis)
ax.plot(dd[:, 0], dd[:, 1], '+', ms=10, mew=2, color='gray')
fig.colorbar(cax, ax=ax) # draw colorbar
ax.axis('equal', adjustable='box')
ax.set_xlim(-5, 65)
ax.set_ylim(-5, 65)
sns.despine(ax=ax)


# %% Interpolate on a grid --- function

def ginterp(data, stagger_i, gap_j):
    # https://scipy-cookbook.readthedocs.io/items/
    # Matplotlib_Gridding_irregularly_spaced_data.html

    trial = 1
    cl_d = data[trial]['lift_d'][stagger_i, gap_j]
    cd_d = data[trial]['drag_d'][stagger_i, gap_j]
    cl_u = data[trial]['lift_u'][stagger_i, gap_j]
    cd_u = data[trial]['drag_u'][stagger_i, gap_j]
    cl = (cl_d + cl_u) / 2
    cd = (cd_d + cd_u) / 2
    clcd = cl / cd

    cl, cd, clcd = cl_d, cd_d, cl_d / cd_d  #TODO: just downstream airfoil

    aoa_u = np.r_[30, 30, 30, 30]
    aoa_d = np.r_[0, 20, 40, 60]
    dd_1 = np.c_[aoa_u, aoa_d, cl, cd, clcd]

    trial = 2
    cl_d = data[trial]['lift_d'][stagger_i, gap_j]
    cd_d = data[trial]['drag_d'][stagger_i, gap_j]
    cl_u = data[trial]['lift_u'][stagger_i, gap_j]
    cd_u = data[trial]['drag_u'][stagger_i, gap_j]
    cl = (cl_d + cl_u) / 2
    cd = (cd_d + cd_u) / 2
    clcd = cl / cd

    cl, cd, clcd = cl_d, cd_d, cl_d / cd_d  #TODO: just downstream airfoil

    aoa_u = np.r_[0, 20, 40, 60]
    aoa_d = np.r_[30, 30, 30, 30]
    dd_2 = np.c_[aoa_u, aoa_d, cl, cd, clcd]

    dd = np.r_[dd_1, dd_2]
    aoa_u = np.linspace(0, 60, 61)
    aoa_d = np.linspace(0, 60, 61)
    cli = griddata((dd[:, 0], dd[:, 1]), dd[:, 2],
                   (aoa_u[None, :], aoa_d[:, None]), method='cubic')
    cdi = griddata((dd[:, 0], dd[:, 1]), dd[:, 3],
                   (aoa_u[None, :], aoa_d[:, None]), method='cubic')
    clcdi = griddata((dd[:, 0], dd[:, 1]), dd[:, 4],
                     (aoa_u[None, :], aoa_d[:, None]), method='cubic')

    return aoa_u, aoa_d, cli, cdi, clcdi


aoa_u, aoa_d, cli, cdi, clcdi = ginterp(data, 5, 3)


arr = cli

fig, ax = plt.subplots()
ax.contour(aoa_u, aoa_d, arr, 20, linewidths=0.5, colors='k')
cax = ax.contourf(aoa_u, aoa_d, arr, 20, cmap=plt.cm.viridis)
ax.plot(dd[:, 0], dd[:, 1], '+', ms=10, mew=2, color='gray')
fig.colorbar(cax, ax=ax) # draw colorbar
ax.axis('equal', adjustable='box')
ax.set_xlim(-5, 65)
ax.set_ylim(-5, 65)
sns.despine(ax=ax)


# %%

# %%

#https://matplotlib.org/users/colormapnorms.html
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


show_title = True

fig, axs = plt.subplots(6, 4, sharex=True, sharey=True, figsize=(9, 11))

for i, stagger in enumerate(staggers):
    for j, gap in enumerate(gaps):

        aoa_u, aoa_d, cli, cdi, clcdi = ginterp(data, i, j)

        ax = axs[i, 3 - j]
        if show_title:
            ax.set_title('g{0} s{1}'.format(gap, stagger),
                         fontsize='xx-small')

        arr, vmin, vmax, cmap = cli, -1., 2., plt.cm.RdBu_r
#        arr, vmin, vmax, cmap = cdi, 0., 2, plt.cm.RdBu_r
#        arr, vmin, vmax, cmap = clcdi, -2, 3.5, plt.cm.RdBu_r

        contours = np.r_[vmin:vmax + .25:.25] # + .25/2
        norm = MidpointNormalize(midpoint=0)

        ax.contour(aoa_u, aoa_d, arr, contours, linewidths=0.5, colors='k',
                   vmin=vmin, vmax=vmax, norm=norm)
        cax = ax.contourf(aoa_u, aoa_d, arr, contours, cmap=cmap,
                          vmin=vmin, vmax=vmax, norm=norm)
#        cax = fig.colorbar(cax, ax=ax) # draw colorbar
        ax.axis('equal', adjustable='box')
        ax.set_xlim(-5, 65)
        ax.set_ylim(-5, 65)
        ax.set_xticks([0, 30, 60])
        ax.set_yticks([0, 30, 60])
#        ax.axis('off')
        sns.despine(ax=ax)

axs[-1, 0].plot(dd[:, 0], dd[:, 1], '+', ms=8, mew=2, color='gray')
axs[-1, 0].set_xlabel('aoa upstream')
axs[-1, 0].set_ylabel('aoa downstream')


# %%

#https://matplotlib.org/users/colormapnorms.html
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


show_title = False

fig, ax = plt.subplots(figsize=(9, 11))

for i, stagger in enumerate(staggers):
    for j, gap in enumerate(gaps):

        aoa_u, aoa_d, cli, cdi, clcdi = ginterp(data, i, j)
        aoa_u -= 30  # center at 0
        aoa_d -= 30  # center at 0

        aoa_u -= j * 1 * 60
        aoa_d -= i * 60

        arr, vmin, vmax, cmap = cli, -1., 2., plt.cm.RdBu_r
        arr, vmin, vmax, cmap = cdi, 0., 2, plt.cm.RdBu_r
        arr, vmin, vmax, cmap = clcdi, -2, 3.5, plt.cm.RdBu_r

        contours = np.r_[vmin:vmax + .25:.25] # + .25/2
        norm = MidpointNormalize(midpoint=0)

        ax.contour(aoa_u, aoa_d, arr, contours, linewidths=0.5, colors='k',
                   vmin=vmin, vmax=vmax)
        cax = ax.contourf(aoa_u, aoa_d, arr, contours, norm=norm, cmap=cmap,
                          vmin=vmin, vmax=vmax)#, extend='both')
        # cax = ax.pcolormesh(aoa_u, aoa_d, arr, norm=norm, cmap=cmap,
        #                    vmin=vmin, vmax=vmax)#, extend='both')

cbar = fig.colorbar(cax, ax=ax, shrink=.45, ticks=contours[::4])
cbar.solids.set_edgecolor('face')
ax.axis('equal', adjustable='box')
ax.axis('off')


# %%

sns.despine(bottom=True)
for ax in axs.flatten():
    ax.axhline(0, color='gray', lw=1)
    ax.set_ylim(-2.5, 3)
    ax.set_yticks([-2, -1, 0, 1, 2, 3])
    ax.xaxis.set_tick_params(length=0, width=.5)
    ax.yaxis.set_tick_params(length=4, width=.5)  # length=5
    [tick.label.set_fontsize('xx-small') for tick in ax.xaxis.get_major_ticks()]
    [tick.label.set_fontsize('xx-small') for tick in ax.yaxis.get_major_ticks()]

# add degree symbol to angles
fig.canvas.draw()
for ax in axs[-1].flatten():
    ticks = ax.get_xticklabels()
    newticks = []
    for tick in ticks:
        tick.set_rotation(0)
        text = tick.get_text()  # remove float
        newticks.append(text + u'\u00B0')
    ax.set_xticklabels(newticks)

for ax in axs[:, 0].flatten():
    ticks = ax.get_yticklabels()
    newticks = []
    for tick in ticks:
        tick.set_rotation(0)
        text = tick.get_text()  # remove float
        newticks.append(text)
    ax.set_yticklabels(newticks)

#    fig.savefig(FIG.format('new_coeffs_{0}'.format(trial)), **FIGOPT)


# %%



fig, ax = plt.subplots()
ax.plot(dd[:, 0], dd[:, 1], 'o')
sns.despine()


# %%

fig, ax = plt.subplots()
ax.plot(single['aoa_all'], single['drag_all'], 'k')
for i, gap in enumerate(gaps):
    for j, stagger in enumerate(staggers):
        cl_d = data[1]['drag_d'][j, i, :]
        ax.plot(alphas, cl_d, label='g {0}, s {1}'.format(gap, stagger))
#        for k, alpha in enumerate(alphas):
#            cl_d = drag_d_1[i, j, k]

#ax.legend(loc='best')
sns.despine()


# %%

fig, ax = plt.subplots()
ax.plot(single['aoa_all'], single['lift_all'], 'k')
for i, stagger in enumerate(staggers):
    for j, gap in enumerate(gaps):
        cl_d = data[1]['lift_d'][i, j, :]
        ax.plot(alphas, cl_d, label='g {0}, s {1}'.format(gap, stagger))
#        for k, alpha in enumerate(alphas):
#            cl_d = drag_d_1[i, j, k]

#ax.legend(loc='best')
sns.despine()


# %%

fig, ax = plt.subplots()
for i, stagger in enumerate(staggers):
    for j, gap in enumerate(gaps):
        cd_d = data[1]['drag_d'][i, j, :]
        cl_d = data[1]['lift_d'][i, j, :]
        clcd = cl_d / cd_d
        ax.plot(alphas, clcd, label='g {0}, s {1}'.format(gap, stagger))
#        for k, alpha in enumerate(alphas):
#            cl_d = drag_d_1[i, j, k]

ax.plot(single['aoa_all'], single['lift_all'] / single['drag_all'], 'k')
ax.plot(single['aoa'], single['lift'] / single['drag'], 'sk', ms=9)

#ax.legend(loc='best')
sns.despine()


# %%

# %% Build DataFrame with all of the data for bar plots

df_list = []
for trial in [1, 2]:
    for k, aoa in enumerate(alphas):
        if trial == 1:
            aoa_u = 30
            aoa_d = aoa
            down_vary = True
            up_vary = False
        elif trial == 2:
            aoa_u = aoa
            aoa_d = 30
            down_vary = False
            up_vary = True

        for i, stagger in enumerate(staggers):
            for j, gap in enumerate(gaps):
                cl_d = data[trial]['lift_d'][i, j, k]
                cd_d = data[trial]['drag_d'][i, j, k]
                clcd_d = cl_d / cd_d

                cl_u = data[trial]['lift_u'][i, j, k]
                cd_u = data[trial]['drag_u'][i, j, k]
                clcd_u = cl_u / cd_u

                cl = (cl_d + cl_u) / 2
                cd = (cd_d + cd_u) / 2
                clcd = cl / cd

                cl_s = single['lift'][k]
                cd_s = single['drag'][k]
                clcd_s = cl_s / cd_s

                cl_s30 = single['lift30']
                cd_s30 = single['drag30']
                clcd_s30 = cl_s30 / cd_s30

                dcl_d = (cl_d - cl_s) / cl_s * 100
                dcd_d = (cd_d - cd_s) / cd_s * 100
                dclcd_d = (clcd_d - clcd_s) / clcd_s * 100

                dd = {}
                dd['cl_d'] = cl_d
                dd['cd_d'] = cd_d
                dd['clcd_d'] = clcd_d

                dd['cl_u'] = cl_u
                dd['cd_u'] = cd_u
                dd['clcd_u'] = clcd_u

                dd['cl'] = cl
                dd['cd'] = cd
                dd['clcd'] = clcd

                dd['cl_s'] = cl_s
                dd['cd_s'] = cd_s
                dd['clcd_s'] = clcd_s

                dd['cl_s30'] = cl_s30
                dd['cd_s30'] = cd_s30
                dd['clcd_s30'] = clcd_s30

                dd['dcl_d'] = dcl_d
                dd['dcd_d'] = dcd_d
                dd['dclcd_d'] = dclcd_d

                dd['aoa_d'] = aoa_d
                dd['aoa_u'] = aoa_u
                dd['down_vary'] = down_vary
                dd['up_vary'] = up_vary

                dd['stagger'] = stagger
                dd['gap'] = gap

                df_list.append(dd)

df = pd.DataFrame(df_list)


# %%


# %%

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

trial = 2
for k in np.arange(4):
    for i, stagger in enumerate(staggers):
        for j, gap in enumerate(gaps):
            cl_d = data[trial]['lift_d'][i, j, k]
            cd_d = data[trial]['drag_d'][i, j, k]
            cl_u = data[trial]['lift_u'][i, j, k]
            cd_u = data[trial]['drag_u'][i, j, k]

            ax1.plot(cd_d, cl_d, 'o', c=bmap[k])
            ax2.plot(cd_u, cl_u, 'o', c=bmap[k])

sns.despine()