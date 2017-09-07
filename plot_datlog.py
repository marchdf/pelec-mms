#!/usr/bin/env python3
#
#
"""@package plotting

"""
__author__ = 'Marc T. Henry de Frahan'

#=========================================================================
#
# Imports
#
#=========================================================================
import argparse
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axis as axis
import pandas as pd


#=========================================================================
#
# Parse arguments
#
#=========================================================================
parser = argparse.ArgumentParser(
    description='A simple plot tool for the Taylor-Green vortex')
parser.add_argument('-s', '--show', help='Show the plots', action='store_true')
args = parser.parse_args()


#=========================================================================
#
# Some defaults variables
#
#=========================================================================
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
cmap_med = ['#F15A60', '#7AC36A', '#5A9BD4', '#FAA75B',
            '#9E67AB', '#CE7058', '#D77FB4', '#737373']
cmap = ['#EE2E2F', '#008C48', '#185AA9', '#F47D23',
        '#662C91', '#A21D21', '#B43894', '#010202']
dashseq = [(None, None), [10, 5], [10, 4, 3, 4], [
    3, 3], [10, 4, 3, 4, 3, 4], [3, 3], [3, 3]]
markertype = ['s', 'd', 'o', 'p', 'h']

#=========================================================================
#
# Function definitions
#
#=========================================================================


#=========================================================================
#
# Problem setup
#
#=========================================================================

fname = 'datlog'
df = pd.read_csv(fname, delim_whitespace=True)
cnt = 0

plt.figure(cnt)
p = plt.plot(df['time'], df['rho_K'])
cnt += 1

plt.figure(cnt)
p = plt.plot(df['time'], df['rho_e'])
cnt += 1

plt.figure(cnt)
p = plt.plot(df['time'], df['rho_E'])
cnt += 1

plt.show()
