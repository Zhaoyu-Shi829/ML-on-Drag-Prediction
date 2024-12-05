import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# tag five datasets and the mixed of all (hyb)
tags = ['gaus', 'pos', 'neg', 'Ex', 'Ez', 'hyb']
palette = ['#e9b03e', '#de1b15', '#989191', '#5ecce3', '#3e70e9', '#652ADE', '#f23b27']
labels = [r'$I:Sk_0$', r'$II:Sk_{pos}$', r'$III:Sk_{neg}$', r'$IV:\lambda_x$', r'$V:\lambda_z$']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def configure_plots():
    rc = {
        'font.family': 'sans-serif',
        'font.serif': 'Times New Roman',
        'mathtext.fontset': 'stix',
        'font.size': 20
    }
    plt.rcParams.update(rc)

