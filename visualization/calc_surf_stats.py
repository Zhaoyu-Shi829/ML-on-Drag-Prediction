import math
import os
import sys
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import scipy.stats
from scipy.stats import spearmanr, pearsonr
import scipy.integrate as integrate
from scipy import signal
import seaborn as sns
from heatmap import heatmap, corrplot
from matplotlib import cm

rc = {
      'font.family': 'serif', 'font.serif': 'Times New Roman',
      'mathtext.fontset': 'stix', 'font.size': 20
}
plt.rcParams.update(rc)

def calc_integral_length(data, dx, dz):
    # calculate autocorrelation in x/z axis
    var = np.var(data)
    mean = np.mean(data)
    ndata = data - mean
    lags_x = signal.correlation_lags(ndata.shape[1], ndata.shape[1], mode='full')[ndata.shape[1]-1:] * dx
    lags_z = signal.correlation_lags(ndata.shape[0], ndata.shape[0], mode='full')[ndata.shape[0]-1:] * dz
    Rx, Rz = [], []
    for line in ndata:
        acorr_x = signal.correlate(line, line, mode='full')[len(line)-1:]
        acorr_x = acorr_x/var/len(line)
        Rx.append(acorr_x)
    Rxx = np.mean(np.array(Rx), axis=0)
    for line in ndata.T:
        acorr_z = signal.correlate(line, line, mode='full')[len(line)-1:]
        acorr_z = acorr_z/var/len(line)
        Rz.append(acorr_z)
    Rzz = np.mean(np.array(Rz), axis=0)
    # calculate integral length scale
    L_x = integrate.simpson(Rxx, lags_x, even='avg')
    L_z = integrate.simpson(Rzz, lags_z, even='avg')
    return L_x, L_z

def calc_stats(surf):
    '''
        'surf.npy' is friction-scaled, h_w=h/wall_scale=h*Re_tau, h/H=h_w/Re_tau
         make sure all stats are dimensionaless
    '''
    Re_tau = 500
    # height_md = 0.9327808396033839
    # u_tau = 0.1
    channel_height = 1
    wall_scale = channel_height / Re_tau
    # print(f'wall_scale: {wall_scale}')

    lx, lz, ly = 2.4, 0.8, 2.0
    nx, nz = surf.shape[1], surf.shape[0]
    dx, dz = lx / nx, lz / nz

    k = surf
    kavg = np.mean(k)
    Ra = np.mean(np.abs(k - kavg))
    krms = np.sqrt(np.mean((k - kavg)**2))
    kc = np.max(k) - np.min(k)

    skw = np.mean((k - kavg) ** 3) / krms ** 3
    kur = np.mean((k - kavg) ** 4) / krms ** 4
    Po = (1 - np.sum(k)/(nx * nz * kc))

    dkdx_p1 = np.roll(k, 1, axis=1)
    dkdx_n1 = np.roll(k, -1, axis=1)
    dkdx = (dkdx_p1 - dkdx_n1) / (2 * dx)
    ESx = np.mean(np.abs(dkdx)) * wall_scale

    dkdz_p1 = np.roll(k, 1, axis=0)
    dkdz_n1 = np.roll(k, -1, axis=0)
    dkdz = (dkdz_p1 - dkdz_n1) / (2 * dz)
    ESz = np.mean(np.abs(dkdz)) * wall_scale

    dkdx_mean, dkdz_mean = np.mean(dkdx), np.mean(dkdz)
    dkdx_rms, dkdz_rms = np.sqrt(np.mean((dkdx-dkdx_mean)**2)), np.sqrt(np.mean((dkdz-dkdz_mean)**2))
    dkdx_skw, dkdz_skw = np.mean((dkdx-dkdx_mean)**3)/dkdx_rms**3, np.mean((dkdz-dkdz_mean)**3)/dkdz_rms**3
    incx = np.arctan(0.5 * dkdx_skw)
    incz = np.arctan(0.5 * dkdz_skw)

    L_x, L_z = calc_integral_length(data=k, dx=dx, dz=dz)

    I11 = ESx * ESz
    I12 = ESx * skw
    I13 = ESx * kur
    I14 = ESz * skw
    I15 = ESz * kur
    I16 = skw * kur
    I17 = ESx * ESx
    I18 = ESz * ESz
    I19 = skw * skw

    return kc, krms, Ra, skw, kur, ESx, ESz, Po, incx, incz, I11, I12, I13, I14, I15, I16, I17, I18, I19


# save stats with surf class tag
tags = ['gaus', 'pos', 'neg', 'Ex', 'Ez']
def calc_save_stats():
    base_dir = './Copy_Data'
    src_dir = ['Data_Gaus_new', 'Data_Pos_skw_new', 'Data_Neg_skw_new', 'Data_Ex', 'Data_Ez']
    surf_file = [sorted(glob(os.path.join(base_dir, src_dir[i], 'surf/surf_*.npy'))) for i in range(len(tags))]
    uplus_file = [sorted(glob(os.path.join(base_dir, src_dir[i], 'uplus/uplus_*.npy'))) for i in range(len(tags))]

    stats_list, uplus_list, stats_hyb, uplus_hyb = [[] for _ in range(4)]
    no_samp = []
    for i in range(len(src_dir)):
        print('----no. of surf----', i)
        tmp_stats, tmp_uplus = [], []
        # ind = np.load('./Copy_Data/heesoo_cnn_test/train/ind_{}.npy'.format(tags[i]))
        for m, src in enumerate(surf_file[i]):
            surf = np.load(src)
            kc, krms, Ra, skw, kur, ESx, ESz, Po, incx, incz,\
            I11, I12, I13, I14, I15, I16, I17, I18, I19 = calc_stats(surf)
            stats_list.append([kc, krms, Ra, skw, kur, ESx, ESz, Po, incx, incz,
                               I11, I12, I13, I14, I15, I16, I17, I18, I19, tags[i]])
            tmp_stats.append([kc, krms, Ra, skw, kur, ESx, ESz, Po, incx, incz,
                              I11, I12, I13, I14, I15, I16, I17, I18, I19])
            # if m in ind:
            #     shutil.copy(src, './Copy_Data/heesoo_cnn_test/surf_{}'.format(tags[i]))
        for m, src in enumerate(uplus_file[i]):
            uplus = np.load(src)
            if np.isnan(uplus) == True:
                uplus = np.nan_to_num(4.7600)       # replace the ill data with right value (Seyed gives)
            uplus_list.append([uplus, tags[i]])
            tmp_uplus.append(uplus)
            # if m in ind:
            #     shutil.copy(src, './Copy_Data/heesoo_cnn_test/uplus_{}'.format(tags[i]))

        no_samp.append(len(tmp_uplus))
        # np.save(os.path.join(base_dir, 'new_data_five_types/10primary_stats/stats_{}.npy'.format(tags[i])), tmp_stats)
        # np.save(os.path.join(base_dir, 'new_data_five_types/uplus_{}.npy'.format(tags[i])), tmp_uplus)

    stats_hyb.append(stats_list)
    uplus_hyb.append(uplus_list)
    stats_hyb = np.squeeze(np.array(stats_hyb))
    uplus_hyb = np.squeeze(np.array(uplus_hyb))

    # np.save('./Copy_Data/new_data_five_types/10primary_stats/stats_hyb.npy', stats_hyb)
    # np.save('./Copy_Data/new_data_five_types/uplus_hyb.npy', uplus_hyb)

# calc_save_stats()


def plot_U_stats():
    base_dir = './Copy_Data/new_data_five_types/10primary_stats'
    stats_src_dir = [os.path.join(base_dir, 'stats_{}.npy').format(i) for i in tags]
    uplus_src_dir = [os.path.join(base_dir, 'uplus_{}.npy').format(i) for i in tags]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    ec = ['#e9b03e', '#de1b15', '#989191', '#5ecce3', '#3e70e9', '#652ADE', '#f23b27']
    xlabel = [r'$k_c$', r'$k_{rms}$', r'$Skw$', r'$Ku$', r'$ES_x$', r'$ES_z$', r'$Inc_x$', r'$Inc_z$']
    stats_id = [0, 1, 3, 4, 5, 6, 8, 9]
    type = [r'$I:Sk_0$', r'$II:Sk_+$', r'$III:Sk_-$', r'$IV:\lambda_x$', r'$V:\lambda_z$']
    for k in range(5):
        stats_data = np.load(stats_src_dir[k])
        uplus_data = np.load(uplus_src_dir[k])
        trans_no = len([i for i in uplus_data if i <= 6.3])
        fully_no = len(uplus_data) - trans_no
        print(f'no. trans rough: {trans_no}, no. fully rough: {fully_no}')
        print(np.min(uplus_data), np.max(uplus_data))
        data = np.column_stack((stats_data, uplus_data))
        print(f'type: {k}, isotropy: {np.mean(data[:, 6]/data[:, 5])}')
        for i, ax in enumerate(fig.axes):
            ax.locator_params(axis='x', nbins=6)
            if i in range(len(stats_id)):
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.set_xlabel(xlabel[i], fontsize=20)
                ax.scatter(data[:, stats_id[i]], data[:, -1], c='None', edgecolors=ec[k], lw=2, s=22, alpha=0.3, label=type[k])
            if i in [0, 4]:
                ax.set_ylabel(r'$\Delta U^+$')
            if i in [2, 3, 5, 6, 7]:
                ax.axes.get_yaxis().set_visible(False)
            if i in [2]:
                ax.set_xlim(-2, 2)
            if i in [7]:
                ax.set_xlim(-1, 1)
            plt.subplots_adjust(wspace=None, hspace=None)

    leg = plt.legend(scatterpoints=1, frameon=False, columnspacing=0.2, handletextpad=0.1, ncol=1, fontsize=12, loc=3)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.3)
    # plt.savefig('/lscratch/zhaoyus/proj/cnn/My_CNN/plots/stats_hyb_data_2.jpeg', dpi=600)
    # plt.show()

# plot_U_stats()

def plot_single_U():
    base_dir = './Copy_Data/new_data_five_types/10primary_stats'
    stats_src_dir = [os.path.join(base_dir, 'stats_{}.npy').format(i) for i in tags]
    uplus_src_dir = [os.path.join(base_dir, 'uplus_{}.npy').format(i) for i in tags]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # [ax[i, j].set_rasterized(True) for i in (0, 1) for j in (0, 1)]
    ax[0, 0].set_ylabel(r'$\Delta U^+$')
    ax[1, 0].set_ylabel(r'$\Delta U^+$')
    [ax[i, j].set_ylim(0, 9) for i in (0, 1) for j in (0, 1)]

    ec = ['#e9b03e', '#de1b15', '#989191', '#5ecce3', '#3e70e9', '#652ADE', '#f23b27']
    xlabel = [r'$k^+_{rms}$', r'$ES_x$', r'$Skw$', r'$Ku$']
    type = [r'$I:Sk_0$', r'$II:Sk_{+}$', r'$III:Sk\_$', r'$IV:\lambda_x$', r'$V:\lambda_z$']
    ax[0, 0].set_xlabel(xlabel[0], fontsize=22)
    ax[0, 0].set_xlim(1.5, 13.5)
    ax[1, 0].set_xlabel(xlabel[1], fontsize=22)
    ax[1, 0].set_xlim(0.1, 0.9)
    ax[0, 1].set_xlabel(xlabel[2], fontsize=22)
    ax[0, 1].set_xlim(-2, 2)
    ax[1, 1].set_xlabel(xlabel[3], fontsize=22)
    ax[1, 1].set_xlim(2, 8)
    ax[0, 1].yaxis.set_ticklabels([])
    ax[1, 1].yaxis.set_ticklabels([])
    for k in range(5):
        stats_data = np.load(stats_src_dir[k])
        uplus_data = np.load(uplus_src_dir[k])
        data = np.column_stack((stats_data, uplus_data))

        ax[0, 0].scatter(data[:, 1], data[:, -1], c='None', edgecolors=ec[k], lw=2, s=22, alpha=0.3,
                         label=type[k])
        ax[0, 1].scatter(data[:, 3], data[:, -1], c='None', edgecolors=ec[k], lw=2, s=22, alpha=0.3,
                         label=type[k])
        ax[1, 0].scatter(data[:, 5], data[:, -1], c='None', edgecolors=ec[k], lw=2, s=22, alpha=0.3,
                         label=type[k])
        ax[1, 1].scatter(data[:, 4], data[:, -1], c='None', edgecolors=ec[k], lw=2, s=22, alpha=0.3,
                         label=type[k])
    ax[0, 0].plot([3.2, 6], [2.7, 6.2], [6, 11], [4.0, 6.5], ls='--', color='k')
    circle = plt.Circle((5.3, 4.5), 0.8, ls='--', lw=1.6, color='k', fill=False)
    ax[1, 1].add_patch(circle)

    leg = plt.legend(scatterpoints=1, frameon=False, columnspacing=0.1, handletextpad=0.1, ncol=5, fontsize=22,
                     loc='upper center', bbox_to_anchor=(-0.1, 2.58), markerscale=3)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.12, hspace=0.3)
    plt.savefig('./Figures/draft/scatter_params.jpeg', format='jpeg', dpi=600)
    plt.show()

# plot_single_U()


def plot_bivariate_stats():
    # kc, krms, Ra, skw, kur, ESx, ESz, Po, incx, incz
    # Ex*Ez[10], Ex*Sk[11], Ez*Sk[13], Ex*Ex[17], Ez*Ez[18]
    base_dir = './Copy_Data/new_data_five_types/10primary_stats'
    ec = ['#e9963e', '#B80F0A', '#65a9d7', '#652ADE', '#304f9e', '#00145A', '#f23b27']
    # ec = ['#e9963e', '#f23b27', '#65a9d7', '#304f9e']
    xtick = [r'$k_c$', r'$Ra$', r'$k_{rms}$', r'$Sk$', r'$Kur$', r'$E_x$', r'$E_z$', r'$Po$']
    # xtick = [r'$E_x*E_z$', r'$E_x*Sk$', r'$E_z*Sk$', r'$E_x*E_x$', r'$E_z*E_z$']

    stats_dir = [os.path.join(base_dir, 'stats_{}.npy').format(tags[i]) for i in range(5)]
    fig, ax = plt.subplots(1, 5, figsize=(14, 4), constrained_layout=True)
    [ax[i].get_yaxis().set_visible(False) for i in [1, 2, 3, 4]]
    [ax[i].set_xlabel(xtick[i]) for i in range(5)]
    [ax[i].tick_params(axis='both', which='major', labelsize=16) for i in range(5)]

    for i in range(len(stats_dir)):
        data = np.load(stats_dir[i])
        krms, skw, kurt, Ex, Ez, Po = data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 7]
        print(i, np.min(data[:, 8]), np.max(data[:, 8]), np.min(data[:, 9]), np.max(data[:, 9]))
        I10, I11, I13, I17, I18 = data[:, 10], data[:, 11], data[:, 13], data[:, 17], data[:, 18]
        ax[0].scatter(I10, krms, color=ec[i], s=22, alpha=0.4)
        ax[0].set_ylabel(r'$k_{rms}$', fontsize=26)
        ax[1].scatter(I11, krms, color=ec[i], s=22, alpha=0.4)
        ax[2].scatter(I13, krms, color=ec[i], s=22, alpha=0.4)
        ax[3].scatter(I17, krms, color=ec[i], s=22, alpha=0.4)
        ax[4].scatter(I18, krms, color=ec[i], s=22, alpha=0.4)

    plt.tight_layout()
    plt.show()

# plot_bivariate_stats()


def plot_correlation_map():
    # kc, krms, Ra, skw, kur, ESx, ESz, Po, incx, incz
    cls = 4
    Nstats = 10
    base_dir = './Copy_Data/new_data_five_types/10primary_stats'
    stats_dir = [os.path.join(base_dir, 'stats_{}.npy').format(tags[i]) for i in range(5)]
    uplus_dir = [os.path.join(base_dir, 'uplus_{}.npy').format(tags[i]) for i in range(5)]

    stats = np.load(stats_dir[cls])[:, :Nstats]
    uplus = np.load(uplus_dir[cls])
    data = np.column_stack((stats, uplus))
    columns = [r'$k^+_c$', r'$k^+_{rms}$', r'$Ra^+$', r'$Skw$', r'$Ku$', r'$ES_x$', r'$ES_z$', r'$Po$',
               r'$Inc_x$', r'$Inc_z$', r'$\Delta U^+$']
    df = pd.DataFrame(data, columns=columns)
    # corr = df[columns].corr(method='pearson')
    corr = df[columns].corr(method='spearman')
    corr_copy = pd.melt(corr.reset_index(), id_vars='index')
    corr_copy.columns = ['x', 'y', 'value']
    # mask and melt correlation matrix
    mask = np.tril(np.ones_like(corr, dtype=bool))
    mask[np.diag_indices_from(mask)] = False
    np.fill_diagonal(mask, False)
    corr_melt = corr.mask(mask).melt(ignore_index=False).reset_index()
    corr_melt.columns = ['x', 'y', 'value']
    # print('melt', corr_melt.to_string())
    corr_melt = corr_melt.dropna()

    # linear_corr = np.array([pearsonr(data[:, i], data[:, j]) for i in range(Nstats) for j in range(Nstats)])[:, 0]
    # nonlinear_corr = np.array([spearmanr(data[:, i], data[:, j]) for i in range(Nstats) for j in range(Nstats)])[:, 0]
    # df['columns'] = np.random.randint(0, 2, size=len(df))
    def heatmap(x, y, size):
        fig = plt.figure(figsize=(10, 10))
        plot_grid = plt.GridSpec(1, 20, hspace=0.2, wspace=0.5)
        ax = plt.subplot(plot_grid[:, :-1])
        ax.spines[['right', 'top']].set_visible(False)
        type_name = [r'$I:Sk_0$', r'$II:Sk_{+}$', r'$III:Sk\_$', r'$IV:\lambda_x$', r'$V:\lambda_z$']
        ax.text(5, 5, "Type"+" "+type_name[cls], rotation=-45, fontsize=38)

        # Mapping from column names to integer coordinates
        x_labels = [v for v in x.unique()]
        y_labels = [v for v in y[::-1].unique()]
        x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
        y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

        color_min, color_max = [-1, 1]
        n_colors = 20
        palette = sns.diverging_palette(230, 16, 60, 60, n=n_colors)
        def value_to_color(val):
            val_position = float((val - color_min)) / (color_max - color_min)
            ind = int(val_position * (n_colors - 1))
            return palette[ind]

        size_scale = 1500
        ax.scatter(
            x=x.map(x_to_num),      # Use mapping for x
            y=y.map(y_to_num),      # Use mapping for y
            s=size * size_scale,    # Vector of square sizes, proportional to size parameter
            c=corr_melt['value'].apply(value_to_color),
            marker='o',
        )

        # Annotate upper triangle
        pd = corr.mask(mask).melt(ignore_index=False).reset_index()
        pd.columns = ['x', 'y', 'value']
        x_ind = pd['x'].map(x_to_num)
        y_ind = pd['y'].map(y_to_num)
        for index, row in pd.iterrows():
            val = row['value']
            x = x_ind.iloc[index]
            y = y_ind.iloc[index]
            if math.isnan(val) == False and 0.5 < abs(val) < 1.0:
                ax.text(x, y, f"{val:.2f}", size=20, ha="center", va="center", weight="bold",
                        color='black')

        # Show column labels on the axes
        ax.set_xticks([x_to_num[v] for v in x_labels])
        ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='center')
        ax.set_yticks([y_to_num[v] for v in y_labels])
        ax.set_yticklabels(y_labels)
        ax.axes.get_yaxis().set_visible(True)

        # Add colorbar to the right side

        ax = plt.subplot(plot_grid[:, -1])
        col_x = [0]*len(palette)
        bar_y = np.linspace(color_min, color_max, n_colors)
        bar_height = bar_y[1] - bar_y[0]
        ax.barh(y=bar_y, width=[2]*len(palette), left=col_x, height=bar_height, color=palette, lw=0, edgecolor='white')
        ax.set_xlim(1, 2)
        ax.set_ylim(-1, 1)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
        ax.yaxis.tick_right()
        ax.set_facecolor('white')
        ax.text(3.3, -0.03, r'$\rho_{i, j}$', rotation=90, fontsize=36)

    heatmap(x=corr_melt['x'], y=corr_melt['y'], size=corr_melt['value'].abs())
    # plt.figure(figsize=(8, 8))
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # np.fill_diagonal(mask, False)
    # hmp = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, fmt='.2f', annot_kws={"size": 9, "fontweight": 500}, mask=mask,
    #                   cmap=sns.diverging_palette(230, 20, n=256, as_cmap=True), cbar_kws={"shrink": .5},
    #                   square=True)
    # hmp.set_xticklabels(hmp.get_xticklabels(), rotation=45, horizontalalignment='center', fontsize=16)
    # hmp.set_yticklabels(hmp.get_yticklabels(), fontsize=16)
    # corrplot(corr, size_scale=500, marker='s')
    cls_name = ["Sk0", "Sk+", "Sk-", "Ex", "Ez"]
    plt.savefig(os.path.join("./Figures/draft", cls_name[cls]+"_corr.jpeg"), dpi=300)
    plt.show()

# plot_correlation_map()


def plot_kde():
    cls = -1
    tags = ['gaus', 'pos', 'neg', 'Ex', 'Ez', 'hyb']
    base_dir = './Copy_Data/new_data_five_types/10primary_stats'
    stats_dir = os.path.join(base_dir, 'stats_{}.npy'.format(tags[cls]))
    uplus_dir = os.path.join(base_dir, 'uplus_{}.npy'.format(tags[cls]))
    stats_data = np.squeeze(np.load(stats_dir))
    uplus_data = np.squeeze(np.load(uplus_dir))

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    # for i, j in ax:
    #     ax[i, j].set_xticklabels(fontsize=10)
    # krms, Ex, Ez, Skw
    x_range, kde = [], []
    for i in [1, 3, 5, 6]:
        data = stats_data[:, i].astype('float32')
        kde.append(scipy.stats.gaussian_kde(data))
        x_range.append(np.linspace(np.min(data), np.max(data), 200))

    ax[0, 0].plot(x_range[0], kde[0](x_range[0]), lw=2)  # krms
    ax[0, 0].set_xlabel(r'$k_{rms}^+$')
    ax[0, 1].plot(x_range[1], kde[1](x_range[1]), lw=2)  # Ex
    ax[0, 1].set_xlabel(r'$Skw$')
    ax[1, 0].plot(x_range[2], kde[2](x_range[2]), lw=2)  # Ez
    ax[1, 0].set_xlabel(r'$E_x$')
    ax[1, 1].plot(x_range[3], kde[3](x_range[3]), lw=2)  # Skw
    ax[1, 1].set_xlabel(r'$E_z$')
    plt.tight_layout()
    plt.show()

plot_kde()
