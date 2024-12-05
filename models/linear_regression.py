import os
import sys
import random
import numpy as np
from numpy.linalg import inv
import matplotlib
from scipy import stats
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd
from sklearn.metrics import r2_score

rc = {
      'font.family': 'serif', 'font.serif': 'Times New Roman',
      'mathtext.fontset': 'stix', 'font.size': 22
}
plt.rcParams.update(rc)

no_stats = 19
no_sd = 42
cls = -1
uplus_pos = -2  # hybrid data -2; single data -1
save_pred_dir = os.path.join('./pred_dns_data/lin_sd42_{}.csv').format(no_stats)
save_weights_dir = os.path.join('./pred_dns_data/weights_lin_{}.csv').format(no_stats)
def main():
    # --------------PRE-PROCESS DATA------------------ #
    base_dir = './Copy_Data/new_data_five_types/10primary_stats'
    tags = ['gaus', 'pos', 'neg', 'Ex', 'Ez', 'hyb']
    xtick = [r'$I:Sk_0$', r'$II:Sk_{pos}$', r'$III:Sk_{neg}$', r'$IV:\lambda_x$', r'$V:\lambda_z$']

    # load hybrid data
    stats_dir = os.path.join(base_dir, 'stats_{}.npy'.format(tags[cls]))
    uplus_dir = os.path.join(base_dir, 'uplus_{}.npy'.format(tags[cls]))
    stats_data = np.squeeze(np.load(stats_dir))
    uplus_data = np.squeeze(np.load(uplus_dir))
    # -------Load Dataset-------
    data_one = pd.DataFrame(np.column_stack((stats_data, uplus_data)))
    train_one, test_one = train_test_split(data_one, test_size=0.2, random_state=no_sd)
    x_train = train_one.iloc[:, :no_stats].to_numpy().astype('float32')
    x_test = test_one.iloc[:, :no_stats].to_numpy().astype('float32')
    y_train = train_one.iloc[:, uplus_pos].to_numpy().astype('float32')
    y_test = test_one.iloc[:, uplus_pos].to_numpy().astype('float32')

    print('-------' + os.path.join('data type: ' + tags[cls]) + '-------')
    print(f'train: {train_one.shape}, test: {test_one.shape}')
    # save train+test dataset index in dataframe
    # np.save('./Copy_Data/heesoo_cnn_test/ind_{}'.format(tags[i]), train_single.index)
    pred, err, r2, coefs = regression(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # --------Plot 45 diag--------
    if cls == -1:
        tag = test_one.iloc[:, -1]
        # dns_pred_tag = np.stack((np.squeeze(y_test), np.squeeze(pred), tag)).T
        save_pred = {"pred": np.squeeze(pred), "dns": np.squeeze(y_test), "tag": tag}
        df = pd.DataFrame(save_pred)
        plot_dns_pred_hyb(data=df.to_numpy(), err=err, tag=tags, label=xtick)
        # df.to_csv(save_pred_dir, index=False)

    # --------Plot Error--------
    # plot_err(Err=err_list[i], ind=i, tags=tags[:-1], xtick=xtick)
    # -------Plot Weights--------
    # plot_weights(wgt=coef_list[i], ind=i)

    # plt.xticks([0, 1, 2, 3, 4], xtick_err, fontsize=26)
    # plt.xticks([9, 18], [r'$I_{10}$', r'$I_{19}$'])
    plt.tight_layout()
    plt.show()

def regression(x_train, y_train, x_test, y_test):
    reg = LinearRegression().fit(x_train, y_train)
    print(f'coefs: {reg.coef_}, interp: {reg.intercept_}')
    coefs, intercept = reg.coef_, reg.intercept_
    weights = pd.DataFrame(np.append(coefs, intercept))
    weights.to_csv(save_weights_dir, index=False, header=['coefs'])
    y_pred = reg.predict(x_test)
    comparison = np.stack((np.squeeze(y_pred), np.squeeze(y_test))).T
    # print(f'prediction and ground-truth: \n {comparison}')

    err = 100*(np.abs(np.squeeze(y_pred)-np.squeeze(y_test))/np.squeeze(y_test))
    R2 = r2_score(np.squeeze(y_test), np.squeeze(y_pred))
    print(f'mean err: {np.mean(err)}, max err: {np.max(err)}')
    return y_pred, err, R2, np.abs(reg.coef_)

def plot_dns_pred_hyb(data, err, tag, label):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(r'$\Delta U^+_{dns}$', fontsize=25)
    ax.set_ylabel(r'$\Delta U^+_{LR_{21}}$', fontsize=25)
    ax.set_ylim(0, 9)
    ax.set_xlim(0, 9)
    ax.set_xticks(np.arange(0, 10, 1))
    # palette = ['#e9963e', '#B80F0A', '#65a9d7', '#652ADE', '#304f9e', '#00145A', '#f23b27']
    palette = ['#e9b03e', '#de1b15', '#989191', '#5ecce3', '#3e70e9', '#652ADE', '#f23b27']
    ax.plot(np.arange(0, 10), np.arange(0, 10), color='black', ls='--', lw=2.5)
    split_lst = [[] for _ in range(5)]
    for i in range(5):
        split_lst[i] = [item[0:2].astype('float') for item in data if str(item[-1]) == tag[i]]
        ax.scatter(np.squeeze(split_lst[i])[:, 0], np.squeeze(split_lst[i])[:, 1], c=palette[i], s=50, alpha=0.8,
                   label=label[i])
    ax.text(0.5, 8, os.path.join(r'$Err=$' + '{:.2f}%'.format(np.mean(err))), fontsize=24)
    ax.text(0.5, 7, os.path.join(r'$R_2=$' + '{:.2f}'.format(r2_score(data[:, 0], data[:, 1]))), fontsize=24)
    # plt.legend(framealpha=0.0, fontsize=24, loc=4, handletextpad=0.1)

def plot_weights(wgt, ind):
    plt.gcf().set_size_inches(8, 4)
    plt.ylabel(r'$w_i$', fontsize=35)
    palette = ['#e9963e', '#f23b27', '#65a9d7', '#304f9e', '#00145A']
    plt.plot(np.arange(no_stat), np.squeeze(wgt), marker='o', ls='--', color=palette[ind])

def plot_err(Err, ind, tags, xtick):
    sns.set_style('whitegrid')
    plt.gcf().set_size_inches(10, 8)
    plt.ylabel(r'$Err\%$', fontsize=35)

    palette = ['#e9963e', '#f23b27', '#65a9d7', '#304f9e']
    # palette = ['#FF2709', '#0030D7', '#01FBEE', '#09FF10', '#FA70B5', '#00145A']
    boxprops = dict(lw=2.5, color='#00145A')
    flierprops = dict(marker='o', ms=2, ls='none')
    whiskerprops = dict(lw=2.5, color='#00145A')
    capprops = dict(lw=2.5, color='#00145A')
    meanpointprops = dict(marker='D', ms=6)
    meanlineprops = dict(ls='--', lw=5)

    position = np.arange(5)
    x = np.random.normal(ind, 0.04, len(np.squeeze(Err)))
    if ind == 4:
        plt.boxplot(np.squeeze(Err)[:, 0].astype('float64'), notch=False, showmeans=True, meanline=True, widths=0.4,
                    showfliers=True, positions=[ind], meanprops=meanpointprops, capprops=capprops, boxprops=boxprops,
                    whiskerprops=whiskerprops)
        plt.text(position[ind] + 0.24, np.mean(np.squeeze(Err)[:, 0].astype('float64'))+0.25,
                 '{:.2f}'.format(np.mean(np.squeeze(Err)[:, 0].astype('float64'))))
        sca_list = [[] for _ in range(4)]
        for i in range(4):
            sca_list[i] = [item[0].astype('float64') for item in np.squeeze(Err) if str(item[1]) == tags[i]]
        x_split = np.split(x, [len(sca_list[0]), len(sca_list[0])+len(sca_list[1]),
                               len(sca_list[0])+len(sca_list[1])+len(sca_list[2])])
        [plt.scatter(x_split[i], sca_list[i], color=palette[i], alpha=0.5) for i in range(4)]
    else:
        plt.boxplot(Err, notch=False, showmeans=True, meanline=True, showfliers=True, widths=0.4,
                    positions=[ind], meanprops=meanpointprops, capprops=capprops, boxprops=boxprops,
                    whiskerprops=whiskerprops)
        plt.scatter(x, Err, color=palette[ind], alpha=0.5, label=xtick[ind])
        plt.text(position[ind] + 0.24, np.mean(Err) + 0.25, '{:.2f}'.format(np.mean(Err)))
        plt.legend(framealpha=0.0, fontsize=28)


if __name__ == "__main__":
    main()

