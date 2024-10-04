import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import collections
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from numpy import ndarray
from scipy import stats
from scipy.interpolate import NearestNDInterpolator
from sklearn.metrics import mean_squared_error
# matplotlib.use('Agg')
import seaborn as sns
from glob import glob
# print('python version:', sys.version)
# print('python interpreter:', sys.executable)
import random
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd
from sklearn.metrics import r2_score

rc = {
      'font.family': 'serif', 'font.serif': 'Times New Roman',
      'mathtext.fontset': 'stix', 'font.size': 20
}
plt.rcParams.update(rc)

def norm_2Ddata(data, min_inp, max_inp):    # RESCALED INPUT TO [-1, 1]
    norm_data = np.copy(data)
    for i in range(len(min_inp)):
        alpha = 2/(max_inp[i]-min_inp[i])
        beta = -1-alpha*min_inp[i]
        norm_data[:, i] = alpha*data[:, i]+beta
    return norm_data

def norm_1Ddata(data, min_out, max_out):    # RESCALED DRAG TO [0, 1]
    alpha = 1/(max_out - min_out)
    beta = -alpha * min_out
    norm_data = alpha * data + beta
    return norm_data

def denorm_1Ddata(data, min, max):          # RESCALE TO PHYSICAL LENGTH
    alpha = 1 / (max - min)
    beta = -alpha * min
    denorm_data = (data - beta) / alpha
    return denorm_data

def svr_linear_model(ind):
    '''
        C: Regularization trade-off between max(margin) and min(err);
            large C(smaller margin but fewer errors)=regularization strength is inversely proportional to C
        kernel: map input into a higher-dim feature space, where linear regression can be performed
        gamma: kernel coeffs for non-linear kernel; smaller-smoother, larger-complex decision boundary
            1 / (n_features * X.var())
        epsilon: margin tolerance for prediction error
        K(x, x')=x^T x': measure of two inputs' similarity, this value     is used to compute the weights of SVR model;
            smaller gamma leads to wider kernel (more data points are considered similar to each other);
    '''
    eps = [0.1, 0.01, 0.01, 0.1, 0.01, 0.01]
    regr_linear = svm.SVR(kernel='linear', gamma='scale', C=1, epsilon=eps[ind], shrinking=False, verbose=True)
    # sensitivity: C little influence
    return regr_linear

def svr_rbf_model(ind):
    # K(x, x')=exp(-gamma ||x-x'||^2)
    # min-max norm: epsilon=0.01, C=1; std-norm: C=1000~5000
    eps = [0.04, 0.001, 0.01, 0.001, 0.04, 0.01]
    regr_rbf = svm.SVR(kernel='rbf', gamma='scale', C=10, epsilon=eps[ind], shrinking=False, verbose=True)
    return regr_rbf

def svr_poly_model(ind):
    # K(x, x')=(gamma x^T x' + r)^d
    eps = [0.1, 0.004, 0.01, 0.01, 0.1, 0.1]
    regr_poly = svm.SVR(kernel='poly', degree=3, gamma='scale', C=1, epsilon=eps[ind], shrinking=False, verbose=True)
    return regr_poly

# -------------------------------------------------------------------------
val_flag = True
test_flag = True
no_sd = 42
no_stats = 10
cls = -1
save_pred_dir = os.path.join('./pred_dns_data/SVR_rbf_sd42_{}.csv').format(no_stats)
def main():
    base_dir = './Copy_Data/new_data_five_types/10primary_stats'
    tags = ['gaus', 'pos', 'neg', 'Ex', 'Ez', 'hyb']
    xtick = [r'$I:Sk_0$', r'$II:Sk_+$', r'$III:Sk\_$', r'$IV:\lambda_x$', r'$V:\lambda_z$']

    regr_linear, regr_rbf, regr_poly = svr_linear_model(ind=cls), svr_rbf_model(ind=cls), svr_poly_model(ind=cls)
    stats_dir = os.path.join(base_dir, 'stats_{}.npy'.format(tags[cls]))
    uplus_dir = os.path.join(base_dir, 'uplus_{}.npy'.format(tags[cls]))
    stats_data = np.squeeze(np.load(stats_dir))
    uplus_data = np.squeeze(np.load(uplus_dir))
    # hybrid data split
    data_one = pd.DataFrame(np.column_stack((stats_data, uplus_data)))
    dev_one, test_one = train_test_split(data_one, test_size=0.2, random_state=no_sd)
    train_one, val_one = train_test_split(dev_one, test_size=0.2, random_state=no_sd)
    x_train = train_one.iloc[:, :no_stats].to_numpy().astype('float32')
    y_train = train_one.iloc[:, -2].to_numpy().astype('float32')
    x_val = val_one.iloc[:, :no_stats].to_numpy().astype('float32')
    y_val = val_one.iloc[:, -2].to_numpy().astype('float32')
    x_test = test_one.iloc[:, :no_stats].to_numpy().astype('float32')
    y_test = test_one.iloc[:, -2].to_numpy().astype('float32')
    #--------there is one duplicate data for negative surfs-----------
    seen_y = set()
    doub_y = [x for x in y_test if x not in seen_y and not seen_y.add(x)]
    seen_x = set()
    doub_x = [x for x in x_test[:, 1] if x not in seen_x and not seen_x.add(x)]
    #------------------------------------------------------------------
    tag_test = test_one.iloc[:, -1].to_numpy().astype('str')
    print('-------' + os.path.join('data type: ' + tags[cls]) + '--------')
    print(f'train: {train_one.shape}, val: {val_one.shape}, test: {test_one.shape}')
    #---------------------FOR REVIEWER 2--------------------
    # -----------NearestNeighbor Interpolator---------------------
    interp = NearestNDInterpolator(x_train, y_train)
    y_pred_NN = interp(x_test)
    mape_NN, err_gaus, err_pos, err_neg, err_Ex, err_Ez = [], [], [], [], [], []
    for item_a, item_b in zip(y_pred_NN, y_test):
        err = 100 * np.abs(item_b - item_a) / item_b
        mape_NN.append(err)
    mape_NN_tag = list(zip(mape_NN, tag_test))
    for value in mape_NN_tag:
        if value[1] == 'gaus':
            err_gaus.append(value[0])
        if value[1] == 'pos':
            err_pos.append(value[0])
        if value[1] == 'neg':
            err_neg.append(value[0])
        if value[1] == 'Ex':
            err_Ex.append(value[0])
        if value[1] == 'Ez':
            err_Ez.append(value[0])
    print(f'gaus:{np.mean(err_gaus)}, pos:{np.mean(err_pos)}, neg:{np.mean(err_neg)}, '
          f'Ex:{np.mean(err_Ex)}, Ez:{np.mean(err_Ez)}')
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=0.2, bottom=0.2)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel(r'$k_{rms}^+$', fontsize=20)
    ax.set_ylabel(r'$\Delta U^+$', fontsize=20)
    axins = zoomed_inset_axes(ax, 2.5, loc='lower right', borderpad=1.8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    for s in ['top', 'bottom', 'left', 'right']:
        axins.spines[s].set(color='grey', lw=1, ls='solid')
    axins.set(xlim=[5.4, 7.2], ylim=[0.1, 1.2])
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax.scatter(x_train[:, 1], y_train, c='None', ec='black', lw=2, s=22, alpha=0.5)
    ax.scatter(x_test[:, 1], y_test, c='None', ec='red', lw=2, s=50, alpha=0.5)
    neg_test = test_one[test_one.iloc[:, -1].to_numpy().astype('str') == "neg"]
    neg_train = train_one[train_one.iloc[:, -1].to_numpy().astype('str') == "neg"]
    axins.scatter(x_train[:, 1], y_train, c='None', ec='black', lw=2, s=22, alpha=0.5)
    axins.scatter(x_test[:, 1], y_test, c='None', ec='red', lw=2, s=50, alpha=0.5)
    plt.tight_layout()
    plt.show()

    sys.exit()
    # ------------------------------------------------------------
    # min_max norm
    x_max, x_min = np.max(x_train, axis=0), np.min(x_train, axis=0)
    x_train_norm = norm_2Ddata(x_train, x_min, x_max)
    print(f'kernel coef gamma: {1/(x_train_norm.var()*no_stats)}')
    x_val_norm = norm_2Ddata(x_val, x_min, x_max)
    print('//----Train----')
    regr_rbf.fit(x_train_norm, y_train)
    regr_linear.fit(x_train_norm, y_train)
    regr_poly.fit(x_train_norm, y_train)
    if val_flag:
        print('//----Val----')
        score_linear = regr_linear.score(x_val_norm, y_val)
        score_rbf = regr_rbf.score(x_val_norm, y_val)
        score_poly = regr_poly.score(x_val_norm, y_val)
        print(f'no_stat={no_stats}: linear_score={score_linear}, rbf_score={score_rbf}, poly_score={score_poly}')
    if test_flag:
        print('//----Test----')
        # min_max norm
        x_test_norm = norm_2Ddata(x_test, x_min, x_max)
        pred_lin, pred_rbf, pred_poly = regr_linear.predict(x_test_norm), regr_rbf.predict(x_test_norm), \
                                        regr_poly.predict(x_test_norm)
        err_lin = 100*np.abs((np.squeeze(pred_lin)-np.squeeze(y_test)))/np.squeeze(y_test)
        err_rbf = 100*np.abs((np.squeeze(pred_rbf)-np.squeeze(y_test)))/np.squeeze(y_test)
        err_poly = 100*np.abs((np.squeeze(pred_poly)-np.squeeze(y_test)))/np.squeeze(y_test)

        # --------Plot 45 diag--------
        # dns_pred_tag = np.stack((np.squeeze(y_test), np.squeeze(pred_rbf), tag_test)).T
        save_pred = {"pred": np.squeeze(pred_rbf), "dns": np.squeeze(y_test), "tag": tag_test}
        df = pd.DataFrame(save_pred)
        plot_dns_pred_hyb(data=df.to_numpy(), err=err_rbf, tag=tags, label=xtick)
        df.to_csv(save_pred_dir, index=False)

        # --------Plot Error--------
        # plot_err(Err=np.squeeze(err_list[i]), ind=i, tags=tags[:-1], xtick=xtick)

    # plt.text(-0.2, 35, r'$L:K_{linear}\ \ M:K_{rbf}\ \ R:K_{poly}$', fontsize=24)
    # plt.hlines(15, xmin=-0.8, xmax=0.8, colors='#01FBEE', ls='--', lw=2.5)
    # plt.xticks([0, 1, 2, 3, 4], xtick, fontsize=26)


def plot_dns_pred_hyb(data, err, tag, label):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(r'$\Delta U^+_{dns}$', fontsize=25)
    ax.set_ylabel(r'$\Delta U^+_{SVR\_rbf}$', fontsize=25)
    ax.set_ylim(0, 9)
    ax.set_xlim(0, 9)
    ax.set_xticks(np.arange(0, 10, 1))
    palette = ['#e9b03e', '#de1b15', '#989191', '#5ecce3', '#3e70e9', '#652ADE', '#f23b27']
    ax.plot(np.arange(0, 10), np.arange(0, 10), color='k', ls='--', lw=2.5)
    split_lst = [[] for _ in range(5)]
    for i in range(5):
        split_lst[i] = [item[0:2].astype('float') for item in data if str(item[-1]) == tag[i]]

        ax.scatter(np.squeeze(split_lst[i])[:, 0], np.squeeze(split_lst[i])[:, 1], c=palette[i], s=50, alpha=0.8,
                   label=label[i])
    ax.text(0.5, 8, os.path.join(r'$Err=$' + '{:.2f}%'.format(np.mean(err))), fontsize=24)
    ax.text(0.5, 7, os.path.join(r'$R_2=$' + '{:.2f}'.format(r2_score(data[:, 0], data[:, 1]))), fontsize=24)
    # plt.legend(framealpha=0.0, fontsize=24, loc=4, handletextpad=0.1)
    plt.tight_layout()
    # plt.savefig('/scratch/zhaoyus/Err_Ranking/Pytorch_ML/Copy_Data/err_linReg.jpeg', dpi=600)
    plt.show()


def plot_err(Err, ind, tags, xtick):
    sns.set_style('whitegrid')
    plt.gcf().set_size_inches(8, 8)
    plt.ylabel(r'$Err\%$', fontsize=35)

    palette = ['#e9963e', '#f23b27', '#65a9d7', '#304f9e']
    # palette = ['#FF2709', '#0030D7', '#01FBEE', '#09FF10', '#FA70B5', '#00145A']
    boxprops = dict(lw=2.0, color='#00145A')
    flierprops = dict(marker='o', ms=2.5, ls='none')
    whiskerprops = dict(lw=2.0, color='#00145A')
    capprops = dict(lw=2, color='#00145A')
    meanpointprops = dict(marker='D', ms=4, color='#01FBEE', mfc='#01FBEE')
    meanlineprops = dict(ls='--', lw=5)

    bw = [-0.35, 0, 0.35]
    for m in range(3):      # three models
        if ind == 4:
            x = np.random.normal(ind, 0.04, len(Err))
            plt.boxplot(Err[:, m].astype('float64'), positions=[ind+bw[m]], notch=False, showmeans=True, meanline=True,
                        showfliers=True, meanprops=meanpointprops, capprops=capprops, boxprops=boxprops, whiskerprops=whiskerprops)
            plt.text(bw[m]-0.25, np.mean(Err[:, m].astype('float64'))+0.5, '{:.2f}'.format(np.mean(Err[:, m].astype('float64'))),
                     fontsize=22)
            sca_list = [[] for _ in range(4)]
            for n in range(4):      # four surf types
                sca_list[n] = [item[m].astype('float64') for item in Err if str(item[-1]) == tags[n]]
            print(f'no. of test samples: {len(sca_list[0])}, {len(sca_list[1])}, {len(sca_list[2])},{len(sca_list[3])}')
            x_split = np.split(x+bw[m], [len(sca_list[0]), len(sca_list[0])+len(sca_list[1]),
                               len(sca_list[0])+len(sca_list[1])+len(sca_list[2])])
            [plt.scatter(x_split[n], sca_list[n], color=palette[n], s=24, alpha=0.5) for n in range(4)]
            print(ind, m, np.mean(Err[:, m].astype('float64')), np.max(Err[:, m].astype('float64')))
        else:
            x = np.random.normal(ind, 0.02, len(Err[0]))
            plt.boxplot(Err.T[:, m], positions=[ind+bw[m]], notch=True, showmeans=True, meanline=True, showfliers=True,
                        meanprops=meanpointprops, capprops=capprops, boxprops=boxprops, whiskerprops=whiskerprops)
            plt.scatter(x+bw[m], Err.T[:, m], color=palette[ind], s=24, alpha=0.5, label=xtick[ind])
            plt.text(ind+bw[m]-0.13, np.mean(Err.T[:, m])+2.5, '{:.2f}'.format(np.mean(Err.T[:, m])), fontsize=18, weight='bold')
            print(ind, m, np.mean(Err.T[:, m]), np.max(Err.T[:, m]))
            plt.legend(framealpha=0.0, fontsize=28)

if __name__ == "__main__":
    main()


def k_fold_CV():
    # If K too large, this leads to less variance across the training set
    # and limit the model accuracy difference across the iterations
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    err = [pd.DataFrame(columns=['linear', 'rbf']) for _ in range(3)]
    outliers = [[] for _ in range(3)]
    sel_size = [10, 19]

    name = ['Gaus', 'Pos', 'Hyb']
    score_list = [[] for _ in range(3)]
    for co, data in enumerate([stats_uplus_tag_Gaus, stats_uplus_tag_pos, stats_uplus_tag_shuffle]):
        train, test = train_test_split(data, test_size=0.2, random_state=0)
        print(f'---------Type: {co, name[co]}---------')
        print(f'TRAIN+VAL samp/size: \n {train.index.tolist()[0:5]}; {len(train)} \n '
              f'TEST samp/size: \n {test.index.tolist()[0:5]}; {len(test)}')

        # USE REGRESSION METRIC R2 TO SCORE ESTIMATOR ON TRAINING+VAL DATASET
        regr_linear, regr_rbf = svr_linear_model(), svr_rbf_model()
        sel = sel_size[1]       # change sel size manually
        for fold, (train_ind, val_ind) in enumerate(kf.split(train)):
            # print(f'train_ind: {train_ind} \n val_ind: {val_ind}')
            print(f'Fold={fold} train_size: {len(train.iloc[train_ind])} val_size: {len(train.iloc[val_ind])}')

            train_uplus = norm_1Ddata(train.iloc[train_ind, -2], np.min(train.iloc[train_ind, -2]),
                                      np.max(train.iloc[train_ind, -2]))
            train_max, train_min = np.max(train.iloc[:, :sel], axis=0), np.min(train.iloc[:, :sel], axis=0)
            train_stats = norm_2Ddata(train.iloc[train_ind, :sel].to_numpy(), train_min, train_max)

            val_stats = norm_2Ddata(train.iloc[val_ind, :sel].to_numpy(), train_min, train_max)
            val_uplus = norm_1Ddata(train.iloc[val_ind, -2], np.min(train.iloc[train_ind, -2]),
                                    np.max(train.iloc[train_ind, -2]))

            regr_rbf.fit(train_stats, train_uplus)
            regr_linear.fit(train_stats, train_uplus)

            score_linear = regr_linear.score(val_stats, val_uplus)
            score_rbf = regr_rbf.score(val_stats, val_uplus)
            # print(f' sel={sel} \n linear_ker_score={score_linear} \n rbf_ker_score={score_rbf}')

        # PREDICTION
        test_stats = norm_2Ddata(test.iloc[:, :sel].to_numpy(), np.min(train.iloc[:, :sel], axis=0),
                                 np.max(train.iloc[:, :sel], axis=0))

        pred_rbf = regr_rbf.predict(test_stats)
        pred_rbf_phy = denorm_1Ddata(pred_rbf, np.min(train.iloc[:, -2]), np.max(train.iloc[:, -2]))
        # test[os.path.join('pred_rbf_' + str(sel))] = pred_rbf_phy

        pred_lin = regr_linear.predict(test_stats)
        pred_lin_phy = denorm_1Ddata(pred_lin, np.min(train.iloc[:, -2]), np.max(train.iloc[:, -2]))
        # test[os.path.join('pred_lin_' + str(sel))] = pred_lin_phy

        err_rbf = 100 * np.abs((pred_rbf_phy - test['uplus'])) / test['uplus']
        err_lin = 100 * np.abs((pred_lin_phy - test['uplus'])) / test['uplus']
        err[co]['linear'] = err_lin
        err[co]['rbf'] = err_rbf


'''
pd.set_option('display.max_rows', None)
sample_dns = test.loc[:, ['uplus', 'tag']].iloc[:20, :]
sample_pred_1 = test.loc[:, ['pred_l_10', 'tag']].iloc[:20, :]
sort_dns = sample_dns.sort_values(by='uplus')
sort_pred = sample_pred_1.sort_values(by='pred_l_10')
# print('sample_dns:', '\n', sample_dns, '\n', 'sample_pred:', '\n', sample_pred)
print('sort_dns:', '\n', sort_dns, '\n', 'sort_pred:', '\n', sort_pred)
'''


