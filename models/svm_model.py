import os
import sys
print('python version:', sys.version)
print('python interpreter:', sys.executable)
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from utils import configure_plots, palette, labels, tags
from config import svm_val_flag, svm_test_flag, svm_pred_dir, stats_base_dir
from config import cls, no_stats, uplus_pos

configure_plots()

def norm_2Ddata(data, min_inp, max_inp):    # RESCALED INPUT TO [-1, 1]
    norm_data = np.copy(data)
    for i in range(len(min_inp)):
        alpha = 2/(max_inp[i]-min_inp[i])
        beta = -1-alpha*min_inp[i]
        norm_data[:, i] = alpha*data[:, i]+beta
    return norm_data


# CHOOSE DIFFERENT KERNELS FOR SVM MODEL
# one can still use Bayesian to choose hyparams: C and epsilon,
# this current dataset does not need much tuning

def svr_rbf_model(ind):
    # K(x, x')=exp(-gamma ||x-x'||^2)
    # min-max norm: epsilon=0.01, C=1; std-norm: C=1000~5000
    eps = [0.04, 0.001, 0.01, 0.001, 0.04, 0.01]
    regr_rbf = svm.SVR(kernel='rbf', gamma='scale', C=10, epsilon=eps[ind], shrinking=False, verbose=True)
    return regr_rbf

def main():
    stats_dir = os.path.join(stats_base_dir, 'stats_{}.npy'.format(tags[cls]))
    uplus_dir = os.path.join(stats_base_dir, 'uplus_{}.npy'.format(tags[cls]))
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
    # ------ To justify svm actually learn pattern instead of interpolate ------- #
    # ------ use NearestNeighbor Interpolator to compare with svm --------- #
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

    # min_max norm
    x_max, x_min = np.max(x_train, axis=0), np.min(x_train, axis=0)
    x_train_norm = norm_2Ddata(x_train, x_min, x_max)
    print(f'kernel coef gamma: {1/(x_train_norm.var()*no_stats)}')
    x_val_norm = norm_2Ddata(x_val, x_min, x_max)
    print('//----Train----')
    regr_rbf = svr_rbf_model(ind=cls)
    regr_rbf.fit(x_train_norm, y_train)

    if svm_val_flag:
        print('//----Val----')
        score_rbf = regr_rbf.score(x_val_norm, y_val)
        print(f'no_stat={no_stats}: rbf_score={score_rbf}')
    if svm_test_flag:
        print('//----Test----')
        # min_max norm
        x_test_norm = norm_2Ddata(x_test, x_min, x_max)
        pred_rbf = regr_rbf.predict(x_test_norm)
        err_rbf = 100*np.abs((np.squeeze(pred_rbf)-np.squeeze(y_test)))/np.squeeze(y_test)

        # --------Plot pred .vs. dns--------
        save_pred = {"pred": np.squeeze(pred_rbf), "dns": np.squeeze(y_test), "tag": tag_test}
        df = pd.DataFrame(save_pred)
        plot_dns_pred_hyb(data=df.to_numpy(), err=err_rbf, tag=tags, label=labels)
        df.to_csv(svm_pred_dir, index=False)


def plot_dns_pred_hyb(data, err, tag, label):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(r'$\Delta U^+_{dns}$', fontsize=25)
    ax.set_ylabel(r'$\Delta U^+_{SVR\_rbf}$', fontsize=25)
    ax.set_ylim(0, 9)
    ax.set_xlim(0, 9)
    ax.set_xticks(np.arange(0, 10, 1))
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
    plt.show()


if __name__ == "__main__":
    main()





























