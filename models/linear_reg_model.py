import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils import configure_plots, palette, labels, tags
from config import lr_pred_dir, cls, uplus_pos, no_stats
from config import stats_base_dir, lr_pred_dir

configure_plots()

def main():
    # --------------PRE-PROCESS DATA------------------ #
    # load hybrid data
    stats_dir = os.path.join(stats_base_dir, 'stats_{}.npy'.format(tags[cls]))
    uplus_dir = os.path.join(stats_base_dir, 'uplus_{}.npy'.format(tags[cls]))
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

    pred, err, r2, coefs = regression(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # --------Plot pred .vs. dns--------
    if cls == -1:
        tag = test_one.iloc[:, -1]
        save_pred = {"pred": np.squeeze(pred), "dns": np.squeeze(y_test), "tag": tag}
        df = pd.DataFrame(save_pred)
        plot_dns_pred_hyb(data=df.to_numpy(), err=err, tag=tags, label=labels)
        df.to_csv(lr_pred_dir, index=False)


def regression(x_train, y_train, x_test, y_test):
    reg = LinearRegression().fit(x_train, y_train)
    print(f'coefs: {reg.coef_}, interp: {reg.intercept_}')
    coefs, intercept = reg.coef_, reg.intercept_
    weights = pd.DataFrame(np.append(coefs, intercept))
    y_pred = reg.predict(x_test)
    comparison = np.stack((np.squeeze(y_pred), np.squeeze(y_test))).T
    print(f'prediction and ground-truth: \n {comparison}')

    err = 100*(np.abs(np.squeeze(y_pred)-np.squeeze(y_test))/np.squeeze(y_test))
    R2 = r2_score(np.squeeze(y_test), np.squeeze(y_pred))
    print(f'mean err: {np.mean(err)}, max err: {np.max(err)}')
    return y_pred, err, R2, np.abs(reg.coef_)

def plot_dns_pred_hyb(data, err, tag, label):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(r'$\Delta U^+_{dns}$', fontsize=25)
    ax.set_ylabel(r'$\Delta U^+_{LR}$', fontsize=25)
    ax.set_ylim(0, 9)
    ax.set_xlim(0, 9)
    ax.set_xticks(np.arange(0, 10, 1))
    ax.plot(np.arange(0, 10), np.arange(0, 10), color='black', ls='--', lw=2.5)
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


















































