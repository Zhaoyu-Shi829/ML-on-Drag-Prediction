import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['MPLCONFIGDIR'] = '/lscratch/zhaoyus/tmp'
#sys.path.append('/lscratch/zhaoyus/python_packages')
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from keras import initializers, regularizers
from keras import losses
from keras import activations
from keras import metrics
from keras.layers import LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from glob import glob
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from skopt import dump, load
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

rc = {
      'font.family': 'sans-serif', 'font.serif': 'Times New Roman',
      'mathtext.fontset': 'stix', 'font.size': 20
}
plt.rcParams.update(rc)

print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Tensorflow version:", tf.__version__)

# deterministic/global seeding
tf.config.experimental.enable_op_determinism()
keras.utils.set_random_seed(42)

# set up n_call=no_eval, random_state=no_sd(split dataset), epochs=no_epoch
no_eval = 100
no_sd = 42
no_epoch = 300
# flag to control training and inference
opt_flag = False
train_test_flag = True
# save directory of BO hyperparams, training
save_BO_dir = './tmp_cnn_BO/pre_test_4_rerun/hyb_sd{}_n{}_BOhp_ofstd_old.csv'.format(no_sd, no_eval)
save_BOloss_dir = './tmp_cnn_BO/pre_test_4_rerun/hyb_sd{}_n{}_BOloss_ofstd.csv'.format(no_sd, no_eval)
save_loss_dir = './tmp_cnn_BO/pre_test_4_rerun/hyb_sd{}_n{}_loss_pred_ofstd.jpeg'.format(no_sd, no_eval)
save_pred_dir = './tmp_cnn_BO/pre_test_4_rerun/hyb_sd{}_n{}_pred_dns_ofstd.csv'.format(no_sd, no_eval)

# ---------------------PRE-PROCESSING DATA------------------------------- #
base_dir = './Copy_Data/surf_height_cnn'
h_uplus_hyb = pd.DataFrame(np.load(os.path.join(base_dir, 'h_uplus_hyb.npy')))
# seeding here is for BO-training
dev, test = train_test_split(h_uplus_hyb, test_size=0.2, random_state=no_sd)
train, val = train_test_split(dev, test_size=0.2, random_state=no_sd)
# seeding here is for testing the BO-trained model with various dataset
# dev, test = train_test_split(h_uplus_hyb, test_size=0.2, random_state=12)
# train, val = train_test_split(dev, test_size=0.2, random_state=12)
print(f'--------train/test dataset: {len(dev)}/{len(test)}---------')

map_dim = [102, 302]

def norm_minmax(data, vmin, vmax):
    alpha = 2 / (vmax - vmin)
    norm_data = alpha * (data - vmin) - 1
    return norm_data

def denorm_uplus(data, vmin, vmax):
    alpha = 2 / (vmax - vmin)
    denorm_data = (data + 1) / alpha + vmin
    return denorm_data

def norm_std(data):
    norm_data = data/np.std(data)
    return norm_data

def lr_scheduler(epoch, learning_rate):
    lr = best_params_csv.iloc[0].values[11]
    if epoch < 100:
        learning_rate = lr
        return learning_rate
    elif epoch < 200:
        learning_rate = lr*0.1
        return learning_rate
    elif epoch < 300:
        learning_rate = lr*0.01
        return learning_rate
    else:
        learning_rate = lr*0.001
        return learning_rate

new_lr = LearningRateScheduler(lr_scheduler, verbose=1)

# NORMALIZATION
x_train = train.iloc[:, :-1].to_numpy().reshape(len(train), map_dim[0], map_dim[1])
y_train = train.iloc[:, -1].to_numpy()
# ------- k_i - <k>_all_surfs -------
x_train_ofs = x_train-np.mean(x_train)
# ------ min-max norm -------
x_min, x_max = train.iloc[:, :-1].to_numpy().min(), train.iloc[:, :-1].to_numpy().max()
#x_train_norm = norm_minmax(x_train, x_min, x_max)
# ------ std norm -------
# x_train_norm = norm_std(x_train)
x_train_norm = x_train_ofs/np.std(x_train_ofs)
y_train_norm = norm_minmax(y_train, y_train.min(), y_train.max())

x_val = val.iloc[:, :-1].to_numpy().reshape(len(val), map_dim[0], map_dim[1])
y_val = val.iloc[:, -1].to_numpy()
# ------- k_i - <k>_all_surfs -------
x_val_ofs = x_val-np.mean(x_train)
# ------ min-max norm -------
#x_val_norm = norm_minmax(x_val, x_min, x_max)
# ------ std norm -------
# x_val_norm = norm_std(x_val)
x_val_norm = x_val/np.std(x_train_ofs)
y_val_norm = norm_minmax(y_val, y_train.min(), y_train.max())

x_test = test.iloc[:, :-1].to_numpy().reshape(len(test), map_dim[0], map_dim[1])
y_test = test.iloc[:, -1].to_numpy()
# ------- k_i - <k>_all_surfs -------
x_test_ofs = x_test-np.mean(x_train)
# ------ min-max norm -------
#x_test_norm = norm_minmax(x_test, x_min, x_max)
# ------ std norm -------
# x_test_norm = norm_std(x_test_ofs)
x_test_norm = x_test_ofs/np.std(x_train_ofs)

# ---------------------BAYESIAN OPTIMIZATION CONFIG----------------------- #
# define BO searching space of hyperparams
space = [
        #Integer(2, 6, name='n_layers'), we fix the no. of blocks = 5 based on previous test; it usually ranges in [4,6]
    Integer(8, 64, name='n_filter_1'),
    Integer(8, 64, name='n_filter_2'),
    Integer(8, 64, name='n_filter_3'),
    Integer(8, 64, name='n_filter_4'),
    Integer(8, 64, name='n_filter_5'),
    Integer(3, 15, name='kernel_size_1'),
    Integer(3, 15, name='kernel_size_2'),
    Integer(3, 15, name='kernel_size_3'),
    Integer(3, 15, name='kernel_size_4'),
    Integer(3, 15, name='kernel_size_5'),
    Integer(3, 15, name='kernel_size'),
    Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
    Real(1e-4, 1e-2, prior='log-uniform', name='regularization'),
    Categorical(['leakyrelu', 'tanh', 'sigmoid'], name='activation'),
    Categorical(['GlorotNormal', 'GlorotUniform', 'HeNormal', 'HeUniform'], name='initialization'),
    Integer(2, 21, name='batch_size'),
    Integer(2, 32, name='val_batch_size'),
]
@use_named_args(dimensions=space)

def build_model(**space):
    model = keras.Sequential()
    # Add input layer; images have one channel (102, 302, 1)
    input_shape = (map_dim[0], map_dim[1], 1)
    model.add(layers.InputLayer(input_shape=input_shape))
    #for i in range(space['n_layers']):
    for i in range(5):
        model.add(Conv2D(filters=space['n_filter_{}'.format(i+1)],
                         kernel_size=(space['kernel_size_{}'.format(i+1)], space['kernel_size_{}'.format(i+1)]),
                         padding='same',
                         kernel_regularizer=regularizers.l2(space['regularization']),
                         kernel_initializer=space['initialization']))
        if space['activation'] == 'leakyrelu':
            model.add(LeakyReLU(alpha=0.3))
        else:
            model.add(layers.Activation(space['activation']))
    model.add(BatchNormalization())
    # Add output layer
    model.add(Conv2D(filters=1, kernel_size=(space['kernel_size'], space['kernel_size']), padding='same'))
    if space['activation'] == 'leakyrelu':
        model.add(LeakyReLU(alpha=0.3))
    else:
        model.add(layers.Activation(space['activation']))
    model.add(GlobalAveragePooling2D())
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(space['lr']), loss=losses.MeanSquaredError(),
                  metrics=keras.metrics.MeanSquaredError())
    return model

start = time.time()

def objective(space):
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    model = build_model(space)
    # ckpt_filepath = './tmp_cnn_v1/cnn_ckpt/model_{epoch:03d}_{val_loss:.4f}.h5'
    # ckpt = ModelCheckpoint(ckpt_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    start_epoch = time.time()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    history = model.fit(x_train_norm, y_train, batch_size=space[-2], epochs=no_epoch,
                        validation_data=(x_val_norm, y_val), validation_batch_size=space[-1],
                        callbacks=es)
    print('---- training time for each evaluation %s seconds ----' % (time.time() - start_epoch))
    train_loss, val_loss = history.history['loss'], history.history['val_loss']
    '''
    # plot train/val loss at the last evaluation of BO
    if iter_fun.counter == no_eval - 1:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(np.arange(0, len(np.squeeze(val_loss))), np.squeeze(val_loss), color='#e9963e',
                marker='o', lw=2.5, ms=5, mfc='none', mew=2, label=os.path.join(r'$val$'))
        ax.plot(np.arange(0, len(np.squeeze(train_loss))), np.squeeze(train_loss), color='#f23b27',
                lw=2.5, label=os.path.join(r'$train$'))
        plt.show()
    '''
    return val_loss[-1]

# define a custom callback func to print evaluation number
def iter_fun(result):
    iter_fun.counter += 1
    save_iter_result = {"n_filter_1": result.x[0], "n_filter_2": result.x[1], "n_filter_3": result.x[2], "n_filter_4": result.x[3], "n_filter_5": result.x[4],
                        "kernel_size_1": result.x[5], "kernel_size_2": result.x[6], "kernel_size_3": result.x[7], "kernel_size_4": result.x[8],
                        "kernel_size_5": result.x[9], "kernel_size": result.x[10], "lr": result.x[11],
                        "regularization": result.x[12], "activation": result.x[13], "initialization": result.x[14],
                        "batch_size": result.x[15], "val_batch_size": result.x[16]}
    df_BO = pd.DataFrame(save_iter_result, index=[0])
    df_BO.to_csv(save_BO_dir, mode='a', header=False, index=False, sep=';')
    print(f"Iteration {iter_fun.counter} - f({result.x}): {result.fun}")
    df_BO_loss = pd.DataFrame(result.func_cals)
    df_BO_loss.to_csv(save_BOloss_dir, mode='a', sep=';')

if opt_flag:
    print("-----------Run BO to find the optimal hps------------")
    iter_fun.counter = 0
    result = gp_minimize(func=objective, dimensions=space, n_calls=no_eval, random_state=42, acq_func='gp_hedge',
                         n_restarts_optimizer=10, n_jobs=-1, verbose=False, callback=iter_fun)
    print("Best parameters: ", result.x)
    print("Minimum val loss: ", result.fun)


if train_test_flag:
    print("----------Loading BO hyperparams -----------")
    best_params_csv = pd.read_csv(save_BO_dir, sep=";")
    final_model = build_model(best_params_csv.iloc[-1].values.tolist())
    print(f'best params: {best_params_csv.iloc[-1].values.tolist()}')
    print("----------Starting Training -----------")
    final_model.summary()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    final_history = final_model.fit(x_train_norm, y_train, batch_size=best_params_csv.iloc[0].values[-2], epochs=no_epoch,
                                    validation_data=(x_val_norm, y_val), validation_batch_size=best_params_csv.iloc[0].values[-1],
                                    callbacks=[new_lr], verbose=2)
    train_loss, val_loss = final_history.history['loss'], final_history.history['val_loss']
    train_mse, val_mse = final_history.history['mean_squared_error'], final_history.history['val_mean_squared_error']
    print("Total training time: {:.2f} seconds".format(time.time()-start))
    print("----------Starting Testing-----------")
    y_pred = final_model.predict(x_test_norm)
    # y_pred_phy = denorm_height(y_pred, y_train.min(), y_train.max())
    # y_pred_phy = StandardScaler().inverse_transform(y_pred)
    save_pred = {"pred": np.squeeze(y_pred), "dns": y_test}
    df = pd.DataFrame(save_pred)
    df.to_csv(save_pred_dir, index=False, sep=';')
    #df.to_csv('./tmp_cnn_BO/pre_test_3/hyb_sd22_sd12_pred_dns_ofsglobal_stdnorm_fs.csv', index=False, sep=';')
    print(f'pred .vs. dns \n {np.c_[y_pred, y_test]}')
    print(f'pred mean U+: {np.mean(y_pred)}, DNS mean U+: {np.mean(y_test)}')
    err = 100 * np.abs(np.squeeze(y_pred) - y_test) / y_test
    print('mean err %: ', np.mean(err))


    # Visualize the loss curve and the prediction error
    fig, axs = plt.subplots(ncols=3, figsize=(24, 8))
    axs[0].set_xlabel(r'$epoch$', fontsize=25)
    axs[0].set_ylabel(r'$loss$', fontsize=25)
    axs[0].set_ylim(0, 6)
    axs[1].set_xlabel(r'$epoch$', fontsize=25)
    axs[1].set_ylabel(r'$mse$', fontsize=25)
    axs[1].set_ylim(0, 6)
    axs[2].set_ylim(0, 9)
    axs[2].set_xlim(0, 9)
    axs[2].set_xlabel(r'$\Delta U^+_{dns}$', fontsize=25)
    axs[2].set_ylabel(r'$\Delta U^+_{pred}$', fontsize=25)
    palette = ['#e9963e', '#f23b27', '#65a9d7', '#304f9e', '#00145A']
    axs[0].plot(np.arange(0, len(np.squeeze(val_loss))), np.squeeze(val_loss), color='#DC381F',
                marker='o', lw=2.5, ms=5, mfc='none', mew=2, label=os.path.join(r'$validation$'))
    axs[0].plot(np.arange(0, len(np.squeeze(train_loss))), np.squeeze(train_loss), color='#5865F2',
                marker='o', lw=2.5, ms=5, mfc='none', mew=2, label=os.path.join(r'$training$'))
    axs[1].plot(np.arange(0, len(np.squeeze(val_mse))), np.squeeze(val_mse), color=palette[0],
                marker='o', lw=2.5, ms=5, mfc='none', mew=2, label=os.path.join(r'$validation$'))
    axs[1].plot(np.arange(0, len(np.squeeze(train_mse))), np.squeeze(train_mse), color=palette[1],
                lw=2.5, label=os.path.join(r'$training$'))
    axs[2].text(0.5, 7, os.path.join(r'$R^2=$' + '{:.2f}'.format(r2_score(y_test, y_pred))), fontsize=24)
    axs[2].plot(np.arange(0, 9), np.arange(0, 9), color='k', ls='--', lw=2.5)
    axs[2].scatter(y_test, y_pred, c=palette[2], s=50)
    axs[2].text(0.5, 8, os.path.join(r'$Err=$'+'{:.2f}%'.format(np.mean(err))), fontsize=24)
    axs[0].legend(framealpha=0.0, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_loss_dir, dpi=600)
    plt.show()


