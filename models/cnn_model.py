import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['MPLCONFIGDIR'] = '/lscratch/zhaoyus/tmp'  # change into your own DIR
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers, activations
from keras.layers import LeakyReLU, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from keras import metrics, losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from config import map_dim, no_eval_cnn, no_sd, no_epoch
from config import cnn_base_dir, cnn_BO_dir, cnn_loss_dir, cnn_ckpt_dir
from config import opt_flag, train_Test_flag
from utils import tags, configure_plots

configure_plots()

print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Tensorflow version:", tf.__version__)

# deterministic/global seeding
tf.config.experimental.enable_op_determinism()
keras.utils.set_random_seed(42)

# ---------------------PRE-PROCESSING DATA------------------------------- #
# load surface height map
h_uplus_hyb = pd.DataFrame(np.load(os.path.join(base_dir, 'h_uplus_hyb.npy')))
# data splitting with deterministic seeding for BO-training
dev, test = train_test_split(h_uplus_hyb, test_size=0.2, random_state=no_sd)
train, val = train_test_split(dev, test_size=0.2, random_state=no_sd)
print(f'--------train/test dataset: {len(dev)}/{len(test)}---------')

# learning curve reschedular
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

# define min-max normalilzation func
def norm_minmax(data, vmin, vmax):
    alpha = 2 / (vmax - vmin)
    norm_data = alpha * (data - vmin) - 1
    return norm_data

# reshape 2D map into 1D
x_train = train.iloc[:, :-1].to_numpy().reshape(len(train), map_dim[0], map_dim[1])
x_val = val.iloc[:, :-1].to_numpy().reshape(len(val), map_dim[0], map_dim[1])
x_test = test.iloc[:, :-1].to_numpy().reshape(len(test), map_dim[0], map_dim[1])
y_train = train.iloc[:, -1].to_numpy()
y_val = val.iloc[:, -1].to_numpy()
y_test = test.iloc[:, -1].to_numpy()

# --------k_i-<k>_of_all_surfaces-------- #
x_train_ofs = x_train-np.mean(x_train)
x_val_ofs = x_val-np.mean(x_train)
x_test_ofs = x_test-np.mean(x_train)

# three ways to normalize height map
def norm_stats():
    # ------ min-max norm -------
    if minmax_norm:
        x_min, x_max = train.iloc[:, :-1].to_numpy().min(), train.iloc[:, :-1].to_numpy().max()
        x_train_norm = norm_minmax(x_train, x_min, x_max)
        x_val_norm = norm_minmax(x_val, x_min, x_max)
        x_test_norm = norm_minmax(x_test, x_min, x_max)
        return x_train_norm, x_val_norm, x_test_norm
    # ------ std norm -------
    if std_ofs_norm:
        x_train_norm = x_train_ofs/np.std(x_train_ofs)
        x_val_norm = x_val / np.std(x_train_ofs)
        x_test_norm = x_test_ofs / np.std(x_train_ofs)
        return x_train_norm, x_val_norm, x_test_norm
    if std_norm:
        x_train_norm = x_train / np.std(x_train)
        x_val_norm = x_val / np.std(x_train)
        x_test_norm = x_test / np.std(x_train)
        return x_train_norm, x_val_norm, x_test_norm

# ---------------------BAYESIAN OPTIMIZATION CONFIG----------------------- #
# define BO searching space of hyperparams
space = [
    # we fix the no. of blocks = 5 based on previous test
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
    for i in range(no_layer):
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
  strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
  with strategy.scope():
    model = build_model(space)
    # ckpt_filepath = os.path.join(cnn_ckpt_dir, 'model_{epoch:03d}_{val_loss:.4f}.h5')
    # ckpt = ModelCheckpoint(ckpt_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    start_epoch = time.time()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    x_train_norm, x_val_norm = norm_stats()[0], norm_stats()[1]
    history = model.fit(x_train_norm, y_train, batch_size=space[-2], epochs=no_epoch,
                        validation_data=(x_val_norm, y_val), validation_batch_size=space[-1],
                        callbacks=es)
    print('---- training time for each evaluation %s seconds ----' % (time.time() - start_epoch))
    train_loss, val_loss = history.history['loss'], history.history['val_loss']
    '''
    # plot train/val loss at the last evaluation of BO
    if iter_fun.counter == no_eval_cnn - 1:
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
    save_iter_result = {"n_filter_1": result.x[0], "n_filter_2": result.x[1], "n_filter_3": result.x[2],
                        "n_filter_4": result.x[3], "n_filter_5": result.x[4],
                        "kernel_size_1": result.x[5], "kernel_size_2": result.x[6], "kernel_size_3": result.x[7],
                        "kernel_size_4": result.x[8], "kernel_size_5": result.x[9], "kernel_size": result.x[10],
                        "lr": result.x[11],
                        "regularization": result.x[12],
                        "activation": result.x[13],
                        "initialization": result.x[14],
                        "batch_size": result.x[15],
                        "val_batch_size": result.x[16]}
    df_BO = pd.DataFrame(save_iter_result, index=[0])
    df_BO.to_csv(cnn_BO_dir, mode='a', header=False, index=False, sep=';')
    print(f"Iteration {iter_fun.counter} - f({result.x}): {result.fun}")


def plot_loss(val_loss, val_mse, train_loss, train_mse):
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
    [ax[i].set_box_aspect(1) for i in range(3)]
    [ax[i].locator_params(axis='y', nbins=10) for i in range(2)]
    ax[0].set_xlabel(r'$epoch$', fontsize=25)
    ax[0].set_ylabel(r'$loss$', fontsize=25)
    ax[1].set_xlabel(r'$epoch$', fontsize=25)
    ax[1].set_ylabel(r'$mse$', fontsize=25)

    ax[0].plot(np.arange(1, len(np.squeeze(val_loss))), np.squeeze(val_loss)[1:], color='#DC381F',
               marker='o', lw=2., ms=5, mfc='none', mew=2, label=os.path.join(r'$validation$'))
    ax[0].plot(np.arange(1, len(np.squeeze(train_loss))), np.squeeze(train_loss[1:]), color='#5865F2',
               marker='o', lw=2., ms=5, mfc='none', mew=2, label=os.path.join(r'$training$'))
    ax[1].plot(np.arange(1, len(np.squeeze(val_mse))), np.squeeze(val_mse[1:]), color='#DC381F',
               marker='o', lw=2., ms=5, mfc='none', mew=2, label=os.path.join(r'$validation$'))
    ax[1].plot(np.arange(1, len(np.squeeze(train_mse))), np.squeeze(train_mse[1:]), color='#5865F2',
               marker='o', lw=2., ms=5, mfc='none', mew=2, label=os.path.join(r'$training$'))

    ax[0].legend(framealpha=0.0, fontsize=20, handletextpad=0.1)
    plt.savefig(cnn_loss_dir, dpi=600)
    plt.show()

if opt_flag:
    print("-----------Run BO to find the optimal hps------------")
    iter_fun.counter = 0
    result = gp_minimize(func=objective, dimensions=space, n_calls=no_eval_cnn, random_state=42, acq_func='gp_hedge',
                         n_restarts_optimizer=10, n_jobs=-1, verbose=False, callback=iter_fun)
    print(f"Best parameters: , {result.x}")
    print(f"Minimum val loss: , {result.fun}")

if train_test_flag:
    print("----------Loading BO hyperparams -----------")
    best_params_csv = pd.read_csv(cnn_BO_dir, sep=";")
    final_model = build_model(best_params_csv.iloc[-1].values.tolist())
    print(f'best params: {best_params_csv.iloc[-1].values.tolist()}')
    print("----------Starting Training -----------")
    final_model.summary()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    x_train_norm, x_val_norm, x_test_norm = norm_stats()[0], norm_stats()[1], norm_stats()[2]
    final_history = final_model.fit(x_train_norm, y_train, batch_size=best_params_csv.iloc[0].values[-2], epochs=no_epoch,
                                    validation_data=(x_val_norm, y_val), validation_batch_size=best_params_csv.iloc[0].values[-1],
                                    callbacks=[new_lr], verbose=2)
    train_loss, val_loss = final_history.history['loss'], final_history.history['val_loss']
    train_mse, val_mse = final_history.history['mean_squared_error'], final_history.history['val_mean_squared_error']
    print("Total training time: {:.2f} seconds".format(time.time()-start))
    print("----------Starting Testing-----------")
    y_pred = final_model.predict(x_test_norm)
    save_pred = {"pred": np.squeeze(y_pred), "dns": y_test, "tag": tag_test}
    df = pd.DataFrame(save_pred)
    df.to_csv(cnn_pred_dir, index=False, sep=';')
    # df.to_csv('./tmp_cnn_BO/pre_test_3/hyb_sd22_sd12_pred_dns_ofsglobal_stdnorm_fs.csv', index=False, sep=';')
    print(f'pred .vs. dns \n {np.c_[y_pred, y_test]}')
    print(f'pred mean U+: {np.mean(y_pred)}, DNS mean U+: {np.mean(y_test)}')
    err = 100 * np.abs(np.squeeze(y_pred) - y_test) / y_test
    print(f'mean err %: , {np.mean(err)}')

    plot_loss(val_loss=val_loss, val_mse=val_mse, train_loss=train_loss, train_mse=train_mse)




















