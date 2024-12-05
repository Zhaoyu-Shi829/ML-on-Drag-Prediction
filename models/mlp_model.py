import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MPLCONFIGDIR'] = '/tmp2/'
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, initializers, activations, losses, metrics
from keras.layers import LeakyReLU, ReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.metrics import r2_score
from utils import tags, configure_plots
from utils import no_eval_mlp, no_sd, no_epoch, no_stats, no_layer
from config import opt_flag, train_Test_flag, cls
from config import stats_base_dir, mlp_BO_dir, mlp_loss_dir, mlp_ckpt_dir

configure_plots()

print(f'tensor version: {tf.version.VERSION}')
keras.utils.set_random_seed(1)

uplus_pos = -2   # single dataset -1 (no tag); hybrid dataset -2 (tag)

# DEFINE GENERAL NORM EQUATIONS
def norm_2Ddata(data, min_inp, max_inp):    # rescaled input to [-1, 1]
    norm_data = np.copy(data)
    for i in range(len(min_inp)):
        alpha = 2/(float(max_inp[i]) - float(min_inp[i]))
        beta = -1-alpha * float(min_inp[i])
        norm_data[:, i] = alpha * data[:, i] + beta
    return norm_data

def lr_scheduler(epoch, learning_rate):
    lr = best_params_csv.iloc[-1].values[no_layer]
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


# load dataset
stats_dir = os.path.join(stats_base_dir, 'stats_{}.npy'.format(tags[cls]))
uplus_dir = os.path.join(stats_base_dir, 'uplus_{}.npy'.format(tags[cls]))
stats_data = np.squeeze(np.load(stats_dir))
uplus_data = np.squeeze(np.load(uplus_dir))
data_one = pd.DataFrame(np.column_stack((stats_data, uplus_data)))
dev_one, test_one = train_test_split(data_one, test_size=0.2, random_state=no_sd)
train_one, val_one = train_test_split(dev_one, test_size=0.2, random_state=no_sd)
print('-------' + os.path.join('data type: ' + tags[cls]) + '--------')
print(f'train: {train_one.shape}, val: {val_one.shape}, test: {test_one.shape}')

# separate dataset
x_train = train_one.iloc[:, :no_stats].to_numpy().astype('float32')
y_train = train_one.iloc[:, uplus_pos].to_numpy().astype('float32')
x_val = val_one.iloc[:, :no_stats].to_numpy().astype('float32')
y_val = val_one.iloc[:, uplus_pos].to_numpy().astype('float32')
x_test = test_one.iloc[:, :no_stats].to_numpy().astype('float32')
y_test = test_one.iloc[:, uplus_pos].to_numpy().astype('float32')
tag_test = test_one.iloc[:, -1].to_numpy().astype('str')

# CHOOSE NORMALIZATION (U+ scalar is unnecessary to norm)
def norm_stats():
    if minmax_norm:
        # MINMAX[-1,1]
        x_min, x_max = np.min(x_train, axis=0), np.max(x_train, axis=0)
        x_train_norm = norm_2Ddata(x_train, x_min, x_max)
        y_min, y_max = np.min(y_train), np.max(y_train)
        y_train_norm = norm_1Ddata(y_train, y_min, y_max)
        x_val_norm = norm_2Ddata(x_val, x_min, x_max)
        y_val_norm = norm_1Ddata(y_val, y_min, y_max)
        x_test_norm = norm_2Ddata(x_test, x_min, x_max)
        return x_train_norm, x_val_norm, x_test_norm, y_train_norm, y_val_norm
    elif std_norm:
        # STD
        x_train_norm = x_train/np.std(x_train)
        x_val_norm = x_val/np.std(x_val)
        x_test_norm = x_test/np.std(x_test)
        return x_train_norm, x_val_norm, x_test_norm

    # ---------------------BAYESIAN OPTIMIZATION CONFIG----------------------- #
    # define searching space of each hyperparam
    space = [
        # first BO estimates the range of n_layers, then fix it and BO n_neurons in each layer
        # n_layers can be fixed as 3
        Integer(32, 256, name='n_neuron_1'),
        Integer(32, 256, name='n_neuron_2'),
        Integer(32, 256, name='n_neuron_3'),
        # Integer(32, 256, name='n_neuron_4'),
        Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
        Real(1e-4, 1e-2, prior='log-uniform', name='regularization'),
        Categorical(['relu', 'leakyrelu', 'tanh', 'sigmoid'], name='activation'),
        Categorical(['GlorotNormal', 'GlorotUniform', 'HeNormal', 'HeUniform'], name='initialization'),
        Integer(2, 32, name='batch_size'),
        Integer(2, 32, name='val_batch_size'),
    ]

@use_named_args(dimensions=space)

def build_model(**space):
    model = keras.Sequential()
    # Add input layer
    model.add(layers.InputLayer(input_shape=(no_stats, )))
    # Define the model-building function
    for i in range(no_layer):
        model.add(layers.Dense(space['n_neuron_{}'.format(i+1)], kernel_initializer=space['initialization'],
                               kernel_regularizer=regularizers.l2(space['regularization'])))
        if space['activation'] == 'leakyrelu':
            model.add(LeakyReLU(alpha=0.3))
        else:
            model.add(layers.Activation(space['activation']))
    # Add output layer
    if space['activation'] == 'leakyrelu':
        model.add(layers.Dense(units=1, activation=LeakyReLU(alpha=0.3)))
    else:
        model.add(layers.Dense(units=1, activation=space['activation']))
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(space['lr']), loss=losses.MeanSquaredError(),
                  metrics=keras.metrics.MeanSquaredError())
    return model

start = time.time()
def objective(space):
    model = build_model(space)
    start_epoch = time.time()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    # ckpt = ModelCheckpoint(ckpt_fp, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    x_train_norm, x_val_norm = norm_stats()[0], norm_stats()[1]
    history = model.fit(x_train_norm, y_train, batch_size=space[-2], epochs=no_epoch,
                        validation_data=(x_val_norm, y_val), validation_batch_size=space[-1],
                        callbacks=es, verbose=1)
    print('---- training time for each evaluation %s seconds ----' % (time.time() - start_epoch))
    train_mse, val_mse = history.history['mean_squared_error'], history.history['val_mean_squared_error']

    return val_mse[-1]

# define a custom callback func to print evaluation number
def iter_fun(result):
    # save BOptimized hyperparams of each iteration
    iter_fun.counter += 1
    save_iter_result = {"n_neuron_1": result.x[0], "n_neuron_2": result.x[1], "n_neuron_3": result.x[2],
                        # "n_neuron_4": result.x[no_layer-1],
                        "lr": result.x[no_layer], "regularization": result.x[no_layer+1],
                        "activation": result.x[no_layer+2], "initialization": result.x[no_layer+3],
                        "batch_size": result.x[no_layer+4], "val_batch_size": result.x[no_layer+5]}
    df_BO = pd.DataFrame(save_iter_result, index=[0])
    df_BO.to_csv(mlp_BO_dir,  mode='a', header=False, index=False, sep=';')
    print(f"Iteration {iter_fun.counter} - f({result.x}): {result.fun}")

def plot_loss_diag(data, err, val_loss, val_mse, train_loss, train_mse):
    fig, ax = plt.subplots(ncols=3, figsize=(24, 8))
    [ax[i].set_box_aspect(1) for i in range(3)]
    [ax[i].locator_params(axis='y', nbins=10) for i in range(2)]
    palette = ['#e9b03e', '#de1b15', '#989191', '#5ecce3', '#3e70e9', '#652ADE', '#f23b27']
    label = [r'$I:Sk_0$', r'$II:Sk_{pos}$', r'$III:Sk_{neg}$', r'$IV:\lambda_x$', r'$V:\lambda_z$']
    ax[0].set_xlabel(r'$epoch$', fontsize=25)
    ax[0].set_ylabel(r'$loss$', fontsize=25)
    ax[1].set_xlabel(r'$epoch$', fontsize=25)
    ax[1].set_ylabel(r'$mse$', fontsize=25)
    ax[2].set_xlabel(r'$\Delta U^+_{dns}$', fontsize=25)
    ax[2].set_ylabel(r'$\Delta U^+_{mlp}$', fontsize=25)
    ax[2].set_ylim(0, 9)
    ax[2].set_xlim(0, 9)
    ax[2].set_xticks(np.arange(0, 10, 1))
    ax[2].plot(np.arange(0, 10), np.arange(0, 10), color='k', ls='--', lw=2.5)

    ax[0].plot(np.arange(1, len(np.squeeze(val_loss))), np.squeeze(val_loss)[1:], color='#DC381F',
                marker='o', lw=2., ms=5, mfc='none', mew=2, label=os.path.join(r'$validation$'))
    ax[0].plot(np.arange(1, len(np.squeeze(train_loss))), np.squeeze(train_loss[1:]), color='#5865F2',
                marker='o', lw=2., ms=5, mfc='none', mew=2, label=os.path.join(r'$training$'))
    ax[1].plot(np.arange(1, len(np.squeeze(val_mse))), np.squeeze(val_mse[1:]), color='#DC381F',
                marker='o', lw=2., ms=5, mfc='none', mew=2, label=os.path.join(r'$validation$'))
    ax[1].plot(np.arange(1, len(np.squeeze(train_mse))), np.squeeze(train_mse[1:]), color='#5865F2',
                marker='o', lw=2., ms=5, mfc='none', mew=2, label=os.path.join(r'$training$'))

    split_lst = [[] for _ in range(5)]
    for i in range(5):
        split_lst[i] = [item[0:2].astype('float') for item in data.to_numpy() if str(item[-1]) == tags[i]]
        ax[2].scatter(np.squeeze(split_lst[i])[:, 0], np.squeeze(split_lst[i])[:, 1], c=palette[i], s=50, alpha=0.8,
                      label=label[i])
    ax[2].text(0.5, 8, os.path.join(r'$Err=$' + '{:.2f}%'.format(np.mean(err))), fontsize=24)
    ax[2].text(0.5, 7, os.path.join(r'$R_2=$' + '{:.2f}'.format(r2_score(data.to_numpy()[:, 0], data.to_numpy()[:, 1]))),
               fontsize=24)
    ax[0].legend(framealpha=0.0, fontsize=20, handletextpad=0.1)
    ax[2].legend(framealpha=0.0, fontsize=20, loc=4, handletextpad=0.1)
    plt.savefig(mlp_loss_dir, dpi=600)
    plt.show()


if opt_flag:
    # Run BO to find the optimal hps
    iter_fun.counter = 0
    result = gp_minimize(func=objective, dimensions=space, n_calls=no_eval, n_jobs=-1, random_state=42,
                         acq_func='gp_hedge', verbose=False, callback=iter_fun)
    print("Best parameters: ", result.x)
    print("Minimum val loss: ", result.fun)

if train_test_flag:
    print("----------Loading BO hyperparams -----------")
    best_params_csv = pd.read_csv(save_BO_dir, sep=";")
    final_model = build_model(best_params_csv.iloc[-1].values.tolist())
    print(final_model.summary())
    print(f'best params: {best_params_csv.iloc[-1].values.tolist()}')
    print("----------Starting Training -----------")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    x_train_norm, x_val_norm, x_test_norm = norm_stats()[0], norm_stats()[1], norm_stats()[2]
    final_history = final_model.fit(x_train_norm, y_train, batch_size=best_params_csv.iloc[-1].values[-2], epochs=no_epoch,
                                    validation_data=(x_val_norm, y_val),
                                    validation_batch_size=best_params_csv.iloc[-1].values[-1],
                                    callbacks=[new_lr],
                                    verbose=2)
    train_loss, val_loss = final_history.history['loss'], final_history.history['val_loss']
    train_mse, val_mse = final_history.history['mean_squared_error'], final_history.history['val_mean_squared_error']
    print("Total training time: {:.2f} seconds".format(time.time() - start))
    print("----------Starting Testing-----------")
    y_pred = final_model.predict(x_test_norm)
    save_pred = {"pred": np.squeeze(y_pred), "dns": y_test, "tag": tag_test}
    df = pd.DataFrame(save_pred)
    df.to_csv(mlp_pred_dir, index=False)
    print(f'pred .vs. dns \n {np.c_[y_pred, y_test]}')
    print(f'mean uplus: {np.mean(y_pred), np.mean(y_test)}')
    err = 100 * np.abs(np.squeeze(y_pred) - y_test) / y_test
    print(f'mean err: {np.mean(err)}')

    plot_loss_diag(data=df, err=err, val_loss=val_loss, val_mse=val_mse,
                   train_loss=train_loss, train_mse=train_mse)

























