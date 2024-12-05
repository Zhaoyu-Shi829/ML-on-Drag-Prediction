# specify iteration number of Bayesian Optimization(BO): n_call=no_eval_cnn/mlp
# specify number of layers in mlp: no_layer
# specify seeding number for data splitting: random_state=no_sd(split dataset)
# specify number of passing through training dataset: epochs=no_epoch
# specify number of topographical parameters: no_stats
no_eval_cnn = 100
no_eval_mlp = 60
no_epoch = 300
no_sd = 42
no_stats = 10
no_layer = 3

# flags to control training and BO process for mlp and cnn
opt_flag = False
train_Test_flag = True
# flags to control training and validation for svm
svm_val_flag = True
svm_test_flag = True

# flags to select data normalization methods
minmax_norm = False
std_norm = False     # surface height from
std_ofs_norm = True  # surface height reduced by mean height

# height map dimension
map_dim = [102, 302]   # mesh resolution
cls = -1               # correspond to tags, -1: hybrid data
uplus_pos = -2         # hybrid data -2; single data -1

# directories to save BO hyperparams, tranining result (loss, hyparams, prediction)
cnn_base_dir = '../data/surf_height_cnn'
cnn_pred_dir = '../cnn_output/pred_sd{}_n{}.csv'.format(no_sd, no_eval_cnn)
cnn_BO_dir = '../cnn_output/BOhp_sd{}_n{}.csv'.format(no_sd, no_eval_cnn)
cnn_loss_dir = '../cnn_output/loss_sd{}_n{}.jpeg'.format(no_sd, no_eval_cnn)
cnn_ckpt_dir = '../cnn_output/cnn_ckpt'

stats_base_dir = '../data/statistics'
mlp_pred_dir = '../mlp_output/pred_sd{}_n{}.csv'.format(no_sd, no_eval_mlp)
mlp_BO_dir = '../mlp_output/BOhp_sd{}_n{}.csv'.format(no_sd, no_eval_mlp)
mlp_loss_dir = '../mlp_output/loss_sd{}_n{}.jpeg'.format(no_sd, no_eval_mlp)

svm_pred_dir = '../svm_output/pred_rbf_sd{}.csv'.format(no_sd)
lr_pred_dir = '../lr_output/pred_lr_sd{}.csv'.format(no_sd)














