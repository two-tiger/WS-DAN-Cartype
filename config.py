##################################################
# Training Config
##################################################
GPU = '0'                   # GPU
workers = 0                 # number of Dataloader workers
epochs = 60                # number of epochs
batch_size = 6             # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (448, 448)     # size of training images
net = 'inception_mixed_6e'  # feature extractor
num_attentions = 32         # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'cartype'                # 'aircraft', 'bird', 'car', 'dog' or 'cartype'

# saving directory of .ckpt models
save_dir = './FGVC/CarType/ckpt/'
model_name = 'model1009.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = False
eval_ckpt = save_dir + model_name
eval_savepath = './FGVC/CarType/visualize/'

##################################################
# split file
##################################################
split_threshold = 0.6