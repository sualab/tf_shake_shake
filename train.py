import os
import numpy as np
import tensorflow as tf
from datasets import cifar10 as dataset
from models.nn import ShakeNet as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import AccuracyEvaluator as Evaluator

from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data


""" 1. Load and split datasets """
"""
root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')    # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# Load trainval set and split into train/val sets
X_trainval, y_trainval = dataset.read_asirra_subset(trainval_dir, one_hot=True)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.2)    # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])
"""

# Load training set(train_set) and test set(val_set)
X_train, X_val, Y_train, Y_val = dataset.read_CIFAR10_subset()
train_set = dataset.DataSet(X_train, Y_train)
val_set = dataset.DataSet(X_val, Y_val)

# Sanity check
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())

#for i in range(10):
#   print((train_set.labels[:, i] ==1).sum(), '\t')

print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())

#for i in range(10):
#   print((val_set.labels[:, i] ==1).sum(), '\t')


""" 2. Set training hyperparameters """
hp_d = dict()
image_mean = train_set.images.mean(axis=(0, 1, 2))    # mean image
np.save('./CIFAR10_mean.npy', image_mean)    # save mean image
hp_d['image_mean'] = image_mean

# FIXME: Training hyperparameters
hp_d['batch_size'] = 256
hp_d['num_epochs'] = 1800

hp_d['augment_train'] = True
#hp_d['augment_pred'] = False

hp_d['init_learning_rate'] = 0.01
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: Regularization hyperparameters
hp_d['weight_decay'] = 0.0005
hp_d['dropout_prob'] = 0.5

# FIXME: Evaluation hyperparameters
hp_d['score_threshold'] = 1e-4


""" 3. Build graph, initialize a session and start training """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([32, 32, 3], 10, **hp_d)
evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

sess = tf.Session(graph=graph, config=config)
train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)
