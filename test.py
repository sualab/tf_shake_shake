import os
import numpy as np
import tensorflow as tf
from datasets import asirra as dataset
from models.nn import AlexNet as ConvNet
from learning.evaluators import AccuracyEvaluator as Evaluator


""" 1. Load dataset """
root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')    # FIXME
test_dir = os.path.join(root_dir, 'test')

# Load test set
X_test, y_test = dataset.read_asirra_subset(test_dir, one_hot=True)
test_set = dataset.DataSet(X_test, y_test)

# Sanity check
print('Test set stats:')
print(test_set.images.shape)
print(test_set.images.min(), test_set.images.max())
print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())


""" 2. Set test hyperparameters """
hp_d = dict()
image_mean = np.load('/tmp/asirra_mean.npy')    # load mean image
hp_d['image_mean'] = image_mean

# FIXME: Test hyperparameters
hp_d['batch_size'] = 256
hp_d['augment_pred'] = True


""" 3. Build graph, load weights, initialize a session and start test """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, '/tmp/model.ckpt')    # restore learned weights
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test accuracy: {}'.format(test_score))
