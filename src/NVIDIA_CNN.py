# |**********************************************************************;
# Project           : Why Stop Now? Causal Natural Language Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import tensorflow           as      tf
from   src.config_CNN           import  *

class NVIDIA_CNN(object):
  def __init__(self, sess, USE_SINGLE_FRAME=True):
    self.sess           = sess
    self.SENSORFUSE     = False
    self.weight_initializer = tf.contrib.layers.xavier_initializer()

    if USE_SINGLE_FRAME:
        self.x   = tf.placeholder(tf.float32,
            shape=[None, int(config.imgRow/config.resizeFactor), int(config.imgCol/config.resizeFactor), config.imgCh])
    else:
        self.x   = tf.placeholder(tf.float32,
            shape=[None, int(config.imgRow/config.resizeFactor), int(config.imgCol/config.resizeFactor), 4*config.imgCh])

    self.target_course       = tf.placeholder(tf.float32, shape=[None, 1])
    self.target_accelerator  = tf.placeholder(tf.float32, shape=[None, 1])
    self.speed               = tf.placeholder(tf.float32, shape=[None, 1])
    self.goaldir             = tf.placeholder(tf.float32, shape=[None, 1])
    self.create_net()

  def create_net(self):
    # conv1: 24 filters / 5x5 kernel / 2x2 stride
    conv1 = tf.contrib.layers.conv2d(self.x,24, 5, 2, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    # conv2: 36 filters / 5x5 kernel / 2x2 stride
    conv2 = tf.contrib.layers.conv2d(conv1, 36, 5, 2, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    # conv1: 48 filters / 5x5 kernel / 2x2 stride
    conv3 = tf.contrib.layers.conv2d(conv2, 48, 5, 2, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    # conv1: 24 filters / 3x3 kernel / 1x1 stride
    conv4 = tf.contrib.layers.conv2d(conv3, 64, 3, 1, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    # conv1: 24 filters / 3x3 kernel / 1x1 stride
    conv5 = tf.contrib.layers.conv2d(conv4, 64, 3, 1, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)

    # To extract features
    self.conv5 = conv5
    flattened  = tf.contrib.layers.flatten(conv5)

    if self.SENSORFUSE: flattened = tf.concat(1, [flattened, self.speed, self.goaldir])

    # networks for acceleration
    fc_a1       = tf.contrib.layers.fully_connected(flattened,  1164, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    fc_a2       = tf.contrib.layers.fully_connected(fc_a1,       100, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    fc_a3       = tf.contrib.layers.fully_connected(fc_a2,        50, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    fc_a4       = tf.contrib.layers.fully_connected(fc_a3,        10, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    self.out_a  = tf.contrib.layers.fully_connected(fc_a4,         1, activation_fn = None,       weights_initializer=self.weight_initializer)

    # networks for acceleration
    fc_c1       = tf.contrib.layers.fully_connected(flattened,  1164, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    fc_c2       = tf.contrib.layers.fully_connected(fc_c1,       100, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    fc_c3       = tf.contrib.layers.fully_connected(fc_c2,        50, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    fc_c4       = tf.contrib.layers.fully_connected(fc_c3,        10, activation_fn = tf.nn.relu, weights_initializer=self.weight_initializer)
    self.out_c  = tf.contrib.layers.fully_connected(fc_c4,         1, activation_fn = None,       weights_initializer=self.weight_initializer)


    self.l1_a   = tf.nn.l2_loss( tf.subtract(self.target_accelerator, self.out_a) )
    self.l1_c   = tf.nn.l2_loss( tf.subtract(self.target_course,      self.out_c) )
    self.l1     = self.l1_c + self.l1_a

    # Summary
    self.first_summary_train = tf.summary.scalar(name='Train loss', tensor=self.l1)
    self.first_summary_val = tf.summary.scalar(name='Validation loss', tensor=self.l1)

    # training op
    self.train_op  = tf.train.AdamOptimizer(config.lr).minimize(self.l1)
    self.sess.run(tf.global_variables_initializer())

  # optimize
  def process(self, sess, x, c, a, s, g):
    feed = {self.x:                  x, 
            self.target_course:      c, 
            self.target_accelerator: a,
            self.speed:              s,
            self.goaldir:            g}

    return sess.run([self.l1, self.l1_a, self.l1_c, self.train_op, self.first_summary_train], feed)

  # prediction
  def predict(self, sess, x, s, g):
    feed = {self.x: x, self.speed: s, self.goaldir: g}
    return sess.run([self.out_c, self.out_a], feed)

  # validation
  def validate(self, sess, x, c, a, s, g):
    feed = {self.x:                  x, 
            self.target_course:      c, 
            self.target_accelerator: a,
            self.speed:              s,
            self.goaldir:            g}

    return sess.run([self.l1, self.l1_a, self.l1_c, self.out_a, self.first_summary_val], feed)

  # extract features
  def extractFeats(self, sess, x):
    feed = {self.x: x}
    return sess.run([self.conv5], feed)










