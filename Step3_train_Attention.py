#!/usr/bin/env python
# |**********************************************************************;
# Project           : Why Do We Stop? Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import os
import sys
import argparse
import json
import numpy as np
from src.VA import *
from src.preprocessor_VA import *
from src.config_VA import *
from src.utils import *
import dataloader_VA as dataloader #TODO dataloader_VA (BDDX) or dataloader_VA_course2 (SAX)
from PIL import Image
import tensorflow as tf
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    tf.random.set_random_seed(42777)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create a Visual Attention (VA) model
    VA_model = VA(alpha_c=config.alpha_c)
    loss, alpha_reg, first_summary_train, first_summary_val, first_sum_acc, first_sum_course = VA_model.build_model()
    tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE)

    # Exponential learning rate decaying
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = config.lr
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               10000, 0.96, staircase=True) #10000

    momentum = 0.9

    # train op
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.MomentumOptimizer(
        #   learning_rate=learning_rate, momentum=momentum, use_locking=False, name='Momentum', use_nesterov=False
        #)
        grads = tf.gradients(loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

    # Preprocessor
    pre_processor = PreProcessor_VA()

    # Open a tensorflow session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=tfconfig)
    tf.global_variables_initializer().run()

    # saver
    saver = tf.train.Saver(max_to_keep=10)
    if config.pretrained_model_path is not None:
        saver.restore(sess, config.pretrained_model_path)
        print("\rLoaded the pretrained model: {}".format(config.pretrained_model_path))

    # Train over the dataset
    data_train = dataloader.data_iterator(model="VA") # X is feat vector, not image
    data_val = dataloader.data_iterator(validation=True, model="VA")

    log_dir = config.model_path + 'logs/' + current_time + "/"
    train_summary_writer = tf.summary.FileWriter(log_dir)

    val_loss = 999999999.0

    for i in range(config.maxiter):
        i = i + 1
        # Load new dataset
        img, course, speed, curvature, acc, goaldir = next(data_train)

        # Preprocessing
        img_p, _, acc_p, speed_p, course_p, _, goaldir_p, _ = pre_processor.process(sess, img, course, speed, curvature,
                                                                                    acc, goaldir)

        # Training
        feed_dict = {VA_model.features: img_p,
                     VA_model.target_course: course_p,
                     VA_model.target_acc: acc_p,
                     VA_model.speed: speed_p,
                     VA_model.goaldir: goaldir_p}
        _, l1loss, alpha_reg_loss, summary_train, _, _, _ = sess.run([train_op, loss, alpha_reg, first_summary_train, first_summary_val, first_sum_acc, first_sum_course], feed_dict)

        train_summary_writer.add_summary(summary_train, i)

        print('\rStep {}, Loss: {} ({})'.format(i, l1loss, alpha_reg_loss))

        # validation
        if (i % config.val_steps == 0):
            img, course, speed, curvature, acc, goaldir = next(data_val)
            img_p, _, acc_p, speed_p, course_p, _, goaldir_p, _ = pre_processor.process(sess, img, course, speed,
                                                                                        curvature,
                                                                                        acc, goaldir)

            feed_dict = {VA_model.features: img_p,
                         VA_model.target_course: course_p,
                         VA_model.target_acc: acc_p,
                         VA_model.speed: speed_p,
                         VA_model.goaldir: goaldir_p}
            l1loss_val, alpha_reg_val, _, summary_val, sum_acc, sum_course = sess.run([loss, alpha_reg, first_summary_train, first_summary_val, first_sum_acc, first_sum_course], feed_dict)

            train_summary_writer.add_summary(summary_val, i)
            train_summary_writer.add_summary(sum_acc, i)
            train_summary_writer.add_summary(sum_course, i)

            print(
                "\rStep {} | train loss: {} | val loss: {} (attn reg: {})".format(i, l1loss, l1loss_val, alpha_reg_val))
            sys.stdout.flush()

        if i % config.save_steps == 0 and val_loss > l1loss_val:
            print("Last Val Loss: " + str(val_loss))
            val_loss = l1loss_val
            #checkpoint_path = os.path.join(config.model_path, "model.ckpt")
            checkpoint_path = os.path.join(
                "/root/Workspace/explainable-deep-driving-master/model/VA/{}/model.ckpt".format(current_time))
            filename = saver.save(sess, checkpoint_path)
            print("Model saved in file: %s" % filename)


if __name__ == "__main__":
    main()
