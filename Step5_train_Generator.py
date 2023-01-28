#!/usr/bin/env python
# |**********************************************************************;
# Project           : Textual Explanations for Automated Commentary Driving
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
#from server import client_generator
from src.LSTM_Gen_slim_v2_pos_penalties import * #TODO Change whether _pos, _w_penalties or pos_penalties or not
from src.preprocessor_Gen import *
from src.config_VA import *
from src.utils import *
import dataloader_Gen as dataloader
from PIL import Image
import tensorflow as tf
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    tf.random.set_random_seed(42777)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    savepath_dict = "cap"

    Gen_model = LSTM_Gen(alpha_c=config.alpha_c)
    loss, first_summary_train, first_summary_val = Gen_model.build_model()
    tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE)

    # Exponential learning rate decaying
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = config.lr
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               config.batches_per_dataset*25, 0.96, staircase=True) #TODO Adapt parameter
    momentum = 0.9

    # train op
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.MomentumOptimizer(
        #    learning_rate=learning_rate, momentum=momentum, use_locking=False, name='Momentum', use_nesterov=False
        #)
        grads = tf.gradients(loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

    # Open a tensorflow session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=tfconfig)
    tf.global_variables_initializer().run()

    # saver
    saver = tf.train.Saver(max_to_keep=10)
    if config.pretrained_model_path_Gen is not None:
        saver.restore(sess, config.pretrained_model_path_Gen)
        print("\rLoaded the pretrained model: {}".format(config.pretrained_model_path_Gen))

    with open(os.path.join(config.h5path + '{}/{}/word_to_idx.pkl'.format(savepath_dict, 'train')), 'rb') as f:
        word_to_idx = pickle.load(f)

    idx_sep = word_to_idx['<sep>']

    # Train over the dataset
    data_train = dataloader.data_iterator(model="VA")
    data_val = dataloader.data_iterator(validation=True, model="VA")

    log_dir = config.model_path_Gen + 'logs/' + current_time + "/"
    train_summary_writer = tf.summary.FileWriter(log_dir)

    val_loss = 9999999.0
    loss_sum = 0.0
    loss_val_sum = 0.0
    batch_counter = 0
    batch_counter_val = 0

    for i in range(config.maxiter): #each epoch
        #i = i + 1
        loss_sum = 0.0
        loss_val_sum = 0.0

        for b in range(config.batches_per_dataset): #TODO change if Batch size is changed


            _, context, pred_course, pred_accel, caption, feat, masks, _, _, timestamp, _, _, _, _, pos = next(data_train)

            # Convert caption to One-Hot-Encoded with dict size 10000

            caption_onehot = (np.arange(config.dict_size) == caption[..., None]).astype(int)  # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
            caption_onehot = np.squeeze(caption_onehot)

            pos = pos - 1
            pos_onehot = (np.arange(12) == pos[..., None]).astype(int)
            pos_onehot = np.squeeze(pos_onehot)

            # Training
            feed_dict = {Gen_model.context: context,
                    Gen_model.pred_acc: pred_accel,
                    Gen_model.pred_course: pred_course,
                    Gen_model.features: feat,
                    Gen_model.caption: caption,
                    Gen_model.caption_onehot: caption_onehot,
                    Gen_model.pos_gt: pos_onehot,
                    Gen_model.idx_sep: idx_sep}

            _, l1loss, summary_train, _ = sess.run([train_op, loss, first_summary_train, first_summary_val], feed_dict)
            loss_sum += l1loss
            train_summary_writer.add_summary(summary_train, batch_counter)
            batch_counter += 1
            if batch_counter % 5000 == 0:
                checkpoint_path = os.path.join(
                    "/root/Workspace/explainable-deep-driving-master/model/LSTM_Gen/{}/train_fit/model.ckpt".format(
                        current_time))
                filename = saver.save(sess, checkpoint_path)


        # validation
        for bv in range(config.batches_per_dataset_v): #TODO change if Batch size is changed
            _, context, pred_course, pred_accel, caption, feat, masks, _, _, timestamp, _, _, _, _, pos = next(data_val)
            caption_onehot = (np.arange(config.dict_size) == caption[..., None]).astype(int)
            caption_onehot = np.squeeze(caption_onehot)

            pos = pos -1
            pos_onehot = (np.arange(12) == pos[..., None]).astype(int)
            pos_onehot = np.squeeze(pos_onehot)

            #    img_p, _, acc_p, speed_p, course_p, _, goaldir_p, _ = pre_processor.process(sess, img, course, speed,
            #                                                                                curvature,
            #                                                                                acc, goaldir)
            feed_dict = {Gen_model.context: context,
                         Gen_model.pred_acc: pred_accel,
                         Gen_model.pred_course: pred_course,
                         Gen_model.features: feat,
                         Gen_model.caption: caption,
                         Gen_model.caption_onehot: caption_onehot,
                         Gen_model.pos_gt: pos_onehot,
                         Gen_model.idx_sep: idx_sep}

            l1loss_val, _, summary_val = sess.run([loss, first_summary_train, first_summary_val], feed_dict)
            loss_val_sum += l1loss_val
            train_summary_writer.add_summary(summary_val, batch_counter_val)
            batch_counter_val += 1


        #train_summary_writer.add_summary(summary_val, i)
        line = "Step {} | train loss: {} | val loss: {} ".format(i, loss_sum/config.batches_per_dataset, loss_val_sum/config.batches_per_dataset_v)
        print("\rStep {} | train loss: {} | val loss: {} ".format(i, loss_sum/config.batches_per_dataset, loss_val_sum/config.batches_per_dataset_v))
        with open(log_dir + "losses.txt", 'a') as f:
            f.write(line)
            f.write('\n')
        sys.stdout.flush()

        if val_loss > loss_val_sum:
            print("Last Val Loss: " + str(val_loss/config.batches_per_dataset_v))
            val_loss = loss_val_sum
            #checkpoint_path = os.path.join(config.model_path_Gen, "model.ckpt")
            checkpoint_path = os.path.join(
                "/root/Workspace/explainable-deep-driving-master/model/LSTM_Gen/{}/model.ckpt".format(current_time))
            filename = saver.save(sess, checkpoint_path)
            print("Model saved in file: %s" % filename)
            with open(log_dir + "losses.txt", 'a') as f:
                f.write("Model saved with val loss " + str(loss_val_sum/config.batches_per_dataset_v))
                f.write('\n')


if __name__ == "__main__":
    main()
