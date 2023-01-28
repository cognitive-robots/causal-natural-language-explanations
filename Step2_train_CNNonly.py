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
from src.preprocessor_CNN import *
from src.NVIDIA_CNN import *
from src.config_CNN import *
from src.utils import *
import dataloader_CNN_course2 as dataloader #TODO dataloader_CNN (BDDX) or dataloader_CNN_course2 (SAX)
from PIL import Image
import  tensorflow        as tf
import datetime
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    tf.random.set_random_seed(42777)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Open a tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Create a CNN+FF model
    USE_SINGLE_FRAME = False  # if False, model use 4 consecutive frames as an input
    NVIDIA_model = NVIDIA_CNN(sess, USE_SINGLE_FRAME=USE_SINGLE_FRAME)

    # Preprocessor
    if USE_SINGLE_FRAME:
        pre_processor = PreProcessor_CNN()
    else:
        pre_processor = PreProcessor_CNN_4frame()

    # tensorflow saver
    saver = tf.train.Saver(max_to_keep=20)
    if config.pretrained_model_path is not None:
        saver.restore(sess, config.pretrained_model_path)
        print("\rLoaded the pretrained model: {}".format(config.pretrained_model_path))

    # Train over the dataset
    data_train = dataloader.data_iterator(model="CNN")
    data_val = dataloader.data_iterator(validation=True, model="CNN")

    # Implement TensorBoard
    log_dir = config.model_path + 'logs/' + current_time + "/"
    train_summary_writer = tf.summary.FileWriter(log_dir)

    # create folder
    check_and_make_folder(config.model_path)

    val_loss = 999999999.0

    for i in range(config.maxiter):
        i = i+1
        # Load new dataset

        X_batch, course_batch, speed_batch, curvature_batch, accelerator_batch, goaldir_batch = next(data_train)

        # Preprocessing
        Xprep_batch, curvatures, accelerators, speeds, courses, _, goaldirs, _ = pre_processor.process(sess, X_batch,
                                                                                                       course_batch,
                                                                                                       speed_batch,
                                                                                                       curvature_batch,
                                                                                                       accelerator_batch,
                                                                                                       goaldir_batch)


        l1loss, loss_acc, loss_cur, _, summary_train = NVIDIA_model.process(
            sess=sess,
            x=Xprep_batch,
            c=courses,
            a=accelerators,
            s=speeds,
            g=goaldirs)

        train_summary_writer.add_summary(summary_train, i)

        if (i % config.val_steps == 0):
            X_val, course_val, speed_val, curvature_val, accelerator_val, goaldir_val = next(data_val)

            # preprocessing
            Xprep_val, curvatures_val, accelerators_val, speeds_val, \
            courses_val, _, goaldirs_val, _ = pre_processor.process(
                sess, X_val, course_val, speed_val, curvature_val, accelerator_val, goaldir_val)

            l1loss_val, l1loss_val_acc, l1loss_val_cur, a_pred, summary_val = NVIDIA_model.validate(
                sess=sess,
                x=Xprep_val,
                c=courses_val,
                a=accelerators_val,
                s=speeds_val,
                g=goaldirs_val)


            print("\rStep {} | train loss: {} | val loss: {} (acc: {}, cur: {})".format(i, l1loss, l1loss_val,
                                                                                        l1loss_val_acc, l1loss_val_cur))

            train_summary_writer.add_summary(summary_val, i)

            sys.stdout.flush()

        if i % config.save_steps == 0 and val_loss > l1loss_val:
            print("Last val loss: " + str(val_loss))
            val_loss = l1loss_val
            checkpoint_path = os.path.join("/root/Workspace/explainable-deep-driving-master/model/CNN/{}/model.ckpt".format(current_time))
            filename = saver.save(sess, checkpoint_path)
            print("Current model is saved: {}".format(filename))


    print('Success!')
    # End of code


if __name__ == "__main__":
    main()
