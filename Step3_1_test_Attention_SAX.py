#!/usr/bin/env python
# |**********************************************************************;
# Project           : Why Stop Now? Causal Natural Language Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import argparse
import sys
import os
import numpy as np
import h5py
import tensorflow as tf
from collections import namedtuple
from src.utils import *
from src.preprocessor_VA import *
from src.config_VA import *
from src.VA import *
from sys import platform
from tqdm import tqdm
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(args):
    parser = argparse.ArgumentParser(description='Path viewer')
    parser.add_argument('--getscore', type=bool, default=False, help='get performance scores')
    parser.add_argument('--showvideo', type=bool, default=False, help='show video')
    parser.add_argument('--useCPU', type=bool, default=False, help='without GPU processing')
    parser.add_argument('--validation', type=bool, default=False, help='use validation set')
    parser.add_argument('--test', type=bool, default=False, help='use validation set')
    parser.add_argument('--gpu_fraction', type=float, default=0.9, help='GPU usage limit')
    parser.add_argument('--extractAttn', type=bool, default=False, help='extract attention maps')
    parser.add_argument('--extractAttnMaps', type=bool, default=False, help='extract attention maps on images')
    args = parser.parse_args(args)

    if platform == 'linux':
        timestamp = "20220331-064616"
        args.model = "./model/VA/" + timestamp + "/model.ckpt"
        args.savepath = "./result/VA/"
        config.timelen = 3500 + 3
        timelen = 3500
        config.batch_size = 1
    else:
        raise NotImplementedError

    if args.getscore:    check_and_make_folder(args.savepath)
    if args.extractAttn: check_and_make_folder(config.h5path + "attn/")
    check_and_make_folder(config.h5path + "img_w_attn/train/")
    check_and_make_folder(config.h5path + "img_w_attn/test/")
    check_and_make_folder(config.h5path + "img_w_attn/val/")

    # prepare datasets
    if args.validation:
        filenames = os.path.join(config.h5path, 'val_names.txt')  # validation
    elif args.test:
        filenames = os.path.join(config.h5path, 'test_names.txt')
    else:
        filenames = os.path.join(config.h5path, 'train_names.txt')

    with open(filenames, 'r') as f:
        fname = ['%s' % x.strip() for x in f.readlines()]

    # Create VA model
    VA_model = VA(alpha_c=config.alpha_c)
    alphas, contexts, y_acc, y_course, mae_accel, mae_course = VA_model.inference_SAX()

    if args.useCPU:  # Use CPU only
        tfconfig = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=1)
        sess = tf.Session(config=tfconfig)
    else:  # Use GPU
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Preprocessor
    pre_processor = PreProcessor_VA(timelen=timelen, phase='test')

    # Load the pretrained model
    saver = tf.train.Saver()
    if args.model is not None:
        saver.restore(sess, args.model)
        print("\rLoaded the pretrained model: {}".format(args.model))

    mae_accel_out_sum = 0.0
    mae_course_out_sum = 0.0
    dataset_counter = 0

    for dataset in tqdm(fname):
        dataset_counter += 1
        print(bcolors.HIGHL + "Dataset: {}".format(dataset) + bcolors.ENDC)

        log = h5py.File(config.h5path + "log/" + dataset + ".h5", "r")
        feats = h5py.File(config.h5path + "feat/" + dataset + ".h5", "r")
        cam = h5py.File(config.h5path + "cam/" + dataset + ".h5", "r")
        nImg = cam['X'].shape[0]
        nFeat = feats['X'].shape[0]

        # initialization
        feat_batch = np.zeros((timelen, 64, 12, 20))
        curvature_batch = np.zeros((timelen, 1))
        accel_batch = np.zeros((timelen, 1))
        speed_batch = np.zeros((timelen, 1))
        course_batch = np.zeros((timelen, 1))
        goaldir_batch = np.zeros((timelen, 1))
        timestamp_batch = np.zeros((timelen, 1))
        cam_batch = np.zeros((timelen, 90, 160, 3))

        # preprocess logs

        if nImg > timelen:  # Interpolate log values to max timelen
            nImg = timelen

        if nFeat > timelen:
            nFeat = timelen

        if nImg < nFeat:  # make sure that each image has one feature vector
            nFeat = nImg
        elif nImg > nFeat:
            nImg = nFeat

        feat_batch[:nFeat] = feats['X'][:nFeat]
        cam_batch[:nFeat] = cam['X'][:nFeat]
        timestamp_batch[:nFeat] = preprocess_others(log["timestamp"][::10], nImg)  # [3:]
        curvature_batch[:nFeat] = preprocess_others(log["curvature"][::10], nImg)  # [3:]
        accel_batch[:nFeat] = preprocess_others(log["accelerator"][::10], nImg)  # [3:]
        speed_batch[:nFeat] = preprocess_others(log["speed"][::10], nImg)  # [3:]
        course_batch[:nFeat] = preprocess_course_SAX(log["course"][::10], nImg)  # [3:]
        goaldir_batch[:nFeat] = preprocess_others(log["goaldir"][::10], nImg)  # [3:]

        # Preprocessing for tensorflow
        feat_p, _, acc_p, speed_p, course_p, _, goaldir_p, _ = pre_processor.process(
            sess=sess,
            inImg=np.expand_dims(np.array(feat_batch), 0),
            course=np.expand_dims(np.array(course_batch), 0),
            speed=np.expand_dims(np.array(speed_batch), 0),
            curvature=np.expand_dims(np.array(curvature_batch), 0),
            accelerator=np.expand_dims(np.array(accel_batch), 0),
            goaldir=np.expand_dims(np.array(goaldir_batch), 0))

        # Run a model

        alps = np.zeros([timelen, 240])
        ctxts = np.zeros([timelen, 64])
        pred_accel = np.zeros([timelen, 1])
        pred_courses = np.zeros([timelen, 1])
        start_idx = 0
        end_idx = 0
        for split in range(int(timelen / 500)):
            end_idx += 500

            feed_dict = {VA_model.features: feat_p[start_idx:end_idx],
                         VA_model.target_course: course_p[start_idx:end_idx],
                         VA_model.target_acc: acc_p[start_idx:end_idx],
                         VA_model.speed: speed_p[start_idx:end_idx],
                         VA_model.goaldir: goaldir_p[start_idx:end_idx]}

            alps_split, ctxts_split, pred_accel_split, pred_courses_split, mae_accel_out, mae_course_out = sess.run(
                [alphas, contexts, y_acc, y_course, mae_accel, mae_course], feed_dict)

            alps_split = np.squeeze(alps_split)
            ctxts_split = np.squeeze(ctxts_split)

            mae_accel_out_sum = mae_accel_out_sum + mae_accel_out
            mae_course_out_sum = mae_course_out_sum + mae_course_out

            alps[start_idx:end_idx,:] = alps_split[:,:]
            ctxts[start_idx:end_idx,:] = ctxts_split[:,:]
            pred_accel[start_idx:end_idx] = np.expand_dims(pred_accel_split, 1)
            pred_courses[start_idx:end_idx] = np.expand_dims(pred_courses_split, 1)

            print("Split " + str(split) + " done.")

            start_idx += 500

        # Unscale
        pred_accel = pred_accel / 100
        pred_courses = pred_courses / 10

        if args.extractAttnMaps:
            img_att_np = visualize_attnmap_2_SAX(alps[12, :], cam_batch[12, :, :, :])
            img_att = Image.fromarray(img_att_np)
            if args.validation:
                set_n = "val/"  # validation
            elif args.test:
                set_n = "test/"
            else:
                set_n = "train/"
            img_att.save(
                "/root/Workspace/explainable-deep-driving-master/data/processed_SAX_cropped/img_w_attn/" + str(set_n) + str(dataset) + "_1.png") #TODO Change Filepath

            img_att_np = visualize_attnmap_2_SAX(alps[512, :], cam_batch[512, :, :, :])
            img_att = Image.fromarray(img_att_np)
            if args.validation:
                set_n = "val/"  # validation
            elif args.test:
                set_n = "test/"
            else:
                set_n = "train/"
            img_att.save(
                "/root/Workspace/explainable-deep-driving-master/data/processed_SAX_cropped/img_w_attn/" + str(set_n) + str(
                    dataset) + "_2.png")

            img_att_np = visualize_attnmap_2_SAX(alps[2012, :], cam_batch[2012, :, :, :])
            img_att = Image.fromarray(img_att_np)
            if args.validation:
                set_n = "val/"  # validation
            elif args.test:
                set_n = "test/"
            else:
                set_n = "train/"
            img_att.save(
                "/root/Workspace/explainable-deep-driving-master/data/processed_SAX_cropped/img_w_attn/" + str(set_n) + str(
                    dataset) + "_3.png")

            img_att_np = visualize_attnmap_2_SAX(alps[2712, :], cam_batch[2712, :, :, :])
            img_att = Image.fromarray(img_att_np)
            if args.validation:
                set_n = "val/"  # validation
            elif args.test:
                set_n = "test/"
            else:
                set_n = "train/"
            img_att.save(
                "/root/Workspace/explainable-deep-driving-master/data/processed_SAX_cropped/img_w_attn/" + str(set_n) + str(
                    dataset) + "_4.png")
            print("Attention Maps saved.")

        # Total Result acc in m/s^2 and course in deg
        if args.extractAttn:
            print(config.h5path + "attn/" + dataset + ".h5")
            f = h5py.File(config.h5path + "attn/" + dataset + ".h5", "w")
            dset = f.create_dataset("/attn", data=alps, chunks=(20, 240))
            dset = f.create_dataset("/context", data=ctxts, chunks=(20, 64))
            # dset = f.create_dataset("/cam", data=cam_batch, chunks=(20, 90, 160, 3))
            dset = f.create_dataset("/timestamp", data=timestamp_batch, chunks=(20, 1))
            dset = f.create_dataset("/curvature", data=curvature_batch, chunks=(20, 1))
            dset = f.create_dataset("/accel", data=accel_batch, chunks=(20, 1))
            dset = f.create_dataset("/speed", data=speed_batch, chunks=(20, 1))
            dset = f.create_dataset("/course", data=course_batch, chunks=(20, 1))
            dset = f.create_dataset("/goaldir", data=goaldir_batch, chunks=(20, 1))
            dset = f.create_dataset("/pred_accel", data=pred_accel, chunks=(20, 1))
            dset = f.create_dataset("/pred_courses", data=pred_courses, chunks=(20, 1))

            # NOTE: Each vector has size 400 and if original video is less than 40s, there will be zero entries in the vector which need to be skipped
            print("Attention data is saved.")

    # Unscale
    mae_accel_out_sum = mae_accel_out_sum / 100
    mae_course_out_sum = mae_course_out_sum / 10

    # Total Result in m/s^2 and deg/s
    mae_accel_out_sum = mae_accel_out_sum / dataset_counter / int(timelen / 500)
    mae_course_out_sum = mae_course_out_sum / dataset_counter / int(timelen / 500)

    print(bcolors.HIGHL + 'Done' + bcolors.ENDC)
    if args.test:
        print("MAE Accel: " + str(mae_accel_out_sum))
        print("MAE Course: " + str(mae_course_out_sum))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
