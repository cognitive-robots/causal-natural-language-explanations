#!/usr/bin/env python
# Project           : Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import argparse
import os
import sys
from sys import platform
import numpy as np
import h5py
from tqdm import tqdm
import tensorflow as tf
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy import interpolate

from src.NVIDIA_CNN import *
from src.config_CNN import *
from src.utils import *
from src.preprocessor_CNN import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(args):
    parser = argparse.ArgumentParser(description='Path viewer')
    parser.add_argument('--getscore', type=bool, default=False, help='get performance scores')
    parser.add_argument('--showvideo', type=bool, default=False, help='show video')
    parser.add_argument('--useCPU', type=bool, default=False, help='without GPU processing')
    parser.add_argument('--validation', type=bool, default=False, help='use validation set')
    parser.add_argument('--test', type=bool, default=False, help='use test set')
    parser.add_argument('--extractFeature', type=bool, default=True, help='extract conv features')
    parser.add_argument('--gpu_fraction', type=float, default=0.7, help='GPU usage limit')
    args = parser.parse_args(args)

    if platform == 'linux':
        timestamp = "20220331-055148/"
        args.model = "./model/CNN/" + timestamp + "model.ckpt"
        args.savepath = "./result/CNN/"
        config.batch_size = 1
    else:
        raise NotImplementedError

    if args.getscore:       check_and_make_folder(args.savepath)
    if args.extractFeature: check_and_make_folder(config.h5path + "feat/")

    # prepare datasets
    if args.validation:
        filenames = os.path.join(config.h5path, 'val_names.txt')
    elif args.test:
        filenames = os.path.join(config.h5path, 'test_names.txt')
    else:
        filenames = os.path.join(config.h5path, 'train_names.txt')

    with open(filenames, 'r') as f:
        fname = ['%s' % x.strip() for x in f.readlines()]

    # Open a tensorflow session
    if args.useCPU == True:  # Use CPU only
        tfconfig = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=3)
        sess = tf.Session(config=tfconfig)
    else:  # Use GPU
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Create a CNN+FF model
    USE_SINGLE_FRAME = False  # if False, model use 4 consecutive frames as an input
    NVIDIA_model = NVIDIA_CNN(sess, USE_SINGLE_FRAME=USE_SINGLE_FRAME)

    # Preprocessor
    if USE_SINGLE_FRAME:
        pre_processor = PreProcessor_CNN()
    else:
        pre_processor = PreProcessor_CNN_4frame()

    # Load the pretrained model
    saver = tf.train.Saver(max_to_keep=1)
    if args.model is not None:
        saver.restore(sess, args.model)
        print("\rLoaded the pretrained model: {}".format(args.model))

    accumResult = []
    Result = namedtuple("Result", ["fname", "pred_acc", "target_acc", "pred_course", "target_course", "speed"])
    cnt = 0
    for dataset in fname:

        # load dataset
        log = h5py.File(config.h5path + "log/" + dataset + ".h5", "r")
        cam = h5py.File(config.h5path + "cam/" + dataset + ".h5", "r")
        nImg = cam['X'].shape[0]

        # preprocess logs
        curvature_value = preprocess_others(log["curvature"][:], nImg)
        accelerator_value = preprocess_others(log["accelerator"][:], nImg)
        speed_value = preprocess_others(log["speed"][:], nImg)
        course_value = preprocess_course(log["course"][:], nImg)
        goaldir_value = preprocess_others(log["goaldir"][:], nImg)

        if args.extractFeature: feats = []

        for i in tqdm(range(0, nImg - 3)):
            img = cam['X'][i:i + 4]

            # ego-motions
            curvature = curvature_value[i:i + 4]
            accel = accelerator_value[i:i + 4]
            speed = speed_value[i:i + 4]
            course = course_value[i:i + 4]
            goaldir = goaldir_value[i:i + 4]

            # preprocessing
            X, curvatures, accelerators, speeds, courses, _, goaldirs, _ = pre_processor.process(sess,
                                                                                                 img[None, :, :, :, :],
                                                                                                 course[None, :, :],
                                                                                                 speed[None, :, :],
                                                                                                 curvature[None, :, :],
                                                                                                 accel[None, :, :],
                                                                                                 goaldir[None, :, :])

            # inference
            pred_course, pred_acc = NVIDIA_model.predict(sess, x=X, s=speeds, g=goaldirs)

            # rescaling
            pred_acc = np.float(np.squeeze(pred_acc)) / 10.0
            pred_course = np.float(np.squeeze(pred_course)) / 10.0
            accelerators = np.float(np.squeeze(accelerators)) / 10.0
            courses = np.float(np.squeeze(courses)) / 10.0
            speeds = np.float(np.squeeze(speeds))

            if args.getscore:
                accumResult.append(
                    Result(dataset + '_%.5d' % (i), pred_acc, accelerators, pred_course, courses, speeds))

            if args.extractFeature:
                feat = NVIDIA_model.extractFeats(sess, x=X)
                feat = np.squeeze(np.array(feat))
                feat = feat.transpose(2, 0, 1)
                feats.append(feat)

            if args.showvideo:
                img2draw = img[-1]

                plt.cla()  # Clear axis
                plt.clf()  # Clear figure

                TEXT_MARGIN = 7
                plt.text(10, TEXT_MARGIN, 'pred acc {:6.2f} m/s2'.format(pred_acc),
                         bbox=dict(facecolor='red', alpha=0.5))
                plt.text(10, TEXT_MARGIN * 2, 'target acc {:6.2f} m/s2'.format(accelerators),
                         bbox=dict(facecolor='yellow', alpha=0.5))
                plt.text(10, TEXT_MARGIN * 3, 'pred course {:6.2f} degree'.format(pred_course),
                         bbox=dict(facecolor='red', alpha=0.5))
                plt.text(10, TEXT_MARGIN * 4, 'target course {:6.2f} degree'.format(courses),
                         bbox=dict(facecolor='yellow', alpha=0.5))
                plt.text(10, TEXT_MARGIN * 5, 'speed {:6.2f} m/s'.format(speeds),
                         bbox=dict(facecolor='green', alpha=0.5))
                plt.imshow(img2draw)
                plt.axis('off')
                plt.draw()
                plt.pause(0.01)

        if args.getscore:
            _fname, _pred_acc, _target_acc, _pred_course, _target_course, _speed = map(np.array, zip(*accumResult))

            a = np.concatenate(([_pred_acc], [_target_acc], [_pred_course], [_target_course], [_speed]), axis=0)
            np.savetxt(args.savepath + dataset + '.csv', a, delimiter=',', fmt="%s")
            accumResult = []

        if args.extractFeature:
            print(config.h5path + "feat/" + dataset + ".h5")
            f = h5py.File(config.h5path + "feat/" + dataset + ".h5", "w")
            dset = f.create_dataset("/X", data=feats, chunks=(20, 64, 12, 20))



if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
