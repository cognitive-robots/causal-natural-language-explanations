import sys
from src.config_VA import *
import numpy as np
import h5py
from scipy import interpolate
import os
import logging
import time
import traceback
import argparse

logger = logging.getLogger(__name__)
subs = int(config.subsample)


# given a series and alpha, return series of smoothed points
def exponential_smoothing(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return np.array(result)

def concatenate(model, camera_names, time_len):
    if model == "VA":
        logs_names = [x.replace('feat', 'log') for x in camera_names]
    elif model == "CNN":
        logs_names = [x.replace('cam', 'log') for x in camera_names]

    lastidx = 0
    hdf5_camera, c5x, filters = [], [], []
    course, speed, curvature, accelerator, goaldir = [], [], [], [], []

    for cword, tword in zip(camera_names, logs_names):
        try:
            with h5py.File(tword, "r") as t5:
                c5 = h5py.File(cword, "r")
                x = c5["X"]

                c5x.append((lastidx, lastidx + x.shape[0], x))
                hdf5_camera.append(c5)


                curvature_value = t5["curvature"][:]
                accelerator_value = t5["accelerator"][:]
                speed_value = t5["speed"][:]
                course_value = t5["course"][:]
                goaldir_value = t5["goaldir"][:]

                nRecords = speed_value.shape[0]
                nImg = x.shape[0]

                # refine course information
                for idx in range(1, nRecords):
                    if course_value[idx] - course_value[idx - 1] > 180:
                        course_value[idx:] -= 360
                    elif course_value[idx] - course_value[idx - 1] < -180:
                        course_value[idx:] += 360

                # interpolation
                xaxis = np.arange(0, nRecords)
                speed_interp = interpolate.interp1d(xaxis, speed_value)
                course_interp = interpolate.interp1d(xaxis, course_value)
                accelerator_interp = interpolate.interp1d(xaxis, accelerator_value)
                curvature_interp = interpolate.interp1d(xaxis, curvature_value)
                goaldir_interp = interpolate.interp1d(xaxis, goaldir_value)

                idxs = np.linspace(0, nRecords - 1, nImg).astype("float")  # approximate alignment

                speed_value = speed_interp(idxs)
                course_value = course_interp(idxs)
                curvature_value = curvature_interp(idxs)
                accelerator_value = accelerator_interp(idxs)
                goaldir_value = goaldir_interp(idxs)

                # Exponential Smoothing
                if config.use_smoothing == "Exp":  # Single Exponential Smoothing
                    print("Exp Smoothing...", config.use_smoothing)
                    speed_value = exponential_smoothing(speed_value, config.alpha)
                    course_value = exponential_smoothing(course_value, config.alpha)
                    curvature_value = exponential_smoothing(curvature_value, config.alpha)
                    accelerator_value = exponential_smoothing(accelerator_value, config.alpha)
                    goaldir_value = exponential_smoothing(goaldir_value, config.alpha)

                # exponential smoothing in reverse order
                course_value_smooth = np.flip(exponential_smoothing(np.flip(course_value, 0), 0.01), 0) # Exp Smoothing with 0.01 is used
                course_delta = course_value - course_value_smooth

                # accumulation
                course.append(course_delta)
                speed.append(speed_value)
                curvature.append(curvature_value)
                accelerator.append(accelerator_value)
                goaldir.append(goaldir_value)

                # Choose good imgages?
                goods = (np.abs(speed[-1]) >= -1)

                filters.append(np.argwhere(goods)[time_len - 1:] + (lastidx + time_len - 1))
                lastidx += goods.shape[0]

                # check for mismatched length bug
                print("x {} | c {} | s {} | c {} | a {} | f {} | g {}".format(
                    x.shape[0], course_value.shape[0], speed_value.shape[0],
                    curvature_value.shape[0], accelerator_value.shape[0], goods.shape[0], goaldir_value.shape[0]))

                if nImg != curvature[-1].shape[0] or nImg != accelerator[-1].shape[0] or nImg != course[-1].shape[
                    0] or nImg != goaldir[-1].shape[0]:
                    raise Exception("bad shape")

        except IOError:
            traceback.print_exc()
            print("failed to open", tword)

    course = np.concatenate(course, axis=0)
    speed = np.concatenate(speed, axis=0)
    curvature = np.concatenate(curvature, axis=0)
    accelerator = np.concatenate(accelerator, axis=0)
    goaldir = np.concatenate(goaldir, axis=0)
    filters = np.concatenate(filters, axis=0).ravel()

    # print "training on %d/%d examples" % (filters.shape[0], course.shape[0])
    print("training on " + str(filters.shape[0]) + " / " + str(course.shape[0]) + " examples")

    return c5x, course, speed, curvature, accelerator, filters, hdf5_camera, goaldir


def data_iterator(validation=False, test=False, model="CNN"):
    first = True
    parser = argparse.ArgumentParser(description='Dataloader')
    parser.add_argument('--validation', dest='validation', action='store_true', default=False,
                        help='Serve validation dataset instead.')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                        help='Serve test dataset instead.')
    #parser.add_argument('--nogood', dest='nogood', action='store_true', default=False, help='Ignore `goods` filters.')
    args, more = parser.parse_known_args()


    if validation:
        feat_path = config.h5path + "cap/feat/" + "val" + ".h5"
        logs_path = config.h5path + "cap/log/" + "val" + ".h5"
    elif test:
        feat_path = config.h5path + "cap/feat/" + "test" + ".h5"
        logs_path = config.h5path + "cap/log/" + "test" + ".h5"
    else:
        feat_path = config.h5path + "cap/feat/" + "train" + ".h5"
        logs_path = config.h5path + "cap/log/" + "train" + ".h5"

    arrays = []

    # Datagen
    batch_size = config.batch_size_gen

    # Load data
    feat = h5py.File(feat_path, 'r')
    logs = h5py.File(logs_path, 'r')

    alphas_attn = logs["/attn"]
    contexts = logs["/context"] # for SAA
    pred_acc = logs["/pred_accel"]
    pred_course = logs["/pred_courses"]
    captions = logs["/Caption"]
    pos = logs["/POS"]
    features = feat["/X"]
    images = feat["/img"]
    feat_masks = feat["/mask"]
    goaldir = logs["/goaldir"]
    speed = logs["/speed"]
    timestamp = logs["/timestamp"]
    acc = logs['/accelerator']
    course = logs['/course']

    alphas_batch = np.zeros((batch_size, 10, 240), dtype='float32')
    context_batch = np.zeros((batch_size, 10, 64), dtype='float32')
    pred_course_batch = np.zeros((batch_size, 10), dtype='float32')
    pred_accel_batch = np.zeros((batch_size, 10), dtype='float32')
    course_batch = np.zeros((batch_size, 10), dtype='float32')
    accel_batch = np.zeros((batch_size, 10), dtype='float32')
    captions_batch = np.zeros((batch_size, 1, 22), dtype='int32')
    pos_batch = np.zeros((batch_size, 1, 22), dtype='int32')
    feat_batch = np.zeros((batch_size, 10, 64, 12, 20), dtype='float32')
    img_batch = np.zeros((batch_size, 10, 90, 160, 3), dtype='int32')
    masks_batch = np.zeros((batch_size, 10), dtype='int32')
    goaldir_batch = np.zeros((batch_size, 10), dtype='float32')
    speed_batch = np.zeros((batch_size, 10), dtype='float32')
    timestamp_batch = np.zeros((batch_size, 10), dtype='float32')
    seq_id_batch = np.zeros((batch_size, 1), dtype='int32')


    try_counter = 0
    if validation==True:
        num_of_batches = int(config.batches_per_dataset_v)
    elif test==True:
        num_of_batches = int(config.batches_per_dataset_t)
    else:
        num_of_batches = int(config.batches_per_dataset)
    #num_of_batches = int(captions.shape[0]/batch_size)
    i = np.random.randint(0, captions.shape[0], 1)

    while True:
        try:
            t = time.time()
            if try_counter == num_of_batches:
                try_counter = 0
                i = np.random.randint(0, captions.shape[0], 1)

            count = 0
            start = time.time()
            while count < batch_size:
                if i==captions.shape[0]:
                    i = 0

                if timestamp[i].shape == (1,10):

                    if timestamp[i][0][0] < 100.0: #if data is empty
                        i += 1
                        continue
                else:
                    if timestamp[i][0] < 100.0:  # if data is empty
                        i += 1
                        continue

                # GET X_BATCH

                alphas_batch[count] = alphas_attn[i]
                context_batch[count] = contexts[i]
                pred_course_batch[count] = pred_course[i]
                pred_accel_batch[count] = pred_acc[i]
                course_batch[count] = course[i]
                accel_batch[count] = acc[i]
                captions_batch[count] = captions[i]
                pos_batch[count] = pos[i]
                feat_batch[count] = features[i]
                img_batch[count] = images[i]
                masks_batch[count] = feat_masks[i]
                goaldir_batch[count] = goaldir[i]
                speed_batch[count] = speed[i]
                timestamp_batch[count] = timestamp[i]
                seq_id_batch[count] = i

                count += 1
                i += 1

            # sanity check
            assert context_batch.shape == (batch_size, 10, 64)

            # Subsample feature and context arrays
            alphas_batch_s = alphas_batch[:, :, ::(subs**2)]
            feat_batch_s = feat_batch[:, :, ::subs, ::subs, ::subs]
            img_batch_s = img_batch[:, :, ::subs, ::subs, :]
            context_batch_s = context_batch[:, :, ::(subs)]

            if first:
                print("Context vectors", context_batch.shape)
                print("Attn weights", alphas_batch.shape)
                print("pred. accel", pred_accel_batch.shape)
                print("features", feat_batch.shape)
                print("captions", captions_batch.shape)
                first = False

            array = (alphas_batch_s, context_batch_s, pred_course_batch, pred_accel_batch, captions_batch, feat_batch_s, masks_batch, goaldir_batch, speed_batch, timestamp_batch, seq_id_batch, course_batch, accel_batch, img_batch_s, pos_batch)

            try_counter = try_counter + 1

            yield array

        except KeyboardInterrupt:
            raise

        except:
            traceback.print_exc()
            pass


def visualizer(array):
    from PIL import Image

    img = array[0][10][2]
    course = array[1][10][2]
    speed = array[2][10][2]
    curvature = array[3][10][2]
    accel = array[4][10][2]
    goaldir = array[5][10][2]

    frame = Image.fromarray(img)
    frame.save("frame.jpeg")

    print("Course: " + str(course))
    print("Speed: " + str(speed))
    print("Curvature: " + str(curvature))
    print("Accel: " + str(accel))
    print("Goaldir: " + str(goaldir))

    return None
