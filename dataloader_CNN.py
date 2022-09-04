from src.config_CNN import *
import numpy as np
import h5py
from scipy import interpolate
import os
import logging
import time
import traceback
import argparse

logger = logging.getLogger(__name__)


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

                idxs = np.linspace(0, nRecords - 1, nImg).astype("float")

                speed_value = speed_interp(idxs)
                course_value = course_interp(idxs)
                curvature_value = curvature_interp(idxs)
                accelerator_value = accelerator_interp(idxs)
                goaldir_value = goaldir_interp(idxs)

                # Exponential Smoothing
                if config.use_smoothing == "Exp":  # Single Exponential Smoothing; Not used for CNN
                    print("Exp Smoothing...", config.use_smoothing)
                    speed_value = exponential_smoothing(speed_value, config.alpha)
                    course_value = exponential_smoothing(course_value, config.alpha)
                    curvature_value = exponential_smoothing(curvature_value, config.alpha)
                    accelerator_value = exponential_smoothing(accelerator_value, config.alpha)
                    goaldir_value = exponential_smoothing(goaldir_value, config.alpha)

                # exponential smoothing in reverse order
                course_value_smooth = np.flip(exponential_smoothing(np.flip(course_value, 0), 0.01), 0)
                course_delta = course_value - course_value_smooth

                # accumulation
                course.append(course_delta)
                speed.append(speed_value)
                curvature.append(curvature_value)
                accelerator.append(accelerator_value)
                goaldir.append(goaldir_value)

                # Choose good images?
                goods = (np.abs(speed[-1]) >= -1)

                filters.append(np.argwhere(goods)[time_len - 1:] + (lastidx + time_len - 1))
                lastidx += goods.shape[0]

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


    print("training on " + str(filters.shape[0]) + " / " + str(course.shape[0]) + " examples")

    return c5x, course, speed, curvature, accelerator, filters, hdf5_camera, goaldir


def data_iterator(validation=False, model="CNN"):
    first = True
    parser = argparse.ArgumentParser(description='Dataloader')
    parser.add_argument('--validation', dest='validation', action='store_true', default=False,
                        help='Serve validation dataset instead.')
    parser.add_argument('--nogood', dest='nogood', action='store_true', default=False, help='Ignore `goods` filters.')
    args, more = parser.parse_known_args()

    if model == "VA":
        str_data = "feat"
    elif model == "CNN":
        str_data = "cam"

    if validation:
        filenames = os.path.join(config.h5path, 'val_names.txt')
    else:
        filenames = os.path.join(config.h5path, 'train_names.txt')

    with open(filenames, 'r') as f:
        file_paths = ['%s%s/%s.h5' % (config.h5path, str_data, x.strip()) for x in f.readlines()]

    arrays = []

    # Datagen
    time_len = config.timelen
    batch_size = config.batch_size
    ignore_goods = args.nogood

    filter_names = sorted(file_paths)
    logger.info("Loading {} hdf5 buckets.".format(len(filter_names)))
    c5x, course, speed, curvature, accelerator, filters, hdf5_camera, goaldir = concatenate(model, filter_names, time_len=time_len)
    filters_set = set(filters)

    logger.info("camera files {}".format(len(c5x)))

    if model == "VA":
        X_batch = np.zeros((batch_size, time_len, 64, 12, 20), dtype='float32')
    elif model == "CNN":
        X_batch = np.zeros((batch_size, time_len, config.imgRow, 160, 3), dtype='uint8')

    course_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
    speed_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
    curvature_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
    accelerator_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
    goaldir_batch = np.zeros((batch_size, time_len, 1), dtype='float32')

    while True:
        try:
            t = time.time()

            count = 0
            start = time.time()
            while count < batch_size:
                if not ignore_goods:
                    i = np.random.choice(filters)

                    good = True
                    for j in (i - time_len + 1, i + 1):
                        if j not in filters_set:
                            good = False
                    if not good:
                        continue


                # GET X_BATCH

                for es, ee, x in c5x:
                    if i >= es and i < ee:
                        if x[i - es - time_len + 1:i - es + 1].shape != X_batch[count].shape:
                            break
                        X_batch[count] = x[i - es - time_len + 1:i - es + 1]  # !!!
                        break

                if x[i - es - time_len + 1:i - es + 1].shape == X_batch[count].shape:
                    course_batch[count] = np.copy(course[i - time_len + 1:i + 1])[:, None]
                    speed_batch[count] = np.copy(speed[i - time_len + 1:i + 1])[:, None]
                    curvature_batch[count] = np.copy(curvature[i - time_len + 1:i + 1])[:, None]
                    accelerator_batch[count] = np.copy(accelerator[i - time_len + 1:i + 1])[:, None]
                    goaldir_batch[count] = np.copy(goaldir[i - time_len + 1:i + 1])[:, None]

                    count += 1

            # sanity check
            if model == "VA":
                assert X_batch.shape == (batch_size, time_len, 64, 12, 20)
            elif model == "CNN":
                assert X_batch.shape == (batch_size, time_len, config.imgRow, 160, 3)

            logging.debug("load image : {}s".format(time.time() - t))
            print("%5.2f ms" % ((time.time() - start) * 1000.0))

            if first:
                print("X", X_batch.shape)
                print("angle", course_batch.shape)
                print("speed", speed_batch.shape)
                print("curvature", curvature_batch.shape)
                print("accelerator", accelerator_batch.shape)
                print("goaldir", goaldir_batch.shape)
                first = False

            array = (X_batch, course_batch, speed_batch, curvature_batch, accelerator_batch, goaldir_batch)

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



