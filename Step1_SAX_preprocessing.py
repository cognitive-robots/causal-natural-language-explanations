# |**********************************************************************;
# Project           : Why Do We Stop? Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

from scipy import interpolate
import json
import csv
from sys import platform
from tqdm import tqdm
import numpy as np
import os
import time
import glob
import cv2
import scipy.misc
import h5py
from random import shuffle
import skvideo.io
import skvideo.datasets
from scipy.ndimage import rotate
from src.utils import *

# Main function
# ----------------------------------------------------------------------


def main():
    if platform == 'linux':
        config = dict2(**{
            # (video url, start, end, action, justification)
            "annotations":  './SAX-Dataset/Explanations_new/', #BDD-X-Annotations_v1.csv
            "save_path":    './data/processed_SAX_cropped/',
            "data_path":    './SAX-Dataset/',
            "chunksize":    20})

    annot_directory = os.fsencode(config.annotations)

    data_train = {}
    data_train['annotations'] = []
    data_train['info'] = []
    data_train['videos'] = []

    data_val = {}
    data_val['annotations'] = []
    data_val['info'] = []
    data_val['videos'] = []

    data_test = {}
    data_test['annotations'] = []
    data_test['info'] = []
    data_test['videos'] = []

    # Parameters
    maxItems = 75  # Each csv file contains action / justification pairs of 210 at maximum, prev 15 with BDD

    # output path
    if not os.path.exists(config.save_path+"log/"):
        os.makedirs(config.save_path+"log/")
    if not os.path.exists(config.save_path+"cam/"):
        os.makedirs(config.save_path+"cam/")

    # Read information about video clips
    captionid = 0
    videoid = 0
    vidNames = []
    testNames = []
    trainNames = []
    valNames = []
    testCounter = 0
    vidNames_notUSED = []

    for video_file_csv in os.listdir(annot_directory):
        filename = os.fsdecode(video_file_csv)
        if filename.endswith(".csv"):
            with open(config.annotations+filename, encoding='utf-8-sig') as f_obj:
                action_examples = csv.DictReader(f_obj, delimiter=';')
                vidName = filename[:-8]
                video_file = '%sVideos/%s.mp4' % (config.data_path, vidName)
                if os.path.isfile(video_file) == False:
                    vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
                    continue

                # --------------------------------------------------
                # 1. Control signals
                # --------------------------------------------------
                str2find = '%sInfo_v2_processed/%s.csv' % (config.data_path, vidName)
                if os.path.isfile(str2find) == False:
                    vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
                    continue

                timestamp, longitude, course, latitude, speed, gps_x, gps_y, accelerator = [], [], [], [], [], [], [], []
                with open(str2find) as info_data:
                    trajectories = csv_dict_reader(info_data)
                    for trajectory in trajectories:
                        if len(trajectory['Longitude']) == 0:
                            continue
                        else:
                            longitude.append(float(trajectory['Longitude']))
                            course.append(float(trajectory['Heading (deg)']))
                            latitude.append(float(trajectory['Latitude']))
                        timestamp.append(int(trajectory['Timestamp']))
                        speed_ms = (1000 / 3600) * float(trajectory['Vehicle Speed (km/h)'])
                        speed.append(speed_ms)
                        accelerator.append(float(trajectory['A2 (m/s/s)']))


                        _x, _y, _ = lla2flat((float(trajectory['Latitude']), float(trajectory['Longitude']), 1000.0),
                                                 (latitude[0], longitude[0]), 0.0, -100.0)
                        gps_x.append(_x)
                        gps_y.append(_y)

                # Use interpolation to prevent variable periods
                if np.array(timestamp).shape[0] < 2:
                    print(
                        bcolors.FAIL + "Timestamp is not VALID: {}".format(str2find) + bcolors.ENDC)
                    continue

                # extract equally-spaced points
                points, dist_steps, cumulative_dist_along_path = get_equally_spaced_points(
                    gps_x, gps_y)

                # Generate target direction
                goalDirection_equal = get_goalDirection(dist_steps, points)
                goalDirection_interp = interpolate.interp1d(
                    dist_steps, goalDirection_equal)
                goalDirection = goalDirection_interp(cumulative_dist_along_path)

                # Generate curvatures / accelerator
                curvature_raw = compute_curvature(points[0], points[1])
                curvature_interp = interpolate.interp1d(dist_steps, curvature_raw)
                curvature = curvature_interp(cumulative_dist_along_path)

                # --------------------------------------------------
                # 2.Captions
                # --------------------------------------------------
                nEx = 0

                for item in action_examples:
                    if len(item["start_time"]) == 0:
                        continue
                    sTime = item["start_time"]
                    eTime = item["end_time"]
                    action = item["Action"]
                    justification = item["Justification"]

                    nEx += 1
                    captionid += 1

                    # Info
                    feed_dict = {}
                    if testCounter == 4:
                        data_val['info'].append(feed_dict)
                    elif testCounter == 2:
                        data_test['info'].append(feed_dict)
                    else:
                        data_train['info'].append(feed_dict)

                    # annotations
                    feed_dict = {'action': action,
                                 'justification': justification,
                                 'sTime': sTime,
                                 'eTime': eTime,
                                 'id': captionid,
                                 'vidName': vidName,
                                 'video_id': videoid,
                                 }
                    if testCounter == 4:
                        data_val['annotations'].append(feed_dict)
                    elif testCounter == 2:
                        data_test['annotations'].append(feed_dict)
                    else:
                        data_train['annotations'].append(feed_dict)

                    # Video
                    feed_dict = {'video_name': vidName,
                                     'height': 480,
                                     'width': 640,
                                     'video_id': videoid,
                                     }

                    if testCounter == 4:
                        data_val['videos'].append(feed_dict)
                    elif testCounter == 2:
                        data_test['videos'].append(feed_dict)
                    else:
                        data_train['videos'].append(feed_dict)

                print(bcolors.GREEN +
                      "Processed >> Annotations: {} sub-examples".format(nEx) + bcolors.ENDC)

                # --------------------------------------------------
                # 3. Read video clips
                # --------------------------------------------------
                # original image: 720x1280
                # str2read = '%svideos/%s.mov' % (config.data_path, vidName)
                str2read = '%sVideos/%s.mp4' % (config.data_path, vidName)
                frames = []
                cnt = 0
                scalefactor = 1

                if os.path.exists(str2read):
                    metadata = skvideo.io.ffprobe(str2read)

                    if ("side_data_list" in metadata["video"].keys()) == False:
                        rotation = 0
                    else:
                        rotation = float(
                            metadata["video"]["side_data_list"]["side_data"]["@rotation"])

                    cap = cv2.VideoCapture(str2read)
                    nFrames, img_width, img_height, fps = get_vid_info(cap)
                    print(bcolors.GREEN +
                          'ID: {}, #Frames: {}, nGPSrecords: /, Image: {}x{}, FPS: {}'.format(
                              vidName, nFrames, img_width, img_height, fps)
                          + bcolors.ENDC)

                    ###
                    with open(config.data_path + "Info_v2_processed/segments/" + vidName + ".csv", encoding='utf-8-sig') as f_seg:
                        list_of_segments = csv.DictReader(f_seg, delimiter=';')
                        gotImage = True
                        for segment in list_of_segments:
                            while gotImage and cap.get(cv2.CAP_PROP_POS_MSEC) < int(segment["start_time"])*1000:
                                gotImage, frame = cap.read()
                            while gotImage and cap.get(cv2.CAP_PROP_POS_MSEC) <= int(segment["end_time"])*1000:
                                gotImage, frame = cap.read()
                                cnt += 1
                                # print(cnt)
                                if gotImage:
                                    if cnt % 2 == 0:  # reduce to 8Hz

                                        if rotation > 0:
                                            frame = cv2.flip(frame, 0)
                                        elif rotation < 0:
                                            frame = cv2.flip(frame, 1)
                                        else:
                                            frame = frame.swapaxes(1, 0)

                                        # Crop out front part of ego vehicle
                                        frame = frame[0:640,0:405,:]


                                        frame = cv2.resize(
                                            frame, None, fx=(0.125/405*720) * scalefactor, fy=(0.125/640*1280) * scalefactor)

                                        try:
                                            assert frame.shape == (
                                            90 * scalefactor, 160 * scalefactor, 3)
                                        except AssertionError as e:
                                            frame = frame.swapaxes(1, 0)
                                            assert frame.shape == (90 * scalefactor, 160 * scalefactor, 3)

                                        if cnt % 100 == 0:
                                            cv2.imwrite('sample.png', frame)

                                        frame = cv2.cvtColor(
                                            frame, cv2.COLOR_BGR2RGB)  # 640x360x3

                                        frames.append(frame)
                            continue
                    ###

                    cap.release()
                else:
                    print(bcolors.FAIL +
                          'ERROR: Unable to open video {}'.format(str2read)
                          + bcolors.ENDC)
                    break

                frames = np.array(frames).astype(int)

                # --------------------------------------------------
                # 4. Saving
                # --------------------------------------------------
                vidNames.append(str(videoid) + "_" + str(vidName))
                if testCounter == 4:
                    valNames.append(str(videoid) + "_" + str(vidName))
                    testCounter = testCounter + 1
                elif testCounter == 2:
                    testNames.append(str(videoid) + "_" + str(vidName))
                    testCounter = testCounter + 1
                    #testCounter = 0
                else:
                    trainNames.append(str(videoid) + "_" + str(vidName))
                    testCounter = testCounter + 1
                    if testCounter == 10:
                        testCounter = 0

                if (os.path.isfile(config.save_path + "cam/" + str(videoid) + "_" + str(vidName) + ".h5")) == False:
                    cam = h5py.File(config.save_path + "cam/" +
                                    str(videoid) + "_" + str(vidName) + ".h5", "w")
                    dset = cam.create_dataset("/X", data=frames, chunks=(
                        config.chunksize, 90 * scalefactor, 160 * scalefactor, 3), dtype='uint8')
                else:
                    print(bcolors.GREEN +
                          'File already generated (cam): {}'.format(
                              str(videoid) + "_" + str(vidName))
                          + bcolors.ENDC)

                if (os.path.isfile(config.save_path + "log/" + str(videoid) + "_" + str(vidName) + ".h5")) == False:
                    log = h5py.File(config.save_path + "log/" +
                                    str(videoid) + "_" + str(vidName) + ".h5", "w")

                    dset = log.create_dataset("/timestamp", data=timestamp)
                    dset = log.create_dataset("/longitude", data=longitude)
                    dset = log.create_dataset("/course", data=course)
                    dset = log.create_dataset("/latitude", data=latitude)
                    dset = log.create_dataset("/speed", data=speed)
                    # dset = log.create_dataset("/fps",       data=fps)
                    dset = log.create_dataset(
                        "/curvature", data=curvature, dtype='float')
                    dset = log.create_dataset(
                        "/accelerator", data=accelerator, dtype='float')
                    dset = log.create_dataset(
                        "/goaldir", data=goalDirection, dtype='float')

                else:
                    print(bcolors.GREEN +
                          'File already generated (log): {}'.format(
                              str(videoid) + "_" + str(vidName))
                          + bcolors.ENDC)

                videoid += 1



        else:
            continue

    with open(config.save_path+'captions_BDDX_train.json', 'w') as outfile:
        json.dump(data_train, outfile)
    with open(config.save_path+'captions_BDDX_val.json', 'w') as outfile:
        json.dump(data_val, outfile)
    with open(config.save_path+'captions_BDDX_test.json', 'w') as outfile:
        json.dump(data_test, outfile)

    np.savetxt(config.save_path + "train_names.txt", trainNames, fmt="%s")
    np.savetxt(config.save_path + "test_names.txt", testNames, fmt="%s")
    np.savetxt(config.save_path + "val_names.txt", valNames, fmt="%s")

    np.save(config.save_path + 'vidNames_notUSED.npy', vidNames_notUSED)
    print('Success!')


if __name__ == "__main__":
    main()
