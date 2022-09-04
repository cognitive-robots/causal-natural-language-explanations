#!/usr/bin/env python
"""
Helper funtions for steering angle prediction model
"""

import  tensorflow   as     tf
from   	src.config_VA   import *


class PreProcessor_Gen():
    def __init__(self, timelen=config.timelen, phase='train'):
        # Build the Tensorflow graph
        with tf.variable_scope("pre-processor"):
            self.inputImg    = tf.placeholder(shape=[None, timelen, 64, 12, 20], dtype=tf.float32)
            self.curvature   = tf.placeholder(shape=[None, timelen, 1], dtype=tf.float32)
            self.accelerator = tf.placeholder(shape=[None, timelen, 1], dtype=tf.float32)
            self.speed       = tf.placeholder(shape=[None, timelen, 1], dtype=tf.float32)
            self.course      = tf.placeholder(shape=[None, timelen, 1], dtype=tf.float32)
            self.goaldir     = tf.placeholder(shape=[None, timelen, 1], dtype=tf.float32)

            # we use 4 consecutive frames, thus we need remove first and last few frames..
            inputImg    = tf.transpose( self.inputImg,      perm=[1,0,2,3,4] )
            curvature   = tf.transpose( self.curvature,     perm=[1,0,2] )
            accelerator = tf.transpose( self.accelerator,   perm=[1,0,2] )
            speed       = tf.transpose( self.speed,         perm=[1,0,2] )
            course      = tf.transpose( self.course,        perm=[1,0,2] )
            goaldir     = tf.transpose( self.goaldir,       perm=[1,0,2] )

            if phase == 'train':
                # ego-motions
                START_FRAME = 3
                gather_indices_img  = tf.range(timelen-START_FRAME)
                gather_indices_info = tf.range(START_FRAME, timelen) 

                inputImg    = tf.transpose(tf.gather(inputImg, gather_indices_img),     perm=[1,0,2,3,4])
                curvature   = tf.transpose(tf.gather(curvature, gather_indices_info),   perm=[1,0,2])
                accelerator = tf.transpose(tf.gather(accelerator, gather_indices_info), perm=[1,0,2])
                speed       = tf.transpose(tf.gather(speed, gather_indices_info),       perm=[1,0,2])
                course      = tf.transpose(tf.gather(course, gather_indices_info),      perm=[1,0,2])
                goaldir     = tf.transpose(tf.gather(goaldir, gather_indices_info),     perm=[1,0,2])
            else:
                START_FRAME = 0

            # feats
            dst = tf.reshape(   inputImg, [config.batch_size_gen, timelen-START_FRAME, 64, 12*20] )
            dst = tf.reshape(   dst, [-1,64,12*20] )
            dst = tf.transpose( dst, [0,2,1] )
            self.outImg = dst

            # Scaling & Reshape
            SCALE_ACC = 100
            SCALE_CUR = 10

            self.curvature_    = tf.reshape(curvature,   [-1, 1]) * SCALE_CUR
            self.accelerator_  = tf.reshape(accelerator, [-1, 1]) * SCALE_ACC
            self.speed_        = tf.reshape(speed,       [-1, 1])
            self.course_       = tf.reshape(course,      [-1, 1]) * SCALE_CUR
            self.goaldir_      = tf.reshape(goaldir,     [-1, 1])

            # added discrete actions: go straight / stop or slow
            self.discrete_action = tf.to_float(tf.greater(self.accelerator_, -1.0*SCALE_ACC/10.0)) * tf.to_float(tf.greater(self.speed_, 2.0))
            self.discrete_action = tf.squeeze(tf.one_hot(tf.to_int64(self.discrete_action), depth=2))
            self.curvature_mask  = tf.to_float(tf.less_equal(self.curvature_, 0.1*100000)) * tf.to_float(tf.greater(self.speed_, 5))

    # process pre-porcessing
    def process(self, sess, inImg, course, speed, curvature, accelerator, goaldir):

        feed = {self.inputImg:      inImg, 
                self.speed:         speed, 
                self.course:        course, 
                self.curvature:     curvature, 
                self.accelerator:   accelerator,
                self.goaldir:       goaldir }

        return sess.run([self.outImg, self.curvature_, self.accelerator_, self.speed_, self.course_, self.curvature_mask, self.goaldir_, self.discrete_action], feed)

