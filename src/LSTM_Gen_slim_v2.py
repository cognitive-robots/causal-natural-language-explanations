# |**********************************************************************;
# Project           : Why Do We Stop? Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import tensorflow as tf
import numpy as np
from src.config_VA import *
import sys
subs = int(config.subsample)


class LSTM_Gen(object):
    def __init__(self, alpha_c=0.0, dropout=True):

        self.alpha_c = tf.to_float(alpha_c)
        self.USE_DROPOUT = dropout
        self.SENSORFUSE = True
        self.timelen = config.timelen - 3

        dim_hidden = config.dict_size

        # Parameters
        self.T = self.timelen - 1
        self.H = config.dim_hidden_Gen
        self.L = config.ctx_shape[0]/(subs**2)
        self.D = config.ctx_shape[1]/subs
        self.M = config.dim_hidden_Gen
        self.V = 1

        # Place holders

        self.context = tf.placeholder(tf.float32, shape=[None, 10, 64/(subs)])
        self.features = tf.placeholder(tf.float32, shape=[None, 10, 64/subs, 12/subs, 20/subs])
        self.pred_acc = tf.placeholder(tf.float32, shape=[None, 10])
        self.pred_course = tf.placeholder(tf.float32, shape=[None, 10])
        self.acc_gt = tf.placeholder(tf.float32, shape=[None, 10])
        self.course_gt = tf.placeholder(tf.float32, shape=[None, 10])
        self.caption = tf.placeholder(tf.int32, shape=[None, 1, 22])
        self.caption_onehot = tf.placeholder(tf.int32, shape=[None, 22, config.dict_size])
        self.sequence_id = tf.placeholder(tf.int32, shape=[None, 1])


        # Initializer
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):

            features_mean = tf.reduce_mean(features, 2)
            features_mean = features_mean[:, 0]

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])

            return features_proj

    def _project_contexts(self, contexts):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [(64/(subs))+2, 64/(subs)+2], initializer=self.weight_initializer)

            features_flat = tf.reshape(contexts, [-1, int(64/(subs)+2)])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, 10, int(64/(subs)+2)])

            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])
            alpha = tf.nn.softmax(out_att)
            alpha_logp = tf.nn.log_softmax(out_att)
            context = tf.reshape(features * tf.expand_dims(alpha, 2), [-1, self.L * self.D])

            return context, alpha, alpha_logp

    def _temp_attention_layer(self, contexts, contexts_proj, h):
        with tf.variable_scope('attention_layer', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', [self.H, 64/(subs)+2], initializer=self.weight_initializer)
            b = tf.get_variable('b', [64/(subs)+2], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [64/(subs)+2, 22], initializer=self.weight_initializer)

            h_att = tf.nn.relu(contexts_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, int(64/(subs)+2)]), w_att), [-1, 22, 10])
            beta = tf.nn.softmax(out_att)
            beta_logp = tf.nn.log_softmax(out_att)


            z = tf.reshape(tf.matmul(beta, contexts), [-1, 22*int(64/(subs)+2)])


            return z, beta, beta_logp

    def _decode_lstm(self, h, context, scope='logits'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            w_h = tf.get_variable('w_h', [self.H, 22*self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [22*self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [22*self.M, 22*config.dict_size], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [22*config.dict_size], initializer=self.const_initializer)

            h_logits = tf.matmul(h, w_h) + b_h

            w_ctx2out = tf.get_variable(
                name='w_ctx2out',
                shape=[22*int(64/(subs)+2), 22*self.M],
                initializer=self.weight_initializer
            )
            h_logits += tf.matmul(context, w_ctx2out)


            if self.USE_DROPOUT:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out

            out_logits = tf.reshape(out_logits, [-1, 22, config.dict_size])

            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def build_model(self):
        context = self.context
        feats = self.features
        acc = self.pred_acc
        course = self.pred_course


        caption = self.caption
        caption_onehot = self.caption_onehot
        #batch_size = tf.shape(context)[0]
        batch_size = config.batch_size_gen


        # apply batch norm to feature vectors
        context = self._batch_norm(context, mode='train', name='conv_contexts')

        feats = tf.reshape(feats, [batch_size, 10, int(64/subs), int(12 * 20 / (subs**2))])
        feats = tf.transpose(feats, [0, 1, 3, 2])
        feats = self._batch_norm(feats, mode='train', name='conv_features')

        acc = tf.expand_dims(acc, 2)
        course = tf.expand_dims(course, 2)


        # ! CONCATENATE CONTEXT + ACC / COURSE PREDs
        context_acc_course = tf.concat([context, acc, course], 2)

        # Init LSTM with feature vectors
        #c, h = self._get_initial_lstm(features=feats)
        c, h = self._get_initial_lstm(features=feats)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        if self.USE_DROPOUT:
            h = tf.nn.dropout(h, 0.5)

        # Feature projection
        contexts_proj = self._project_contexts(contexts=context_acc_course)

        # losses
        loss = 0.0
        alpha_reg = 0.0

        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):  # (t != 0)):
            # _, (c_expanded[t], h_expanded[t]) = lstm_cell(inputs=z, state=[c_expanded[t], h_expanded[t]])
            #tf.reshape(context_acc_course, [-1, 10 * int(64 / (subs) + 2)])
            _, (c, h) = lstm_cell(inputs=tf.reshape(context_acc_course, [-1, 10 * int(64 / (subs) + 2)]), state=[c, h])

        if self.USE_DROPOUT:
            h = tf.nn.dropout(h, 0.5)


        z, beta, beta_logp = self._temp_attention_layer(context_acc_course, contexts_proj, h)#contexts_proj
                                                            #reuse=(t != 0))

        logits_caption = self._decode_lstm(h, z, scope='logits_acc')

        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=caption_onehot, logits=logits_caption))

        # loss += alpha_reg
        first_summary_train = tf.summary.scalar(name='Train loss', tensor=loss)
        first_summary_val = tf.summary.scalar(name='Validation loss', tensor=loss)

        return loss, first_summary_train, first_summary_val  # , alpha_reg

    def inference(self):
        context = self.context
        feats = self.features
        acc = self.pred_acc
        course = self.pred_course
        batch_size = config.batch_size_gen

        context = self._batch_norm(context, mode='train', name='conv_contexts')

        feats = tf.reshape(feats, [batch_size, 10, int(64 / subs), int(12 * 20 / (subs ** 2))])
        feats = tf.transpose(feats, [0, 1, 3, 2])
        feats = self._batch_norm(feats, mode='train', name='conv_features')

        acc = tf.expand_dims(acc, 2)
        course = tf.expand_dims(course, 2)

        context_acc_course = tf.concat([context, acc, course], 2)

        # Initialize LSTM
        c, h = self._get_initial_lstm(features=feats)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        # Feature projection
        contexts_proj = self._project_contexts(contexts=context_acc_course)


        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):  # (t != 0)):
            _, (c, h) = lstm_cell(inputs=tf.reshape(context_acc_course, [-1, 10 * int(64 / (subs) + 2)]), state=[c, h])

        z, beta, beta_logp = self._temp_attention_layer(context_acc_course, contexts_proj, h)


        logits_caption = self._decode_lstm(h, z, scope='logits_acc')

        logits_softmax = tf.nn.softmax(logits_caption)


        return logits_softmax
