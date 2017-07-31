# -*- coding: utf-8 -*- #
# Author: Jiaquan Fang

import tensorflow as tf
import os
import time
import numpy as np
import data_utils
from tensorflow.contrib import rnn


class Seq2SeqAutoEncoder(object):
    def __init__(self, vocab_size, embedding_size, num_units,
                 num_layers, max_seq_len, max_gradient_norm,
                 learning_rate):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate

        self.global_step = tf.Variable(0, trainable=False)
        self.encoder_inputs = tf.placeholder(tf.int32,
                                             [None, self.max_seq_len])
        self.encoder_lengths = tf.placeholder(tf.int32, [None])
        self.decoder_inputs = tf.placeholder(tf.int32,
                                             [None, self.max_seq_len + 2])
        self.decoder_weights = tf.placeholder(tf.float32,
                                              [None, self.max_seq_len + 2])
        self.feed_previous = tf.placeholder(tf.bool)

        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            embedding = tf.get_variable("embedding",
                                        [self.vocab_size, self.embedding_size],
                                        tf.float32,
                                        tf.random_uniform_initializer(-1.0,
                                                                      1.0))
            embedded_encoder_inputs = tf.unstack(tf.transpose(
                    tf.nn.embedding_lookup(embedding, self.encoder_inputs),
                    [1, 0, 2]))
            embedded_decoder_inputs = tf.unstack(tf.transpose(
                    tf.nn.embedding_lookup(embedding, self.decoder_inputs),
                    [1, 0, 2]))

        cell = tf.contrib.rnn.BasicLSTMCell(self.num_units)

        with tf.variable_scope("encoder"):
            encoder_outputs, self.encoder_state = rnn.static_rnn(
                                        cell,
                                        embedded_encoder_inputs,
                                        sequence_length=self.encoder_lengths,
                                        dtype=tf.float32)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", [self.num_units, self.vocab_size])
            b = tf.get_variable("b", [self.vocab_size])
            output_projection = (W, b)

        def loop_function(prev, _):
            prev = tf.nn.xw_plus_b(prev, output_projection[0],
                                   output_projection[1])
            prev_symbol = tf.argmax(prev, 1)
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            return emb_prev

        with tf.variable_scope("decoder"):
            (decoder_outputs1,
             decoder_state1) = tf.contrib.legacy_seq2seq.rnn_decoder(
                                embedded_decoder_inputs, self.encoder_state,
                                cell, loop_function=None)

            tf.get_variable_scope().reuse_variables()

            (decoder_outputs2,
             decoder_state2) = tf.contrib.legacy_seq2seq.rnn_decoder(
                                embedded_decoder_inputs, self.encoder_state,
                                cell, loop_function=loop_function)

            self.decoder_outputs = tf.where(self.feed_previous,
                                            decoder_outputs2,
                                            decoder_outputs1)

        logits = [tf.nn.xw_plus_b(one, output_projection[0],
                                  output_projection[1])
                  for one in tf.unstack(self.decoder_outputs)]

        self.decoder_symbols = [tf.argmax(one, 1) for one in logits]

        targets = tf.unstack(tf.transpose(self.decoder_inputs, [1, 0]))

        weights = tf.unstack(tf.transpose(self.decoder_weights, [1, 0]))

        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(logits[:-1],
                                                            targets[1:],
                                                            weights[:-1],
                                                            name="loss")

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         self.max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.update = optimizer.apply_gradients(zip(clipped_gradients, params),
                                                global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def initialize(self, model_dir, session=None):
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        if latest_ckpt:
            self.saver.restore(session, latest_ckpt)
        else:
            print "Creating model with fresh parameters."
            session.run(tf.global_variables_initializer())

    def get_batch(self, dataset, batch_size, random=True):
        if random:
            seqs = np.random.choice(dataset, size=batch_size)
        else:
            seqs = dataset[0:batch_size]
        encoder_inputs = np.zeros((batch_size, self.max_seq_len), dtype=int)
        decoder_inputs = np.zeros((batch_size, self.max_seq_len + 2), dtype=int)
        encoder_lengths = np.zeros(batch_size)
        decoder_weights = np.zeros((batch_size, self.max_seq_len + 2),
                                   dtype=float)
        for i, seq in enumerate(seqs):
            encoder_inputs[i] = np.array(list(reversed(seq)) +
                                         [data_utils.PAD_ID] *
                                         (self.max_seq_len - len(seq)))

            decoder_inputs[i] = np.array([data_utils.GO_ID] +
                                         seq +
                                         [data_utils.EOS_ID] +
                                         [data_utils.PAD_ID] *
                                         (self.max_seq_len - len(seq)))
            encoder_lengths[i] = len(seq)
            decoder_weights[i, 0:(len(seq) + 1)] = 1.0

        return (encoder_inputs,
                decoder_inputs,
                encoder_lengths,
                decoder_weights)

    def step(self, encoder_inputs, decoder_inputs=None,
             encoder_lengths=None, decoder_weights=None,
             is_train=False, sess=None):
        feed_dict = {}
        feed_dict[self.encoder_inputs] = encoder_inputs
        feed_dict[self.decoder_inputs] = decoder_inputs
        feed_dict[self.encoder_lengths] = encoder_lengths
        feed_dict[self.decoder_weights] = decoder_weights

        if is_train:
            feed_dict[self.feed_previous] = False
            encoder_state, decoder_symbols, summary, loss, _ = sess.run([
                self.encoder_state, self.decoder_symbols, self.summary_op,
                self.loss, self.update], feed_dict=feed_dict)
        else:
            feed_dict[self.feed_previous] = True
            encoder_state, decoder_symbols, summary, loss = sess.run([
                self.encoder_state, self.decoder_symbols,
                self.summary_op, self.loss],
                feed_dict=feed_dict)

        return (encoder_state, decoder_symbols, summary, loss)

    def fit(self, train_set, validation_set, batch_size,
            steps_per_checkpoint, steps_per_evaluation,
            model_dir, graph_dir):
        with tf.Session() as sess:
            self.initialize(model_dir, sess)

            train_writer = tf.summary.FileWriter(graph_dir + "/train",
                                                 sess.graph)
            dev_writer = tf.summary.FileWriter(graph_dir + "/test",
                                               sess.graph)

            iteration = 0
            while True:
                iteration += 1
                start = time.time()

                (encoder_inputs, decoder_inputs,
                 encoder_lengths, decoder_weights) = self.get_batch(train_set,
                                                                    batch_size)
                _, _, summary, loss = self.step(encoder_inputs, decoder_inputs,
                                                encoder_lengths,
                                                decoder_weights,
                                                True, sess)

                finish = time.time()

                global_step = self.global_step.eval()
                print "global_step: {}, loss: {}, time: {}".format(
                    global_step, loss, finish - start)

                if global_step % steps_per_evaluation == 0:
                    validation_loss = 0.0
                    validation_batch_size = 10240
                    for i in xrange(int(np.ceil(
                                        1.0*len(validation_set) /
                                        validation_batch_size))):
                        start = i * validation_batch_size
                        end = min((i + 1) * validation_batch_size,
                                  len(validation_set))
                        (encoder_inputs, decoder_inputs,
                         encoder_lengths, decoder_weights) = self.get_batch(
                             validation_set[start:end], end - start, False)
                        _, _, summary, step_loss = self.step(encoder_inputs,
                                                             decoder_inputs,
                                                             encoder_lengths,
                                                             decoder_weights,
                                                             False, sess)
                        validation_loss += step_loss * (end - start) \
                            / len(validation_set)
                    print "validation-loss {}".format(validation_loss)

                    dev_writer.add_summary(summary, global_step)

                if global_step % steps_per_checkpoint == 0:
                    self.saver.save(sess, os.path.join(model_dir,
                                                       "checkpoint"),
                                    global_step=global_step)

                train_writer.add_summary(summary, global_step)

            train_writer.close()
