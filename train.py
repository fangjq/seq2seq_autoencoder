# -*- coding: utf-8 -*- #
# Author: Jiaquan Fang

import tensorflow as tf
import os
import seq2seq_network
import data_utils
from tensorflow.python.platform import gfile

tf.flags.DEFINE_integer("embedding_dim", 128,
                        "Dimensionality of word embedding (default: 128)")
tf.flags.DEFINE_integer("vocab_size", 4000,
                        "Vocabulary size (default 4000)")
tf.flags.DEFINE_integer("max_seq_len", 50,
                        "Maximum sequence length")
tf.flags.DEFINE_integer("num_units", 128,
                        "size of hidden states")
tf.flags.DEFINE_integer("num_layers", 1,
                        "size of hidden layers")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
                      "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("validation_ratio", 0.01,
                      "ratio of validation data (default: 0.01)")
tf.flags.DEFINE_float("learning_rate", 0.01,
                      "learning rate")
tf.flags.DEFINE_float("max_gradient_norm", 5.0,
                      "gradient norm (default: 5.0) ")
tf.flags.DEFINE_string("data_path", "data/corpus_sep.txt",
                       "Data path (default: data/corpus_sep.txt)")
tf.flags.DEFINE_string("model_path", "./models",
                       "Model path (default: ./models)")
tf.flags.DEFINE_string("graph_path", "./graphs",
                       "The directory to write model summaries to.")
tf.flags.DEFINE_integer(
        "hidden_units", 50,
        "Number of hidden units in softmax regression layer (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300,
                        "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer(
        "evaluation_step", 50,
        "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_step", 100,
                        "Save model after this many steps (default: 1000)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if __name__ == "__main__":
    ids_path, vocab_path = data_utils.prepare_data(FLAGS.data_path,
                                                   FLAGS.vocab_size)

    dataset = []
    with gfile.GFile(ids_path, mode="r") as ids_file:
        ids = ids_file.readline()
        while ids:
            ids = map(int, ids.split())
            dataset.append(ids)
            ids = ids_file.readline()

    dataset = [one for one in dataset if len(one) <= FLAGS.max_seq_len]
    train_val_split_point = len(dataset) - int(len(dataset) *
                                               FLAGS.validation_ratio)
    train_set = dataset[0:train_val_split_point]
    validation_set = dataset[train_val_split_point:]

    model = seq2seq_network.Seq2SeqAutoEncoder(FLAGS.vocab_size,
                                               FLAGS.embedding_dim,
                                               FLAGS.num_units,
                                               FLAGS.num_layers,
                                               FLAGS.max_seq_len,
                                               FLAGS.max_gradient_norm,
                                               FLAGS.learning_rate)

    if not os.path.exists(FLAGS.model_path):
        os.mkdir(FLAGS.model_path)

    if not os.path.exists(FLAGS.graph_path):
        os.mkdir(FLAGS.graph_path)

    model.fit(train_set, validation_set, FLAGS.batch_size,
              FLAGS.checkpoint_step, FLAGS.evaluation_step,
              FLAGS.model_path, FLAGS.graph_path)
