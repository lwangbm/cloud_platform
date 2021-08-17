from __future__ import absolute_import, division, print_function

import sys
import time

sys.path.append("/Users/ourokutaira/Desktop/edge/implementation/zebra/")

import nltk
import tensorflow as tf

import feder.mnist_inference as mnist
from feder.DiffPrivate_FedLearning_word_master import \
    run_differentially_private_federated_averaging
from feder.MNIST_reader import Data_word_2
from utils import logging

nltk.download("punkt")


class Scheduler(object):
    def __init__(
        self,
        N,
        b,
        e,
        m,
        sigma,
        eps,
        save_dir,
        log_dir,
        broker_factory,
        use_signi,
        threshold,
        methodSig,
    ):
        self.N = N
        self.b = b
        self.e = e
        self.m = m
        self.sigma = sigma
        self.eps = eps
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.hidden1 = 600
        self.hidden2 = 100
        self.hidden = 512
        self.window = 10
        self.num_input = 1
        self.broker_factory = broker_factory
        self.countWorkerUpdate = 0
        self.DATA = Data_word_2(save_dir, N, self.window, b)
        self.num_classes = len(self.DATA.vocabulary_dictionary)
        self.use_signi = use_signi
        self.threshold = threshold
        self.methodSig = methodSig
        self.my_round_master = 0

        with tf.Graph().as_default():
            # Building the model that we would like to train in differentially private federated fashion.
            # We will need the tensorflow training operation for that model, its loss and an evaluation method:

            (
                train_op,
                eval_correct,
                loss,
                data_placeholder,
                labels_placeholder,
            ) = mnist.word_fully_connected_model(
                self.b, self.hidden, self.window, self.num_classes, self.num_input
            )
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            self.oldWeight = [sess.run(tf.trainable_variables()[i]) for i in range(4)]

    def SartNewRound(self, Weights, workerCount_true):
        with tf.Graph().as_default():

            (
                train_op,
                eval_correct,
                loss,
                data_placeholder,
                labels_placeholder,
            ) = mnist.word_fully_connected_model(
                self.b, self.hidden, self.window, self.num_classes, self.num_input
            )

            (
                Weight_for_update,
                accuracy,
                prev_Sanitized_Updates,
            ) = run_differentially_private_federated_averaging(
                self.num_input,
                self.num_classes,
                self.window,
                loss,
                train_op,
                eval_correct,
                self.DATA,
                data_placeholder,
                labels_placeholder,
                b=self.b,
                e=self.e,
                m=self.m,
                sigma=self.sigma,
                eps=self.eps,
                save_dir=self.save_dir,
                log_dir=self.log_dir,
                use_signi=self.use_signi,
                threshold=self.threshold,
                methodSig=self.methodSig,
                Weights=Weights,
                my_round_master=self.my_round_master,
                workerCount_true=workerCount_true,
            )
            print(" - Updating Rounds: %s" % self.my_round_master)
            print(" - The Accuracy on the validation set is: %s" % accuracy)
            print(" - #of clients updating: %s" % workerCount_true)
            print(
                "--------------------------------------------------------------------------------------"
            )
            print(
                "--------------------------------------------------------------------------------------"
            )

            self.my_round_master += 1
            self.Weight_for_update = Weight_for_update
            self.prev_Sanitized_Updates = prev_Sanitized_Updates

    def update_rates(self):
        logging.debug("begining update rates")
        start_clock = time.time()
        if self.sharing_algo in ("SEBF", "HUG", "DRF", "SUPERCOFLOW"):
            # update rates first and then launch the new job
            self.update_rates_kiwi()
        elif self.sharing_algo == "PFF":
            self.update_rates_pff()
        elif self.sharing_algo == "FNDRF":
            self.kiwi_FNDRF()
        elif self.sharing_algo == "PLF":
            self.kiwi_PLF()
        elif self.sharing_algo == "AALO":
            self.kiwi_AALO()
        else:
            logging.error("unsupported scheme.")
            import sys

            sys.exit(0)

        logging.debug("done updating, cost time: {}".format(time.time() - start_clock))

        self.record_allocation()  # record progress for each job at each step(event)
