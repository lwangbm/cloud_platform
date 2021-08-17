from __future__ import absolute_import, division, print_function

import pickle

import nltk
import numpy as np
import tensorflow as tf

import feder.mnist_inference as mnist
from feder.DiffPrivate_FedLearning_word_slave import \
    run_differentially_private_federated_averaging
from feder.MNIST_reader import Data, Data_word, Data_word_2

nltk.download("punkt")


class ModelMaintain(object):
    def __init__(self):
        pass

    def startNewRoundTraining(self, New_weights, Sanitized_Updates):

        fil = open(self.save_dir + "/model.pkl", "rb")
        model = pickle.load(fil)
        fil.close()
        real_step = model["global_step_placeholder:0"]
        print(self.save_dir)

        New_model = dict(zip(self.keys, New_weights))
        New_model["global_step_placeholder:0"] = real_step
        print(self.save_dir)

        filehandler = open(self.save_dir + "/model.pkl", "wb")
        pickle.dump(New_model, filehandler)
        filehandler.close()
        self.cout_true = 0
        with tf.Graph().as_default():

            for index in range(10):

                model, keys, Weight_for_update, weights_accountant = self.basic_train(
                    SubClient_index=index, ifSaveModel=False
                )

                # check Sig
                if self.methodSig == 1:
                    Significance = weights_accountant.checkSignificance(
                        New_weight=Weight_for_update,
                        prev_Sanitized_Updates=Sanitized_Updates,
                        threshold=self.threshold,
                    )

                if self.methodSig == 3:
                    Significance = weights_accountant.checkSignificance_Gia(
                        New_weight=Weight_for_update,
                        prev_Sanitized_Updates=Sanitized_Updates,
                        threshold=self.threshold,
                    )

                if Significance:
                    if self.cout_true == 0:
                        self.WeightList = [
                            np.expand_dims(Weight_for_update[i], -1) for i in range(4)
                        ]
                    else:
                        self.WeightList = [
                            np.concatenate(
                                (
                                    self.WeightList[i],
                                    np.expand_dims(Weight_for_update[i], -1),
                                ),
                                -1,
                            )
                            for i in range(4)
                        ]

                    self.cout_true += 1

            self.oldWeight = [np.mean(self.WeightList[i], -1) for i in range(4)]

            return self.cout_true

        #     train_op, eval_correct, loss, data_placeholder, labels_placeholder = mnist.word_fully_connected_model(self.b,
        #                                                                                                           self.hidden,
        #                                                                                                           self.window,
        #                                                                                                           self.num_classes,
        #                                                                                                           self.num_input)
        #
        #     model, Weight_for_update, keys,weights_accountant,self.save_dir = \
        #         run_differentially_private_federated_averaging(self.num_input, self.num_classes, self.window, loss,
        #                                                        train_op,
        #                                                        eval_correct, self.DATA, data_placeholder,
        #                                                        labels_placeholder, b=self.b, e=self.e, m=self.m,
        #                                                        sigma=self.sigma, eps=self.eps,
        #                                                        save_dir=self.save_dir, log_dir=self.log_dir,
        #                                                        worker_id=self.worker_id,use_signi=self.use_signi, threshold=self.threshold, methodSig=self.methodSig)
        #
        #     self.oldWeight = Weight_for_update
        #     self.model = model
        #
        #
        #     # check Sig
        #     if self.methodSig == 1:
        #         Significance = weights_accountant.checkSignificance( New_weight=self.oldWeight,prev_Sanitized_Updates=Sanitized_Updates, threshold=self.threshold)
        #
        #     if self.methodSig == 3:
        #         Significance = weights_accountant.checkSignificance_Gia(New_weight=self.oldWeight, prev_Sanitized_Updates=Sanitized_Updates, threshold=self.threshold)
        #
        #
        # return Significance

    def basic_train(self, SubClient_index, ifSaveModel):

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
                model,
                Weight_for_update,
                keys,
                weights_accountant,
                self.save_dir,
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
                log_dir=self.log_dir,
                worker_id=self.worker_id,
                use_signi=self.use_signi,
                threshold=self.threshold,
                methodSig=self.methodSig,
                SubClient_index=SubClient_index,
                ifSaveModel=ifSaveModel,
            )
        return model, keys, Weight_for_update, weights_accountant

    def startSlave(self, worker_id):

        self.worker_id = worker_id

        model, keys, Weight_for_update, weights_accountant = self.basic_train(
            SubClient_index=0, ifSaveModel=True
        )

        self.model = model
        self.keys = keys
        self.WeightList = [np.expand_dims(Weight_for_update[i], -1) for i in range(4)]

        for index in range(9):
            model, keys, Weight_for_update, weights_accountant = self.basic_train(
                SubClient_index=index + 1, ifSaveModel=False
            )

            self.WeightList = [
                np.concatenate(
                    (self.WeightList[i], np.expand_dims(Weight_for_update[i], -1)), -1
                )
                for i in range(4)
            ]

        self.oldWeight = [np.mean(self.WeightList[i], -1) for i in range(4)]
