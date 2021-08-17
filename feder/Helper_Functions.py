from __future__ import absolute_import, division, print_function

import copy
import csv
import math
import os
import os.path
import pickle

import numpy as np
import tensorflow as tf

import api.communication.api_pb2 as api_pb2
from feder.accountant import GaussianMomentsAccountant


class PrivAgent:
    def __init__(self, N, Name):
        self.N = N
        self.Name = Name
        self.m = [self.N] * 5000
        self.Sigma = [1] * 10000  # 24
        self.bound = 5000  # 0.001

        # if N == 100:
        #     self.m = [100]*5000
        #     self.Sigma = [1]*10000#24
        #     self.bound = 5000#0.001
        # if N == 1000:
        #     self.m = [300]*1000
        #     self.Sigma = [1]*10000#24
        #     self.bound = 1000#0.00001
        # if N == 10000:
        #     self.m = [300]*1000
        #     self.Sigma = [1]*10000#24
        #     self.bound = 1000#0.000001
        # if(N != 100 and N != 1000 and N != 10000 ):
        #     print('!!!!!!! YOU CAN ONLY USE THE PRIVACY AGENT FOR N = 100, 1000 or 10000 !!!!!!!')

    def get_m(self, r):
        return self.m[r]

    def get_Sigma(self, r):
        return self.Sigma[r]

    def get_bound(self):
        return self.bound


def putIntoProto_slave(New_weights, id, ifSig):
    status = api_pb2.Status()
    status.event_type = status.WORKER_UPDATE

    worker_update = status.worker_update

    # worker_update.my_round=real_round
    worker_update.my_id = id
    worker_update.my_ifSig = ifSig

    if ifSig > 0:

        weightmatrix = worker_update.my_weight

        weightmatrix_0 = weightmatrix.weight_0
        value = weightmatrix_0.value
        Row, Co = New_weights[0].shape
        for i in range(Row):
            for j in range(int(Co)):
                value.append(New_weights[0][i, j])

        Row_t = New_weights[1].shape
        Row = Row_t[0]
        weightmatrix_1 = weightmatrix.weight_1
        value = weightmatrix_1.value
        for i in range(Row):
            value.append(
                New_weights[1][
                    i,
                ]
            )

        Row_t = New_weights[3].shape
        Row = Row_t[0]
        weightmatrix_3 = weightmatrix.weight_3
        value = weightmatrix_3.value
        for i in range(Row):
            value.append(
                New_weights[3][
                    i,
                ]
            )

        weightmatrix_2 = weightmatrix.weight_2
        value = weightmatrix_2.value
        Row, Co = New_weights[2].shape
        for i in range(Row):
            for j in range(Co):
                value.append(New_weights[2][i, j])

    return status


def putIntoProto_master(New_weights, Sanitized_Updates):
    status = api_pb2.Status()
    status.event_type = status.RATELIMIT

    rate_limit = status.rate_limit
    weightmatrix = rate_limit.weightmatrix
    # updatPre=rate_limit.updatPre

    weightmatrix_0 = weightmatrix.weight_0
    value = weightmatrix_0.value
    Row, Co = New_weights[0].shape
    for i in range(Row):
        for j in range(Co):
            value.append(New_weights[0][i, j])

    # New_weights_0_new=New_weights[0]
    # for i in range(Row):
    #     for j in range (Co):
    #         New_weights_0_new[i,j]=value.pop()

    Row_t = New_weights[1].shape
    Row = Row_t[0]
    weightmatrix_1 = weightmatrix.weight_1
    value = weightmatrix_1.value
    for i in range(Row):
        value.append(
            New_weights[1][
                i,
            ]
        )

    Row_t = New_weights[3].shape
    Row = Row_t[0]
    weightmatrix_3 = weightmatrix.weight_3
    value = weightmatrix_3.value
    for i in range(Row):
        value.append(
            New_weights[3][
                i,
            ]
        )

    weightmatrix_2 = weightmatrix.weight_2
    value = weightmatrix_2.value
    Row, Co = New_weights[2].shape
    for i in range(Row):
        for j in range(Co):
            value.append(New_weights[2][i, j])

    ##################
    updatPre = rate_limit.updatPre
    updatPre_0 = updatPre.weight_0
    value = updatPre_0.value
    Row, Co = Sanitized_Updates[0].shape
    for i in range(Row):
        for j in range(Co):
            value.append(Sanitized_Updates[0][i, j])

    # Sanitized_Updates_0_new=Sanitized_Updates[0]
    # for i in range(Row):
    #     for j in range (Co):
    #         Sanitized_Updates_0_new[i,j]=value.pop()

    Row_t = Sanitized_Updates[1].shape
    Row = Row_t[0]
    updatPre_1 = updatPre.weight_1
    value = updatPre_1.value
    for i in range(Row):
        value.append(
            Sanitized_Updates[1][
                i,
            ]
        )

    Row_t = Sanitized_Updates[3].shape
    Row = Row_t[0]
    updatPre_3 = updatPre.weight_3
    value = updatPre_3.value
    for i in range(Row):
        value.append(
            Sanitized_Updates[3][
                i,
            ]
        )

    updatPre_2 = updatPre.weight_2
    value = updatPre_2.value
    Row, Co = Sanitized_Updates[2].shape
    for i in range(Row):
        for j in range(Co):
            value.append(Sanitized_Updates[2][i, j])

    return status


def extractOutProto_slave(Old_weights, status):

    # status = api_pb2.Status()
    New_weights = copy.deepcopy(Old_weights)
    worker_update = status.worker_update

    # real_round=worker_update.my_round
    my_id = worker_update.my_id
    my_ifSig = worker_update.my_ifSig

    if my_ifSig > 0:
        k = 0
        weightmatrix = worker_update.my_weight
        weightmatrix_0 = weightmatrix.weight_0
        value = weightmatrix_0.value
        Row, Co = New_weights[0].shape
        for i in range(Row):
            for j in range(Co):
                New_weights[0][i, j] = value[k]
                k += 1

        k = 0
        weightmatrix_1 = weightmatrix.weight_1
        value = weightmatrix_1.value
        Row_t = New_weights[1].shape
        Row = Row_t[0]
        for i in range(Row):
            New_weights[1][
                i,
            ] = value[k]
            k += 1

        k = 0
        weightmatrix_3 = weightmatrix.weight_3
        value = weightmatrix_3.value
        Row_t = New_weights[3].shape
        Row = Row_t[0]
        for i in range(Row):
            New_weights[3][
                i,
            ] = value[k]
            k += 1

        k = 0
        weightmatrix_2 = weightmatrix.weight_2
        value = weightmatrix_2.value
        Row, Co = New_weights[2].shape
        for i in range(Row):
            for j in range(Co):
                New_weights[2][i, j] = value[k]
                k += 1

    return my_id, my_ifSig, New_weights


def extractOutProto_master(Old_weights, status):

    # status = api_pb2.Status()
    New_weights = copy.deepcopy(Old_weights)
    rate_limit = status.rate_limit
    weightmatrix = rate_limit.weightmatrix

    k = 0
    weightmatrix_0 = weightmatrix.weight_0
    value = weightmatrix_0.value
    Row, Co = New_weights[0].shape
    for i in range(Row):
        for j in range(Co):
            New_weights[0][i, j] = value[k]
            k += 1

    k = 0
    weightmatrix_1 = weightmatrix.weight_1
    value = weightmatrix_1.value
    Row_t = New_weights[1].shape
    Row = Row_t[0]
    for i in range(Row):
        New_weights[1][
            i,
        ] = value[k]
        k += 1

    k = 0
    weightmatrix_3 = weightmatrix.weight_3
    value = weightmatrix_3.value
    Row_t = New_weights[3].shape
    Row = Row_t[0]
    for i in range(Row):
        New_weights[3][
            i,
        ] = value[k]
        k += 1

    k = 0
    weightmatrix_2 = weightmatrix.weight_2
    value = weightmatrix_2.value
    Row, Co = New_weights[2].shape
    for i in range(Row):
        for j in range(Co):
            New_weights[2][i, j] = value[k]
            k += 1

    #########
    Sanitized_Updates = copy.deepcopy(Old_weights)
    rate_limit = status.rate_limit
    updatPre = rate_limit.updatPre

    k = 0
    updatPre_0 = updatPre.weight_0
    value = updatPre_0.value
    Row, Co = Sanitized_Updates[0].shape
    for i in range(Row):
        for j in range(Co):
            Sanitized_Updates[0][i, j] = value[k]
            k += 1

    k = 0
    updatPre_1 = updatPre.weight_1
    value = updatPre_1.value
    Row_t = Sanitized_Updates[1].shape
    Row = Row_t[0]
    for i in range(Row):
        Sanitized_Updates[1][
            i,
        ] = value[k]
        k += 1

    k = 0
    updatPre_3 = updatPre.weight_3
    value = updatPre_3.value
    Row_t = Sanitized_Updates[3].shape
    Row = Row_t[0]
    for i in range(Row):
        Sanitized_Updates[3][
            i,
        ] = value[k]
        k += 1

    k = 0
    updatPre_2 = updatPre.weight_2
    value = updatPre_2.value
    Row, Co = Sanitized_Updates[2].shape
    for i in range(Row):
        for j in range(Co):
            Sanitized_Updates[2][i, j] = value[k]
            k += 1

    return New_weights, Sanitized_Updates


def Assignements(dic):
    return [
        tf.assign(var, dic[Vname_to_Pname(var)]) for var in tf.trainable_variables()
    ]


def Vname_to_Pname(var):
    return var.name[: var.name.find(":")] + "_placeholder"


def Vname_to_FeedPname(var):
    return var.name[: var.name.find(":")] + "_placeholder:0"


def Vname_to_Vname(var):
    return var.name[: var.name.find(":")]


class WeightsAccountant:
    def __init__(self, sess, model, Sigma, real_round):

        self.Weights = [
            np.expand_dims(sess.run(v), -1) for v in tf.trainable_variables()
        ]
        self.keys = [Vname_to_FeedPname(v) for v in tf.trainable_variables()]

        # The trainable parameters are [q x p] matrices, we expand them to [q x p x 1] in order to later stack them
        # along the third dimension.

        # Create a list out of the model dictionary in the order in which the graph holds them:

        self.global_model = [model[k] for k in self.keys]
        self.Sigma = Sigma
        self.Updates = []
        self.median = []
        self.Norms = []
        self.ClippedUpdates = []
        self.m = 0.0
        self.num_weights = len(self.Weights)
        self.round = real_round

    def save_params(self, save_dir):
        filehandler = open(
            save_dir + "/Wweights_accountant_round_" + self.round + ".pkl", "wb"
        )
        pickle.dump(self, filehandler)
        filehandler.close()

    def checkSignificance(self, New_weight, prev_Sanitized_Updates, threshold):

        Updates = [
            np.expand_dims(New_weight[i], -1) - np.expand_dims(self.global_model[i], -1)
            for i in range(self.num_weights)
        ]

        Compare_1 = [
            Updates[i] * np.expand_dims(prev_Sanitized_Updates[i], -1)
            for i in range(self.num_weights)
        ]
        Count_1 = np.sum(
            np.sum(np.array(Compare_1[i]) > 0) for i in range(self.num_weights)
        )

        Compare_2 = [
            Updates[i] - np.expand_dims(prev_Sanitized_Updates[i], -1)
            for i in range(self.num_weights)
        ]
        Count_2 = np.sum(
            np.sum(np.array(Compare_1[i]) == 0) for i in range(self.num_weights)
        )
        Count_2 = Count_2 + Count_1

        # Compare_3 = [Updates[i] * np.expand_dims(prev_Sanitized_Updates[i], -1) for i in range(self.num_weights)]
        Count_3 = np.sum(
            np.sum(np.array(Compare_1[i]) < 0) for i in range(self.num_weights)
        )

        number_weight = np.sum(
            self.global_model[i].size for i in range(self.num_weights)
        )

        Ratio_1 = Count_1 / number_weight
        Ratio_2 = Count_2 / number_weight

        Ratio_3 = 1 - Count_3 / number_weight

        if Ratio_3 > threshold:
            return True
        else:
            return False

    def checkSignificance_Gia(self, New_weight, prev_Sanitized_Updates, threshold):

        Updates = [
            np.expand_dims(New_weight[i], -1) - np.expand_dims(self.global_model[i], -1)
            for i in range(self.num_weights)
        ]
        ratio = [
            np.divide(Updates[i], np.expand_dims(self.global_model[i], -1))
            for i in range(self.num_weights)
        ]

        sum_ratio = np.sum(
            np.sum(np.absolute(ratio[i])) for i in range(self.num_weights)
        )
        num_ratio = np.sum(ratio[i].size for i in range(self.num_weights))
        ratio_aver = sum_ratio / num_ratio

        if ratio_aver > threshold:
            return True
        else:
            return False

    def allocate(self, sess):

        self.Weights = [
            np.concatenate(
                (
                    self.Weights[i],
                    np.expand_dims(sess.run(tf.trainable_variables()[i]), -1),
                ),
                -1,
            )
            for i in range(self.num_weights)
        ]

    def allocate_new(self, New_weights):
        self.Weights = [
            np.concatenate((self.Weights[i], np.expand_dims(New_weights[i], -1)), -1)
            for i in range(self.num_weights)
        ]
        # The trainable parameters are [q x p] matrices, we expand them to [q x p x 1] in order to stack them
        # along the third dimension to the already allocated older variables. We therefore have a list of 6 numpy arrays
        # , each numpy array having three dimensions. The last dimension is the one, the individual weight
        # matrices are stacked along.

    def compute_updates(self):

        # To compute the updates, we subtract the global model from each individual weight matrix. Note:
        # self.Weights[i] is of size [q x p x m], where m is the number of clients whose matrices are stored.
        # global_model['i'] is of size [q x p], in order to broadcast correctly, we have to add a dim.

        self.Updates = [
            self.Weights[i] - np.expand_dims(self.global_model[i], -1)
            for i in range(self.num_weights)
        ]
        self.Weights = None

    def compute_norms(self):

        # The norms List shall have 6 entries, each of size [1x1xm], we keep the first two dimensions because
        # we will later broadcast the Norms onto the Updates of size [q x p x m]

        self.Norms = [
            np.sqrt(
                np.sum(
                    np.square(self.Updates[i]),
                    axis=tuple(range(self.Updates[i].ndim)[:-1]),
                    keepdims=True,
                )
            )
            for i in range(self.num_weights)
        ]

    def clip_updates(self):
        self.compute_updates()
        # self.compute_norms()
        #
        # # The median is a list of 6 entries, each of size [1x1x1],
        #
        # self.median = [np.median(self.Norms[i], axis=-1, keepdims=True) for i in range(self.num_weights)]
        #
        # # The factor is a list of 6 entries, each of size [1x1xm]
        #
        # factor = [self.Norms[i]/self.median[i] for i in range(self.num_weights)]
        # for i in range(self.num_weights):
        #     factor[i][factor[i] > 1.0] = 1.0

        # self.ClippedUpdates = [self.Updates[i]/factor[i] for i in range(self.num_weights)]
        self.ClippedUpdates = [self.Updates[i] for i in range(self.num_weights)]

    def Update_via_GaussianMechanism(self, sess, Acc, FLAGS, Computed_deltas):
        t1, t2, update_perRpund = self.Weights[0].shape

        self.clip_updates()
        self.m = float(self.ClippedUpdates[0].shape[-1])
        MeanClippedUpdates = [
            np.mean(self.ClippedUpdates[i], -1) for i in range(self.num_weights)
        ]

        # GaussianNoise = [(1.0/self.m * np.random.normal(loc=0.0, scale=float(self.Sigma * self.median[i]), size=MeanClippedUpdates[i].shape)) for i in range(self.num_weights)]

        Sanitized_Updates = [
            MeanClippedUpdates[i] for i in range(self.num_weights)
        ]  # +GaussianNoise[i] for i in range(self.num_weights)]

        New_weights = [
            self.global_model[i] + Sanitized_Updates[i] for i in range(self.num_weights)
        ]

        New_weights_status = putIntoProto_master(New_weights, Sanitized_Updates)

        # New_weights_return= extractOutProto_master( New_weights,New_weights_status)

        # New_weights_status.SerializeToString()

        New_model = dict(zip(self.keys, New_weights))

        # t = Acc.accumulate_privacy_spending(0, self.Sigma, self.m)
        # delta = 1
        # if FLAGS.record_privacy == True:
        #     if FLAGS.relearn == False:
        #         # I.e. we never learned a complete model before and have therefore never computed all deltas.
        #         for j in range(len(self.keys)):
        #             sess.run(t)
        #         r = Acc.get_privacy_spent(sess, [FLAGS.eps])
        #         delta = r[0][1]
        #     else:
        #         # I.e. we have computed a complete model before and can reuse the deltas from that time.
        #         delta = Computed_deltas[self.round]
        return New_model, 1, Sanitized_Updates, update_perRpund

    def Update_Slave(self, New_weights):

        New_model = dict(zip(self.keys, New_weights))

        return New_model

    def Update_via_GaussianMechanism_3(self, sess, Acc):
        t1, t2, update_perRpund = self.Weights[0].shape

        self.clip_updates()
        self.m = float(self.ClippedUpdates[0].shape[-1])
        MeanClippedUpdates = [
            np.mean(self.ClippedUpdates[i], -1) for i in range(self.num_weights)
        ]

        GaussianNoise = [
            (
                1.0
                / self.m
                * np.random.normal(
                    loc=0.0,
                    scale=float(self.Sigma * self.median[i]),
                    size=MeanClippedUpdates[i].shape,
                )
            )
            for i in range(self.num_weights)
        ]

        Sanitized_Updates = [
            MeanClippedUpdates[i] for i in range(self.num_weights)
        ]  # +GaussianNoise[i] for i in range(self.num_weights)]

        New_weights = [
            self.global_model[i] + Sanitized_Updates[i] for i in range(self.num_weights)
        ]

        New_model = dict(zip(self.keys, New_weights))

        t = Acc.accumulate_privacy_spending(0, self.Sigma, self.m)
        delta = 1

        return New_model, delta, Sanitized_Updates, update_perRpund

    def Update_via_GaussianMechanism_Method_2(
        self, sess, Acc, FLAGS, Computed_deltas, m, threshold
    ):

        temp_Updates = [
            self.Weights[i] - np.expand_dims(self.global_model[i], -1)
            for i in range(self.num_weights)
        ]
        for i in range(self.num_weights):
            t1, t2, t3 = temp_Updates[i].shape
            for j in range(t1):
                for k in range(t2):
                    for f in range(t3):
                        if temp_Updates[i][j][k][f] > 0:
                            temp_Updates[i][j][k][f] = 1
                        if temp_Updates[i][j][k][f] < 0:
                            temp_Updates[i][j][k][f] = -1
        temp_aver = [np.expand_dims(sess.run(v), -1) for v in tf.trainable_variables()]
        for i in range(self.num_weights):
            temp_aver[i] = np.average(temp_Updates[i], 2)

        temp_aver_0 = np.average(temp_Updates[0], 2)
        temp_aver_1 = np.average(temp_Updates[1], 2)
        temp_aver_2 = np.average(temp_Updates[2], 2)

        delete_index = []

        for c in range(m):
            Compare_1 = [
                temp_Updates[i][:, :, c] * temp_aver[i] for i in range(self.num_weights)
            ]
            Count_3 = np.sum(
                np.sum(np.array(Compare_1[i]) < 0) for i in range(self.num_weights)
            )
            number_weight = np.sum(
                self.global_model[i].size for i in range(self.num_weights)
            )

            Ratio_3 = 1 - Count_3 / number_weight
            if Ratio_3 < threshold:

                delete_index.append(c)
        for i in range(self.num_weights):
            self.Weights[i] = np.delete(self.Weights[i], delete_index, 2)

        t1, t2, update_perRpund = self.Weights[0].shape
        self.clip_updates()
        self.m = float(self.ClippedUpdates[0].shape[-1])
        MeanClippedUpdates = [
            np.mean(self.ClippedUpdates[i], -1) for i in range(self.num_weights)
        ]

        GaussianNoise = [
            (
                1.0
                / self.m
                * np.random.normal(
                    loc=0.0,
                    scale=float(self.Sigma * self.median[i]),
                    size=MeanClippedUpdates[i].shape,
                )
            )
            for i in range(self.num_weights)
        ]

        Sanitized_Updates = [
            MeanClippedUpdates[i] for i in range(self.num_weights)
        ]  # +GaussianNoise[i] for i in range(self.num_weights)]

        New_weights = [
            self.global_model[i] + Sanitized_Updates[i] for i in range(self.num_weights)
        ]

        New_model = dict(zip(self.keys, New_weights))

        t = Acc.accumulate_privacy_spending(0, self.Sigma, self.m)
        delta = 1
        if FLAGS.record_privacy == True:
            if FLAGS.relearn == False:
                # I.e. we never learned a complete model before and have therefore never computed all deltas.
                for j in range(len(self.keys)):
                    sess.run(t)
                r = Acc.get_privacy_spent(sess, [FLAGS.eps])
                delta = r[0][1]
            else:
                # I.e. we have computed a complete model before and can reuse the deltas from that time.
                delta = Computed_deltas[self.round]
        return New_model, delta, Sanitized_Updates, update_perRpund


def create_save_dir(FLAGS, use_signi, threshold):
    """
    :return: Returns a path that is used to store training progress; the path also identifies the chosen setup uniquely.
    """
    raw_directory = os.getcwd() + "/"
    if FLAGS.gm:
        gm_str = "Dp/"
    else:
        gm_str = "non_Dp/"
    if use_signi:
        sign_str = "sig_"
    else:
        sign_str = "Nonsig_"
    if FLAGS.priv_agent:
        model = (
            gm_str
            + sign_str
            + "N_"
            + str(FLAGS.n)
            + "Thre_"
            + str(threshold)
            + "/Epochs_"
            + str(int(FLAGS.e))
            + "_Batches_"
            + str(int(FLAGS.b))
        )
        return raw_directory + str(model) + "/" + FLAGS.PrivAgentName
    else:
        model = (
            gm_str
            + sign_str
            + "N_"
            + str(FLAGS.n)
            + "Thre_"
            + str(threshold)
            + "/Sigma_"
            + str(FLAGS.Sigma)
            + "_C_"
            + str(FLAGS.m)
            + "/Epochs_"
            + str(int(FLAGS.e))
            + "_Batches_"
            + str(int(FLAGS.b))
        )
        return raw_directory + str(model)


def load_from_directory_or_initialize(directory, FLAGS):
    """
    This function looks for a model that corresponds to the characteristics specified and loads potential progress.
    If it does not find any model or progress, it initializes a new model.
    :param directory: STRING: the directory where to look for models and progress.
    :param FLAGS: CLASS INSTANCE: holds general trianing params
    :param PrivacyAgent:
    :return:
    """

    Accuracy_accountant = []
    Count_update_perRpund = []
    Delta_accountant = [0]
    model = []
    real_round = 0
    Acc = GaussianMomentsAccountant(FLAGS.n)
    FLAGS.loaded = False
    FLAGS.relearn = False
    Computed_Deltas = []

    if not os.path.isfile(directory + "/model.pkl"):
        # If there is no model stored at the specified directory, we initialize a new one!
        if not os.path.exists(directory):
            os.makedirs(directory)
        print("No loadable model found. All updates stored at: " + directory)
        print("... Initializing a new model ...")

    else:
        # If there is a model, we have to check whether:
        #  - We learned a model for the first time, and interrupted; in that case: resume learning:
        #               set FLAGS.loaded = TRUE
        #  - We completed learning a model and want to learn a new one with the same parameters, i.o. to average accuracies:
        #       In this case we would want to initialize a new model; but would like to reuse the delta's already
        #       computed. So we will load the deltas.
        #               set FLAGS.relearn = TRUE
        #  - We completed learning models and want to resume learning model; this happens if the above process is
        # interrupted. In this case we want to load the model; and reuse the deltas.
        #               set FLAGS.loaded = TRUE
        #               set FLAGS.relearn = TRUE
        if os.path.isfile(directory + "/specs.csv"):
            with open(directory + "/specs.csv", "rb") as csvfile:
                reader = csv.reader(csvfile)
                Lines = []
                for line in reader:
                    Lines.append([float(j) for j in line])

                Accuracy_accountant = Lines[-1]
                Delta_accountant = Lines[1]

            if math.isnan(Delta_accountant[-1]):
                Computed_Deltas = Delta_accountant
                # This would mean that learning was finished at least once, i.e. we are relearning.
                # We shall thus not recompute the deltas, but rather reuse them.
                FLAGS.relearn = True
                if math.isnan(Accuracy_accountant[-1]):
                    # This would mean that we finished learning the latest model.
                    print(
                        "A model identical to that specified was already learned. Another one is learned and appended"
                    )
                    Accuracy_accountant = []
                    Delta_accountant = [0]
                else:
                    # This would mean we already completed learning a model once, but the last one stored was not completed
                    print(
                        "A model identical to that specified was already learned. For a second one learning is resumed"
                    )
                    # We set the delta accountant accordingly
                    Delta_accountant = Delta_accountant[: len(Accuracy_accountant)]
                    # We specify that a model was loaded
                    real_round = len(Accuracy_accountant) - 1
                    fil = open(directory + "/model.pkl", "rb")
                    model = pickle.load(fil)
                    fil.close()
                    FLAGS.loaded = True
                return (
                    model,
                    Accuracy_accountant,
                    Delta_accountant,
                    Acc,
                    real_round,
                    FLAGS,
                    Computed_Deltas,
                    Count_update_perRpund,
                )
            else:
                # This would mean that learning was never finished, i.e. the first time a model with this specs was
                # learned got interrupted.
                real_round = len(Accuracy_accountant) - 1
                fil = open(directory + "/model.pkl", "rb")
                model = pickle.load(fil)
                fil.close()
                FLAGS.loaded = True
                print(directory)
                # print('A model identical to that specified was already learned. For a second one learning is resumed')
        else:
            print("there seems to be a model, but no saved progress. Fix that.")
            raise KeyboardInterrupt
    return (
        model,
        Accuracy_accountant,
        Delta_accountant,
        Acc,
        real_round,
        FLAGS,
        Computed_Deltas,
        Count_update_perRpund,
    )


def save_progress(
    save_dir,
    model,
    Delta_accountant,
    Accuracy_accountant,
    PrivacyAgent,
    FLAGS,
    Count_update_perRpund,
):
    """
    This function saves our progress either in an existing file structure or writes a new file.
    :param save_dir: STRING: The directory where to save the progress.
    :param model: DICTIONARY: The model that we wish to save.
    :param Delta_accountant: LIST: The list of deltas that we allocared so far.
    :param Accuracy_accountant: LIST: The list of accuracies that we allocated so far.
    :param PrivacyAgent: CLASS INSTANCE: The privacy agent that we used (specifically the m's that we used for Federated training.)
    :param FLAGS: CLASS INSTANCE: The FLAGS passed to the learning procedure.
    :return: nothing
    """
    filehandler = open(save_dir + "/model.pkl", "wb")
    pickle.dump(model, filehandler)
    filehandler.close()

    if FLAGS.relearn == False:
        # I.e. we know that there was no progress stored at 'save_dir' and we create a new csv-file that
        # Will hold the accuracy, the deltas, the m's and we also save the model learned as a .pkl file

        with open(save_dir + "/specs.csv", "wb") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            if FLAGS.priv_agent == True:
                writer.writerow(
                    [0]
                    + [PrivacyAgent.get_m(r) for r in range(len(Delta_accountant) - 1)]
                )
            if FLAGS.priv_agent == False:
                writer.writerow([0] + [FLAGS.m] * (len(Delta_accountant) - 1))
            writer.writerow(Delta_accountant)
            writer.writerow(Accuracy_accountant)
            writer.writerow(Count_update_perRpund)

    if FLAGS.relearn == True:
        # If there already is progress associated to the learned model, we do not need to store the deltas and m's as
        # they were already saved; we just keep track of the accuracy and append it to the already existing .csv file.
        # This will help us later on to average the performance, as the variance is very high.

        if (
            len(Accuracy_accountant) > 1
            or len(Accuracy_accountant) == 1
            and FLAGS.loaded is True
        ):
            # If we already appended a new line to the .csv file, we have to delete that line.
            with open(save_dir + "/specs.csv", "r+w") as csvfile:
                csvReader = csv.reader(csvfile, delimiter=",")
                lines = []
                for row in csvReader:
                    lines.append([float(i) for i in row])
                lines = lines[:-1]

            with open(save_dir + "/specs.csv", "wb") as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                for line in lines:
                    writer.writerow(line)

        # Append a line to the .csv file holding the accuracies.
        with open(save_dir + "/specs.csv", "a") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(Accuracy_accountant)
            writer.writerow(Count_update_perRpund)


def global_step_creator():
    global_step = [v for v in tf.global_variables() if v.name == "global_step:0"][0]
    global_step_placeholder = tf.placeholder(
        dtype=tf.float32, shape=(), name="global_step_placeholder"
    )
    one = tf.constant(1, dtype=tf.float32, name="one")
    new_global_step = tf.add(global_step, one)
    increase_global_step = tf.assign(global_step, new_global_step)
    set_global_step = tf.assign(global_step, global_step_placeholder)
    return increase_global_step, set_global_step


def bring_Accountant_up_to_date(Acc, sess, rounds, PrivAgent, FLAGS):
    """

    :param Acc: A Privacy accountant
    :param sess: A tensorflow session
    :param rounds: the number of rounds that the privacy accountant shall iterate
    :param PrivAgent: A Privacy_agent that has functions: PrivAgent.get_Sigma(round) and PrivAgent.get_m(round)
    :param FLAGS: priv_agent specifies whether to use a PrivAgent or not.
    :return:
    """
    print("Bringing the accountant up to date....")

    for r in range(rounds):
        if FLAGS.priv_agent:
            Sigma = PrivAgent.get_Sigma(r)
            m = PrivAgent.get_m(r)
        else:
            Sigma = FLAGS.sigma
            m = FLAGS.m
        print("Completed " + str(r + 1) + " out of " + str(rounds) + " rounds")
        t = Acc.accumulate_privacy_spending(0, Sigma, m)
        sess.run(t)
        sess.run(t)
        sess.run(t)
    print("The accountant is up to date!")


def print_loss_and_accuracy(global_loss, accuracy, update_perRpund):
    print(" - Current Model has a loss of:           %s" % global_loss)
    print(" - The Accuracy on the validation set is: %s" % accuracy)
    print(" - #of clients updating: %s" % update_perRpund)
    print(
        "--------------------------------------------------------------------------------------"
    )
    print(
        "--------------------------------------------------------------------------------------"
    )


def print_new_comm_round(real_round):
    print(
        "--------------------------------------------------------------------------------------"
    )
    print(
        "------------------------ Communication round %s ---------------------------------------"
        % str(real_round)
    )
    print(
        "--------------------------------------------------------------------------------------"
    )


def check_validaity_of_FLAGS(FLAGS):
    FLAGS.priv_agent = True
    if not FLAGS.m == 0:
        if FLAGS.sigma == 0:
            print(
                "\n \n -------- If m is specified the Privacy Agent is not used, then Sigma has to be specified too. --------\n \n"
            )
            raise NotImplementedError
    if not FLAGS.sigma == 0:
        if FLAGS.m == 0:
            print(
                "\n \n-------- If Sigma is specified the Privacy Agent is not used, then m has to be specified too. -------- \n \n"
            )
            raise NotImplementedError
    if not FLAGS.sigma == 0 and not FLAGS.m == 0:
        FLAGS.priv_agent = False
    return FLAGS


class Flag:
    def __init__(
        self,
        n,
        b,
        e,
        record_privacy,
        m,
        sigma,
        eps,
        save_dir,
        log_dir,
        max_comm_rounds,
        gm,
        PrivAgent,
    ):
        if not save_dir:
            save_dir = os.getcwd()
        if not log_dir:
            log_dir = os.path.join(
                os.getenv("TEST_TMPDIR", "/tmp"),
                "tensorflow/mnist/logs/fully_connected_feed",
            )
        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)
        self.n = n
        self.sigma = sigma
        self.eps = eps
        self.m = m
        self.b = b
        self.e = e
        self.record_privacy = record_privacy
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.max_comm_rounds = max_comm_rounds
        self.gm = gm
        self.PrivAgentName = PrivAgent.Name
