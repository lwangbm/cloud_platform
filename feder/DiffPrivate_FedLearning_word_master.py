from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from feder.Helper_Functions import (Flag, PrivAgent, Vname_to_FeedPname,
                                    Vname_to_Pname, WeightsAccountant,
                                    bring_Accountant_up_to_date,
                                    check_validaity_of_FLAGS, create_save_dir,
                                    global_step_creator,
                                    load_from_directory_or_initialize,
                                    print_loss_and_accuracy,
                                    print_new_comm_round, save_progress)


def run_differentially_private_federated_averaging(
    num_input,
    num_classes,
    window,
    loss,
    train_op,
    eval_correct,
    data,
    data_placeholder,
    label_placeholder,
    privacy_agent=None,
    b=10,
    e=4,
    record_privacy=True,
    m=0,
    sigma=0,
    eps=8,
    save_dir=None,
    worker_id=None,
    log_dir=None,
    max_comm_rounds=3000,
    gm=False,  # True,
    saver_func=create_save_dir,
    save_params=False,
    use_signi=True,
    threshold=0.1,
    methodSig=1,
    Weights=None,
    my_round_master=0,
    workerCount_true=0,
):
    # If no privacy agent was specified, the default privacy agent is used.
    if not privacy_agent:
        privacy_agent = PrivAgent(len(data.client_set), "default_agent")

    # A Flags instance is created that will fuse all specified parameters and default those that are not specified.
    FLAGS = Flag(
        len(data.client_set),
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
        privacy_agent,
    )

    # Check whether the specified parameters make sense.
    FLAGS = check_validaity_of_FLAGS(FLAGS)

    # At this point, FLAGS.save_dir specifies both; where we save progress and where we assume the data is stored
    save_dir = saver_func(FLAGS, use_signi, threshold)

    # This function will retrieve the variable associated to the global step and create nodes that serve to
    # increase and reset it to a certain value.
    increase_global_step, set_global_step = global_step_creator()

    # - model_placeholder : a dictionary in which there is a placeholder stored for every trainable variable defined
    #                       in the tensorflow graph. Each placeholder corresponds to one trainable variable and has
    #                       the same shape and dtype as that variable. in addition, the placeholder has the same
    #                       name as the Variable, but a '_placeholder:0' added to it. The keys of the dictionary
    #                       correspond to the name of the respective placeholder
    model_placeholder = dict(
        zip(
            [Vname_to_FeedPname(var) for var in tf.trainable_variables()],
            [
                tf.placeholder(
                    name=Vname_to_Pname(var), shape=var.shape, dtype=tf.float32
                )
                for var in tf.trainable_variables()
            ],
        )
    )

    # - assignments : Is a list of nodes. when run, all trainable variables are set to the value specified through
    #                 the placeholders in 'model_placeholder'.

    assignments = [
        tf.assign(var, model_placeholder[Vname_to_FeedPname(var)])
        for var in tf.trainable_variables()
    ]

    # load_from_directory_or_initialize checks whether there is a model at 'save_dir' corresponding to the one we
    # are building. If so, training is resumed, if not, it returns:  - model = []
    #                                                                - accuracy_accountant = []
    #                                                                - delta_accountant = []
    #                                                                - real_round = 0
    # And initializes a Differential_Privacy_Accountant as acc

    (
        model,
        accuracy_accountant,
        delta_accountant,
        acc,
        real_round,
        FLAGS,
        computed_deltas,
        Count_update_perRpund,
    ) = load_from_directory_or_initialize(save_dir, FLAGS)

    m = int(FLAGS.m)
    sigma = float(FLAGS.sigma)
    # - m : amount of clients participating in a round
    # - sigma : variable for the Gaussian Mechanism.
    # Both will only be used if no Privacy_Agent is deployed.

    ################################################################################################################

    # Usual Tensorflow...

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    ################################################################################################################

    # If there was no loadable model, we initialize a model:
    # - model : dictionary having as keys the names of the placeholders associated to each variable. It will serve
    #           as a feed_dict to assign values to the placeholders which are used to set the variables to
    #           specific values.

    if not model:
        model = dict(
            zip(
                [Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                [sess.run(var) for var in tf.trainable_variables()],
            )
        )
        model["global_step_placeholder:0"] = 0

        real_round = 0

        weights_accountant = []

    # If a model is loaded, and we are not relearning it (relearning means that we once already finished such a model
    # and we are learning it again to average the outcomes), we have to get the privacy accountant up to date. This
    # means, that we have to iterate the privacy accountant over all the m, sigmas that correspond to already completed
    # communication

    if not FLAGS.relearn and real_round > 0:
        bring_Accountant_up_to_date(acc, sess, real_round, privacy_agent, FLAGS)

    ################################################################################################################

    # This is where the actual communication rounds start:

    data_set_asarray = np.asarray(data.sorted_x_train)
    label_set_asarray = np.asarray(data.sorted_y_train)
    update_thisRound = FLAGS.m

    batch_size = b
    last_batch = len(data.y_vali) % batch_size
    last_batch = len(data.y_vali)
    batch_size = len(data.y_vali)
    timesteps = window
    Y_batch = data.y_vali
    Y_batch_encoded = []
    X_batch = data.x_vali
    for x in Y_batch:
        on_hot_vector = np.zeros([num_classes], dtype=float)
        on_hot_vector[x] = 1.0
        Y_batch_encoded = np.concatenate((Y_batch_encoded, on_hot_vector))
    if len(X_batch) < batch_size:
        X_batch = np.array(X_batch)
        X_batch = X_batch.reshape(last_batch, timesteps, num_input)
        Y_batch_encoded = np.array(Y_batch_encoded)
        Y_batch_encoded = Y_batch_encoded.reshape(last_batch, num_classes)
    else:
        X_batch = np.array(X_batch)
        X_batch = X_batch.reshape(batch_size, timesteps, num_input)
        Y_batch_encoded = np.array(Y_batch_encoded)
        Y_batch_encoded = Y_batch_encoded.reshape(batch_size, num_classes)

    feed_dict1 = {
        str(data_placeholder.name): X_batch,
        str(label_placeholder.name): Y_batch_encoded,
    }

    real_round = real_round + 1

    print_new_comm_round(my_round_master)
    ############################################################################################################
    # Start of a new communication round

    if FLAGS.priv_agent:
        m = int(privacy_agent.get_m(int(real_round)))
        sigma = privacy_agent.get_Sigma(int(real_round))

    print("Clients participating: " + str(m))

    weights_accountant = WeightsAccountant(sess, model, sigma, real_round)

    weights_accountant.Weights = Weights

    # End of a communication round
    ############################################################################################################

    # print('......Communication round %s completed' % str(my_round_master))
    if methodSig == 1:
        (
            model,
            delta,
            prev_Sanitized_Updates,
            update_thisRound,
        ) = weights_accountant.Update_via_GaussianMechanism(
            sess, acc, FLAGS, computed_deltas
        )
    if methodSig == 3:
        (
            model,
            delta,
            prev_Sanitized_Updates,
            update_thisRound,
        ) = weights_accountant.Update_via_GaussianMechanism(
            sess, acc, FLAGS, computed_deltas
        )
    if methodSig == 2:
        (
            model,
            delta,
            prev_Sanitized_Updates,
            update_thisRound,
        ) = weights_accountant.Update_via_GaussianMechanism_Method_2(
            sess, acc, FLAGS, computed_deltas, m, threshold
        )

    sess.run(assignments, feed_dict=model)

    # compute the loss on the validation set.
    global_loss = sess.run(loss, feed_dict=feed_dict1)
    count = sess.run(eval_correct, feed_dict=feed_dict1)
    accuracy = float(count) / float(len(data.y_vali))
    accuracy_accountant.append(accuracy)
    update_perRpund = update_thisRound
    Count_update_perRpund.append(update_perRpund)

    # print_loss_and_accuracy(global_loss, accuracy, update_perRpund)
    save_progress(
        save_dir,
        model,
        delta_accountant,
        accuracy_accountant,
        privacy_agent,
        FLAGS,
        Count_update_perRpund,
    )
    Weight_for_update = [sess.run(tf.trainable_variables()[i]) for i in range(4)]

    f = open(save_dir + "record.txt", "a")

    f.writelines(["\n", str(workerCount_true), " ", str(accuracy)])

    f.close()

    return Weight_for_update, accuracy, prev_Sanitized_Updates
