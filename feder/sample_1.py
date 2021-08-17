import argparse
import os
import sys

import mnist_inference as mnist
import nltk
import tensorflow as tf
from DiffPrivate_FedLearning_word import \
    run_differentially_private_federated_averaging
from MNIST_reader import Data, Data_word, Data_word_2

nltk.download("punkt")


def sample(N, b, e, m, sigma, eps, save_dir, log_dir):

    # Specs for the model that we would like to train in differentially private federated fashion:
    hidden1 = 600
    hidden2 = 100
    hidden = 512
    window = 10
    num_input = 1

    # Specs for the differentially private federated fashion learning process.

    # A data object that already satisfies client structure and has the following attributes:
    # DATA.data_set : A list of labeld training examples.
    # DATA.client_set : A
    DATA = Data_word_2(save_dir, N, window, b)
    num_classes = len(DATA.vocabulary_dictionary)

    with tf.Graph().as_default():

        # Building the model that we would like to train in differentially private federated fashion.
        # We will need the tensorflow training operation for that model, its loss and an evaluation method:

        (
            train_op,
            eval_correct,
            loss,
            data_placeholder,
            labels_placeholder,
        ) = mnist.word_fully_connected_model(b, hidden, window, num_classes, num_input)

        (
            Accuracy_accountant,
            Delta_accountant,
            model,
        ) = run_differentially_private_federated_averaging(
            num_input,
            num_classes,
            window,
            loss,
            train_op,
            eval_correct,
            DATA,
            data_placeholder,
            labels_placeholder,
            b=b,
            e=e,
            m=m,
            sigma=sigma,
            eps=eps,
            save_dir=save_dir,
            log_dir=log_dir,
        )


def main(_):
    sample(
        N=FLAGS.N,
        b=FLAGS.b,
        e=FLAGS.e,
        m=FLAGS.m,
        sigma=FLAGS.sigma,
        eps=FLAGS.eps,
        save_dir=os.getcwd(),
        log_dir=FLAGS.log_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--save_dir',
    #     type=str,
    #     default=os.getcwd(),
    #     help='directory to store progress'
    # )
    parser.add_argument(
        "--N", type=int, default=100, help="Total Number of clients participating"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0,
        help="The gm variance parameter; will not affect if Priv_agent is set to True",
    )
    parser.add_argument("--eps", type=float, default=8, help="Epsilon")
    parser.add_argument(
        "--m", type=int, default=0, help="Number of clients participating in a round"
    )
    parser.add_argument("--b", type=float, default=2, help="Batches per client")
    parser.add_argument("--e", type=int, default=4, help="Epochs per client")
    parser.add_argument("--save_dir", type=str, default=os.getcwd(), help="Directory")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join(
            os.getenv("TEST_TMPDIR", "/tmp"),
            "tensorflow/mnist/logs/fully_connected_feed",
        ),
        help="Directory to put the log data.",
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)