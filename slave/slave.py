#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import tensorflow as tf
from twisted.internet import reactor, utils
from twisted.internet.protocol import ClientFactory
from twisted.protocols.basic import NetstringReceiver

from api.communication import api_pb2
from feder.Helper_Functions import extractOutProto_master, putIntoProto_slave
from slave.model_maintain import ModelMaintain
from utils import logging
from utils.constants import CLOUD_PORT

pwd = os.path.dirname(os.path.abspath(__file__))
tc_path = os.path.join(pwd, "tc.sh")


class Broker(NetstringReceiver):
    def __init__(self):
        self.MAX_LENGTH = 99999999
        self.worker_id = None

    def connectionMade(self):
        logging.info("connected to master.")
        status = api_pb2.Status()
        status.event_type = status.REGISTER
        self.sendString(status.SerializeToString())

    def connectionLost(self, reason):
        logging.info("slave connection lost.")
        logging.info(reason)
        reactor.stop()

    def stringReceived(self, string):
        status = api_pb2.Status()
        status.ParseFromString(string)

        if status.event_type is status.LAUNCH:
            self.launch_flow(status)
            return

        if status.event_type is status.RATELIMIT:
            self.rate_limit_flow(status)
            return

        logging.error("wrong event type.")

    def rate_limit_flow(self, status):
        # rate_limit = status.rate_limit
        New_weights_return, Sanitized_Updates = extractOutProto_master(
            sched.oldWeight, status
        )

        ifSig = sched.startNewRoundTraining(
            New_weights=New_weights_return, Sanitized_Updates=Sanitized_Updates
        )
        logging.info("new round.")
        # check if sig
        self.update_worker(ifSig)
        # logging.info("update worker.")

    def launch_flow(self, status):  # a flow : a worker
        launch = status.launch
        self.worker_id = launch.worker_id
        sched.startSlave(worker_id=self.worker_id)
        logging.info("launch worker.")

        self.update_worker()
        logging.info("update worker.")

    def update_worker(self):

        New_weights_status = putIntoProto_slave(sched.oldWeight, sched.worker_id)
        # real_round, my_id, my_ifSig, New_weights = extractOutProto_slave(sched.oldWeight, New_weights_status)
        my_test_string = New_weights_status.SerializeToString()
        # print(len(my_test_string))
        # print(sys.getsizeof(my_test_string))
        # print ("This round's weight for update is "+str(sched.oldWeight[0][0,0]))
        # self._extractLength(my_test_string)
        self.sendString(New_weights_status.SerializeToString())
        # print(New_weights_status.SerializeToString())

        # for debug

        # status = api_pb2.Status()
        # status.event_type = status.WORKER_UPDATE
        # worker_update = status.worker_update
        # worker_update.my_id = 1
        # print(status.SerializeToString())
        # # self.sendString(status.SerializeToString())


class BrokerFactory(ClientFactory):
    protocol = Broker

    # keep connecting.
    def clientConnectionFailed(self, connector, reason):
        connector.connect()


def serve():
    reactor.connectTCP(b"master", CLOUD_PORT, BrokerFactory(), timeout=0.5)


def main(_):
    # we first reset the tc on this host

    global local_model
    local_model = ModelMaintain()

    deferred = utils.getProcessOutput(tc_path, args=("reset",), env=os.environ)
    deferred.addErrback(lambda _: logging.error("error on tc reset."))
    deferred.addCallback(lambda _: serve())

    reactor.run()  # @UndefinedVariable


if __name__ == "__main__":
    # main()
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
    parser.add_argument("--e", type=int, default=5, help="Epochs per client")
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
    parser.add_argument("--host_n", type=int, default=100, help="Number of servers")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
