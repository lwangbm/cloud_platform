#!/usr/bin/env python

import sys

import numpy as np

sys.path.append("/Users/ourokutaira/Desktop/edge/implementation/zebra/")
import argparse
import os
import sys

import tensorflow as tf
from twisted.internet import endpoints, reactor
from twisted.internet.protocol import ServerFactory
from twisted.protocols.basic import NetstringReceiver

from api.communication import api_pb2
from feder.Helper_Functions import extractOutProto_slave, putIntoProto_master
from master.scheduler import Scheduler
from utils import logging, parse_hosts
from utils.constants import CLOUD_PORT

# two phases
# 1. each slave(host) registers to master
# 2. sync-ing message between slaves/executors


class Broker(NetstringReceiver):
    def __init__(self, factory):
        self.factory = factory
        self.MAX_LENGTH = 99999999
        # self.workerCount=0
        # self.workerCount_true=0

    def connectionLost(self, reason):
        logging.info(reason)
        self.factory.host_n -= 1
        if self.factory.host_n == 0:
            reactor.stop()

    def stringReceived(self, string):
        status = api_pb2.Status()
        status.ParseFromString(string)

        # check if is in the host initialization phase
        if status.event_type is status.REGISTER:
            logging.info("receive a register")
            self.initialize_host()
            return

        if status.event_type is status.WORKER_UPDATE:
            self.update_info(status)
            return

        logging.error("unsupported event type.")

    def update_info(self, status):
        self.factory.status_list.append(status)

        if self.factory.workerCount == 0:
            logging.info("First update arrives in this round")

        self.factory.workerCount += 1

        if self.factory.workerCount == self.factory.host_n:
            logging.info("All uploads have arrived, begin aggregation")

            for i in range(self.factory.host_n):
                status_now = self.factory.status_list[i]
                my_id, my_ifSig, New_weights = extractOutProto_slave(
                    sched.oldWeight, status_now
                )
                if my_ifSig > 0:
                    if self.factory.workerCount_true == 0:
                        self.factory.Weights = [
                            np.expand_dims(New_weights[i], -1)
                            for i in range(sched.oldWeight.__len__())
                        ]
                        logging.info("Strat extract factory.status_list")

                    else:
                        self.factory.Weights = [
                            np.concatenate(
                                (
                                    self.factory.Weights[i],
                                    np.expand_dims(New_weights[i], -1),
                                ),
                                -1,
                            )
                            for i in range(sched.oldWeight.__len__())
                        ]
                    self.factory.workerCount_true += my_ifSig

            sched.SartNewRound(
                Weights=self.factory.Weights,
                workerCount_true=self.factory.workerCount_true,
            )

            self.factory.workerCount = 0
            self.factory.workerCount_true = 0
            self.factory.status_list = []

            New_weights_status = putIntoProto_master(
                sched.Weight_for_update, sched.prev_Sanitized_Updates
            )

            logging.info("Master finished Aver")

            self.factory.update_rates(New_weights_status)

            logging.info("Master finished update in this round")

    ######
    #  #logging.info("Receive a new status, now will start extract it")
    #
    #  my_id, my_ifSig,New_weights = extractOutProto_slave(sched.oldWeight,status)
    #  #logging.info("Started update info, the worker is salve-%s" % my_id)
    #
    #  if my_ifSig:
    #
    #      if self.factory.workerCount_true == 0:
    #          self.factory.Weights = [np.expand_dims(New_weights[i], -1) for i in range(4)];
    #          logging.info("First update arrives in this round")
    #
    #      else:
    #          self.factory.Weights = [
    #              np.concatenate((self.factory.Weights[i], np.expand_dims(New_weights[i], -1)), -1)
    #              for i in range(sched.oldWeight.__len__())]
    #      self.factory.workerCount_true+=1
    #
    #  self.factory.workerCount+=1
    #
    # # logging.info("Finished update info, the worker is salve-%s" % my_id)
    #
    #  if self.factory.workerCount == self.factory.host_n:
    #      logging.info("All updates arrive, begin Aver")
    #
    #      # logging.info("All the workers in this round have updated, started Aver" )
    #      sched.SartNewRound(Weights=self.factory.Weights,workerCount_true=self.factory.workerCount_true)
    #      self.factory.workerCount = 0
    #      self.factory.workerCount_true = 0
    #
    #      New_weights_status = putIntoProto_master ( sched.Weight_for_update,sched.prev_Sanitized_Updates)
    #
    #      logging.info("Master finished Aver")
    #
    #      self.factory.update_rates(New_weights_status)
    #
    #      logging.info("Master finished update in this round")

    def initialize_host(self):
        if self.factory.has_reached_n:
            logging.error("suspicious connection made")
            return

        ip = self.transport.getPeer().host
        hostname = self.factory.ip_to_host[ip]
        logging.info("The received register is %s" % ip)
        logging.info("The received hostname is %s" % hostname)
        self.factory.hosts[hostname] = self

        if self.factory.host_n == len(self.factory.hosts):
            self.factory.has_reached_n = True
            logging.info("server gets all connections.")
            self.factory.launch_job()


class BrokerFactory(ServerFactory):

    protocol = Broker

    def __init__(self, host_n):
        self.host_n = host_n

        # self.rate_idx = defaultdict(int) #default dict, with agrs of TYPE, defalut of int 0, the setting count of each updating-rate to each flow, flow_uid->int:0,1,2,...

        # hostname -> broker, self.factory.hosts[hostname] = self, hostname->broker
        self.hosts = {}
        self.has_reached_n = False
        # a dict for /etc/hosts, ip->host name , 13.59.143.120	->   slave-2
        self.ip_to_host = parse_hosts()

        # remember the server executors to avoid duplicates
        self.recv_addrs = set()  # init a set to avod duplicates
        self.workerCount = 0
        self.workerCount_true = 0
        self.status_list = []

    def buildProtocol(self, addr):
        return self.protocol(self)

    def lose_all_connections(self):
        for broker in self.hosts.values():
            broker.transport.loseConnection()

    def update_rates(self, New_weights_status):
        # the rate in each job in each flow has already been calculated, this is just update, like @onSchedule()

        logging.info("begin update rates in master")
        for send_index in range(self.host_n):
            hostname = "slave-{}".format(send_index)  # hostname is like "slave+id"
            self.hosts[hostname].sendString(New_weights_status.SerializeToString())
        logging.info("done update rates in master")

    def launch_job(self):
        logging.info("launch job.")
        # sched.startMaster()

        for i in range(self.host_n):
            status = api_pb2.Status()
            status.event_type = status.LAUNCH
            launch = status.launch
            launch.worker_id = i
            send_hostname = "slave-{}".format(i)
            self.hosts[send_hostname].sendString(status.SerializeToString())
        logging.info("done launch job.")
        # sched.startMaster()


def serve():
    endpoints.serverFromString(reactor, "tcp:%d" % CLOUD_PORT).listen(broker_factory)
    # reactor.run(installSignalHandlers=0)
    reactor.run()


def get_args():
    import argparse

    # parser = argparse.ArgumentParser(description="Zzzzzebra")
    #
    # parser.add_argument("sharing_algo")
    # parser.add_argument("trace_path")
    # parser.add_argument("fairness", nargs='?', type=float, default=1)
    # return parser.parse_args()
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
    return FLAGS


def main(_):

    global sched, broker_factory

    broker_factory = BrokerFactory(FLAGS.host_n / 10)

    use_signi = True
    threshold = 0.6011
    methodSig = 1

    # FLAGS = get_args()
    sched = Scheduler(
        N=FLAGS.host_n,
        b=FLAGS.b,
        e=FLAGS.e,
        m=FLAGS.m,
        sigma=FLAGS.sigma,
        eps=FLAGS.eps,
        save_dir=os.getcwd(),
        log_dir=FLAGS.log_dir,
        broker_factory=broker_factory,
        use_signi=use_signi,
        threshold=threshold,
        methodSig=methodSig,
    )

    serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
