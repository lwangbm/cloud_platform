#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import codecs
import contextlib
import functools
import logging
import os.path as osp
import pickle
from api.communication import api_pb2
import numpy as np
ZERO = 1e-2


logging.basicConfig(
    format="%(levelname)s:%(module)s:%(funcName)s:L%(lineno)d:%(message)s",
    level=logging.DEBUG,
)


@contextlib.contextmanager
def get_file_writer(*args):
    name = "-".join(map(str, args))
    name = name.replace("/", "|")

    filepath = osp.abspath(
        osp.join(osp.dirname(osp.abspath(__file__)), "../log/{}".format(name))
    )

    with open(filepath, "w") as f:
        yield functools.partial(print, file=f)


# ip -> hostname
def parse_hosts():
    f_path = "/etc/hosts"
    result = {}
    with open(f_path) as f:
        for line in map(str.strip, f):
            if not line or line.startswith("#"):
                continue
            ip, hostname = line.split()
            result[ip] = hostname
    return result


# hostname -> ip
def parse_hosts_rev():
    ip_to_host = parse_hosts()
    return {v: k for k, v in ip_to_host.iteritems()}


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)

def vector_to_proto_vector(vec):
    proto_vector = api_pb2.DataFeatureVector()
    proto_vector.values[:] = list(vec)
    return proto_vector

def matrix_to_proto_matrix(mat):
    proto_matrix = api_pb2.DataFeatureMatrix()

    proto_matrix.vectors.extend(list(mat))
    return proto_matrix

def np_array_to_proto_matrix(array):
    proto_vector_set = []
    for i in range(len(array)):
        proto_vector_set.append(vector_to_proto_vector(array[i,:]))
    proto_matrix = matrix_to_proto_matrix(proto_vector_set)
    return proto_matrix


def proto_vector_to_array(proto_vector):
    vec = []
    for item in proto_vector.values:
        vec.append(item)
    return np.array(vec).reshape(-1)

def proto_matrix_to_array(proto_matrix):
    mat = []
    for item in proto_matrix.vectors:
        mat.append(proto_vector_to_array(item))
    return np.array(mat)