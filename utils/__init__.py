#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import codecs
import contextlib
import functools
import logging
import os.path as osp
import pickle

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
