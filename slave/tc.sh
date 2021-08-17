#!/usr/bin/env bash

# this is somewhat limited due to the fact minor number is of 16bits.
# so we divide tenants, and flows into 8bits
# high8..... : low8.....
# tenant_i   : flow_j
# this indicates we can only have 255 tenants and 255 flows for each tenant.
# XXX CAUTION!
# because for i = 0, j mustn't be 0
# we simply start i and j from 1 instead of 0.


## NOTA.BENE we are not limiting the incoming bandwidth.

cd "$(dirname "${BASH_SOURCE[0]}")"

# ETH=lo
# HOSTNAME=$(hostname)

# if [[ $HOSTNAME == slave-* || $HOSTNAME == kiwi ]]; then
#     ETH=eth0
# fi

ETH=eth0

reset () {
    # egress
    sudo tc qdisc del dev $ETH root >/dev/null 2>&1
    sudo tc qdisc add dev $ETH root handle 1: htb  >/dev/null 2>&1
    :
}


show () {
    sudo tc -s qdisc show dev $ETH
    sudo tc -s class show dev $ETH
    sudo tc -s filter show dev $ETH
}


init-sender () {
    # always in hex form
    classid=${1:-f0}
    sport=${2:-5002}
    dst=${3:-127.0.0.1}
    dport=${4:-5001}

    # it's bits per second
    # must be >= 8, tc doesn't accept rate less than 1byte-per-sec
    rate=${5:-6}

    # sudo tc -s class show dev $ETH | grep -q "class htb 1:$classid"
    # # NOTE
    # # there is a TOCTOU problem, because after checking the existence of the rule, `update-sender' might be
    # # invoked in between.
    # if [[ $? -ne 0 ]]; then
    #     # without unit after ${rate}, it means that ${rate} bits per second
    #     sudo tc class add dev $ETH parent 1:0 classid 1:$classid htb rate ${rate} ceil ${rate} &>/dev/null
    # fi


    # we don't care if `update-sender' is invoked first or not.  if it's invoked first, we will fail to update, which is right.
    sudo tc class add dev $ETH parent 1:0 classid 1:$classid htb rate ${rate} ceil ${rate} &>/dev/null

    sudo tc filter add dev $ETH protocol ip parent 1:0 u32 \
         match ip sport $sport 0xffff \
         match ip dst $dst \
         match ip dport $dport 0xffff \
         flowid 1:$classid

    exit

    # only for testing iperf
    # 41943040 for 40Mbits-per-second
    sudo tc class add dev eth0 parent 1:0 classid 1:f0 htb rate 41943040 ceil 41943040
    sudo tc filter add dev eth0 protocol ip parent 1:0 u32 match ip dport 5001 0xffff flowid 1:f0
}


update-sender () {
    classid=${1:-f0}
    rate=${2:-15}

    sudo tc class replace dev $ETH parent 1:0 classid 1:$classid htb rate ${rate} ceil ${rate}
    exit

    # only for testing iperf
    # 32Mbits-per-second
    sudo tc class replace dev eth0 parent 1:0 classid 1:f0 htb rate 33554432 ceil 33554432
}


iperf-server () {
    # iperf -s -i 1
    iperf -s -i 1 -B master
}

iperf-client () {
    iperf -n 1000M -c master

    # iperf -n 1000M -c $(hostname)
    # iperf -n 1000M -c slave -i 1
}


eval "$@"
