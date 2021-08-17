#!/usr/bin/env bash
protoc -I=. --python_out=. ./zebra.proto


# Ansible start slaves:
#./grid: create beforehand, used to store the slave ip addresses
ansible-playbook -i ./grid luanch.yml

#then copy ./grid into ./hosts, /etc/hosts in myMAC

#update master
ssh-copy-id  luping@master
scp -r /Users/ourokutaira/Desktop/edge/implementation/zebra luping@master:/home/luping/work/
scp /etc/hosts luping@master:/etc/hosts

#update slaves #bash
for HOST in master slave-0 slave-1 slave-2; do
    ssh-copy-id  luping@$HOST
done

for HOST in master slave-0 slave-1 slave-2; do
    scp /etc/hosts luping@$HOST:/etc/hosts
done

# for HOST in master slave-0 slave-1 slave-2; do
#     scp -r /Users/ourokutaira/Desktop/edge/implementation/zebra luping@$HOST:/home/luping/work/
# done

ansible-playbook -i ./hosts_m ec2-run.yml --tags 'clonepull'


sudo snap install [pycharm-professional] --classic

export AWS_ACCESS_KEY_ID=AKIAJFBPSLJG23TORCHQ:$AWS_ACCESS_KEY_ID