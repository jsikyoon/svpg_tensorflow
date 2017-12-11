#!/bin/bash

ps_num=0
worker_num=0
# independently learning or not
independent_flag=1

for i in `eval echo {0..$ps_num}`
do
  python gym_svpg.py --ps_hosts_num=$ps_num --worker_hosts_num=$worker_num --job_name=ps --task_index=$i --independent_flag=$independent_flag &
done

for i in `eval echo {0..$worker_num}`
do
  python gym_svpg.py --ps_hosts_num=$ps_num --worker_hosts_num=$worker_num --job_name=worker --task_index=$i --independent_flag=$independent_flag &
done
