#! /usr/bin/env bash

DURATION=6h
PROCESSES=10

echo "[+] Beginning experiment 1"
##
# Experiment 1
##
timeout $DURATION python main.py --env-name "Experiment-1" --reward 2  --autoshroom --trace-size 2000 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 2 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes $PROCESSES --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01

echo "[+] Beginning experiment 2"
##
# Experiment 2
##
timeout $DURATION python main.py --env-name "Experiment-2" --reward 2  --autoshroom --trace-size 1 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 0 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes $PROCESSES --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01


# echo "[+] Beginning experiment 3"
##
# Experiment 3
##
# timeout $DURATION python main.py --env-name "Experiment-3" --reward 1  --autoshroom --trace-size 2000 --rtrace-length 1 --level 4 --use-linear-lr-decay --recurrent-policy --model 2 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes $PROCESSES --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01


# echo "[+] Beginning experiment 4"
##
# Experiment 4
##
# timeout $DURATION python main.py --env-name "Experiment-4" --reward 1  --autoshroom --trace-size 1 --rtrace-length 1 --level 4 --use-linear-lr-decay --recurrent-policy --model 0 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes $PROCESSES --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01

echo "[+] Beginning experiment 5"
##
# Experiment 5
##
timeout $DURATION python main.py --env-name "Experiment-5" --reward 2  --autoshroom --trace-size 4000 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 2 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes $PROCESSES --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01

echo "[+] Beginning experiment 6"

##
# Experiment 6
##
timeout $DURATION python main.py --env-name "Experiment-6" --reward 2   --trace-size 4000 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 2 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes $PROCESSES --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01

echo "[+] Beginning experiment 7"

#
# Experiment 7
##
timeout $DURATION python main.py --env-name "Experiment-7" --reward 2   --trace-size 2000 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 2 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes $PROCESSES --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01
