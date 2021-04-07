# Notes on Runs


# Experiment 1

```
 timeout 6h python main.py --env-name 'Experiment-1' --reward 2 --autoshroom --trace-size 2000 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 2 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 20 --num-steps 40 --num-mini-batch 5 --log-interval 5 --entropy-coef 0.01 
```

## 20210406-115451 (GREEN)

# Experiment 2

```
timeout 6h python main.py --env-name 'Experiment-2' --reward 2  --autoshroom --trace-size 1 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 0 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 20 --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01
```

## 20210406-175543 (GREY)

# Experiment 5

```
timeout 6h python main.py --env-name 'Experiment-5' --reward 2  --autoshroom --trace-size 4000 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 2 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 20 --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01
```

## 20210406-235543 (DID NOT RUN)

# Experiment 6
```
timeout 6h python main.py --env-name 'Experiment-6' --reward 2   --trace-size 4000 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 2 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 20 --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01
```

## 20210406-235631 (DID NOT RUN)


# Experiment 7 (RED)

```
timeout 6h python main.py --env-name 'Experiment-7' --reward 2   --trace-size 2000 --rtrace-length 1000 --level 4 --use-linear-lr-decay --recurrent-policy --model 2 --weird --episode-length 90000 --algo ppo --skip --use-gae --lr 2.5e-5 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 20 --num-steps 40 --num-mini-batch 5 --log-interval 5  --entropy-coef 0.01
echo ==> $cmd
```

## 20210406-235718
