#!/bin/bash

export PYTHONHASHSEED=0
export WEIGHT=2
export THRESH=0.39
export STARTING_PERCENTAGE=0

export GAME=$1
echo "${GAME}"

export trial=1



export PERSONA=$2

python -u drrn/train.py \
  --game_folder_path ./annotated_games/${GAME} \
  --lm_path PATH_TO_ACTION_CANDIDATE_GENERATOR \
  --output_dir ./result/output/${PERSONA}/starting_percentage_${STARTING_PERCENTAGE}/${trial}/logs_persona/${GAME}/${PERSONA}/ \
  --seed 1 \
  --log_freq 100 \
  --num_envs 8 \
  --batch_size 64 \
  --lm_top_k 40 \
  --max_steps 15000 \
  --cclm_path DIR_TO_PERSONALITY_CLASSIFIER \
  --condition_weight ${WEIGHT} \
  --cond_threshold ${THRESH} \
  --policy_shaping \
  --persona ${PERSONA} \
  --starting_percentage ${STARTING_PERCENTAGE} 


