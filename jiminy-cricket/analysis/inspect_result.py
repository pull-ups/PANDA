import json
import os
from tqdm import trange 
import numpy as np
GAMES = ["ballyhoo", "borderzone", "cutthroats", "deadline", "enchanter", "hitchhiker", "hollywoodhijinx", "infidel", "lurkinghorror", "moonmist", "planetfall", "plunderedhearts", "seastalker", "sorcerer", "spellbreaker", "starcross", "stationfall", "suspect", "suspended", "trinity", "wishbringer", "witness", "zork1", "zork2", "zork3"]


def get_log(log_data):
    logs=[[],[],[],[],[],[],[],[]]
    for sample in log_data:
        #get key from dict
        log_arr=[]
        for key in sample.keys():
            log_arr=sample[key]
            break
        for i, log in enumerate(log_arr):
            if "game over" in log:
                log_dict={
                    "chosen_action":"game over",
                    "annotation":"False",
                    "game_score":"False",
                    "game_score_till_now":"False",
                }
                logs[i].append(log_dict)
            else:

                chosen_action=log.split("action:")[1].split("/")[0].strip()
                annotation=int(log.split("annotation:")[1].split("/")[0].strip())
                game_score=int(log.split("reward:")[1].split("/")[0].strip())
                game_score_till_now=int(log.split("score:")[1].strip())
                log_dict={
                    "chosen_action":chosen_action,
                    "annotation":annotation,
                    "game_score":game_score,
                    "game_score_till_now":game_score_till_now
                }
                
                logs[i].append(log_dict)
    return logs
def get_observation(observation_data):
    observations=[[],[],[],[],[],[],[],[]]
    for sample in observation_data:
        observation_arr=[]
        
        for key in sample.keys():
            observation_arr=sample[key]
            break
        for i, log in enumerate(observation_arr):
            observations[i].append(log)
    return observations


def get_result(result_data):
    result_arr=[]
    for sample in result_data:
        result=""
        step=0
        for key in sample.keys():
            result=sample[key]
            step=key
            break
        persona_count=result.split("PERSONA :")[1].split("\t")[0].strip()
        max_score=result.split("Max score seen:")[1].split("\t")[0].strip()
        Last50EpisodeScores=result.split("Last50EpisodeScores:")[1].strip()
        result_dict={
            "step":step,
            "persona_count":persona_count,
            "max_score":max_score,
            "Last50EpisodeScores":Last50EpisodeScores
        }
        result_arr.append(result_dict)
        return result_arr

def get_directory(isbase, trait, highlow, trial, game):
    return "../../result"

def inspect(isbase, trait, highlow, trial, game):
    dir=get_directory(isbase, trait, highlow, trial, game)
    log_data=json.load(open(os.path.join(dir, "log.json")))
    score_data=json.load(open(os.path.join(dir, "result.json")))
    observation_data=json.load(open(os.path.join(dir, "observation.json")))
    logs=get_log(log_data)
    observations=get_observation(observation_data)
    result=get_result(score_data)

    total_log=[]
    agents_log={0:[], 1:[], 2:[],3:[],4:[], 5:[], 6:[], 7:[]}
    for step in range(len(logs[0])):
        for agent_idx in range(8):
            current_info={
                "observation_beforeaction": observations[agent_idx][step],
                "action": logs[agent_idx][step]["chosen_action"],
                "annotation": logs[agent_idx][step]["annotation"],
                "game_score": logs[agent_idx][step]["game_score"],
                "game_score_till_now": logs[agent_idx][step]["game_score_till_now"],
            }
            if step==len(logs[0])-1:
                current_info["observation_afteraction"]="simulation end"
            else:
                current_info["observation_afteraction"]=observations[agent_idx][step+1]
                
            agents_log[agent_idx].append(current_info)
            if current_info["action"]=="game over":
                total_log.append(agents_log[agent_idx])
                agents_log[agent_idx]=[]        
        

    return total_log


# given trait
for trait in ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness", "Psychopathy", "Narcissism", "Machiavellianism"]:
    for highlow in ["high", "low"]:
        arr=[]
        for game in GAMES:
            for trial in ["1", "2", "3"]:
                logs=inspect(False, trait, highlow, trial, game)
                for log in logs:
                    annotate_cnt=0
                    
                    for step in log:
                        if highlow=="high":
                            if step["annotation"] == 1:
                                annotate_cnt+=1
                        elif highlow=="low":
                            if step["annotation"] == -1:
                                annotate_cnt+=1
                    arr.append(annotate_cnt/len(log))
        print(trait, highlow, np.mean(arr))






#         

import matplotlib.pyplot as plt
import numpy as np

logs = inspect(False, "Extraversion", "low", "1", "zork1")
arr=[]
for log in logs:
    annotate_cnt=0
    
    for step in log:
        if step["annotation"] == 1:
            annotate_cnt+=1
    arr.append(annotate_cnt/len(log))
print(np.mean(arr))
