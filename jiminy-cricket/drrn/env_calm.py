from jericho import *
from jericho.util import *
import sys

from annotated_env import AnnotatedEnv

class EnvCALM:
    ''' Returns valid actions at each step of the game. '''

    def __init__(self, game_folder_path, seed, step_limit=None, get_valid=False, starting_percentage=0, log_dir=None):
        self.game_folder_path = game_folder_path
        self.env = AnnotatedEnv(game_folder_path, seed=seed, starting_percentage=starting_percentage,
                                 log_dir=log_dir)
        self.seed = seed
        self.steps = 0
        self.step_limit = step_limit
        self.get_valid = get_valid
        self.max_score = self.env.get_starting_score()
        self.end_scores = []


    def step(self, action, write_to_log=False):
        ob, reward, done, info = self.env.step(action, write_to_log=False)

        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait', 'yes', 'no']
        if not done:
            try:
                save = self.env.get_state()
                look, _, _, _ = self.env.step('look')
                info['look'] = look
                self.env.set_state(save)
                inv, _, _, _ = self.env.step('inventory')
                info['inv'] = inv
                self.env.set_state(save)
                if self.get_valid:
                    valid = self.env.get_valid_actions()
                    if len(valid) == 0:
                        valid = ['wait', 'yes', 'no']
                    info['valid'] = valid
            except RuntimeError:
                print('RuntimeError: {}, Done: {}, Info: {}'.format(clean(ob), done, info))
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            if not done: 
                self.env.write_done_to_logs()
            done = True
        self.max_score = max(self.max_score, info['score'])
        if done: self.end_scores.append(info['score'])
        return ob, reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()
        save = self.env.get_state()
        look, _, _, _ = self.env.step('look')
        info['look'] = look
        self.env.set_state(save)
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        self.env.set_state(save)
        valid = ['wait', 'yes', 'no'] 
        info['valid'] = valid
        self.steps = 0
        self.max_score = 0
        return initial_ob, info

    def get_dictionary(self):
        if not self.env:
            self.create()
        return self.env.get_dictionary()

    def get_action_set(self):
        return None

    def get_end_scores(self, last=1):
        last = min(last, len(self.end_scores))
        return sum(self.end_scores[-last:]) / last if last else 0

    def close(self):
        self.env.close()
