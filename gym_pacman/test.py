"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import Mnih2016ActorCriticWithDropout
AC_NN_MODEL = Mnih2016ActorCriticWithDropout
ACTOR_HIDDEN_SIZE=256
CRITIC_HIDDEN_SIZE=256
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--layout", type=str, default="microGrid_superEasy1")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def test(opt):
    torch.manual_seed(123)
    env, num_states, num_actions = create_train_env(opt.layout)#,"{}/video_{}.mp4".format(opt.output_path, opt.layout))
    model = AC_NN_MODEL(num_states, num_actions)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/gym-pacman_{}".format(opt.saved_path,opt.layout)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/gym-pacman_{}".format(opt.saved_path, opt.layout),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    while True:
        if done:
            h_0 = torch.zeros((1, ACTOR_HIDDEN_SIZE), dtype=torch.float)
            c_0 = torch.zeros((1, CRITIC_HIDDEN_SIZE), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()

if __name__ == "__main__":
    opt = get_args()
    test(opt)
