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
    parser.add_argument("--layout", type=str, default="random_mnih2016-24hs")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_games_to_play", type=int, default="3")
    args = parser.parse_args()
    return args


def test(opt):
    torch.manual_seed(123)
    env, num_states, num_actions = create_train_env(opt.layout)#,"{}/video_{}.mp4".format(opt.output_path, opt.layout))
    model = AC_NN_MODEL(num_states, num_actions)
    saved_model = "{}/gym-pacman_{}".format(opt.saved_path, opt.layout)
    print("Loading saved model: {}".format(saved_model))
    if not os.path.isfile(saved_model):
        try:
            import urllib.request
            print('File not found, downloading saved model...')
            url = 'https://github.com/LecJackS/saved_models/blob/master/gym_pacman/gym-pacman_random_mnih2016-24hs?raw=true'
            file_name = "gym-pacman_random_mnih2016-24hs"
            urllib.request.urlretrieve(url, '{}/{}'.format(opt.saved_path, file_name))
            print('Download done.')
        except:
            print("Something wrong happened, couldn't download model")

    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/gym-pacman_{}".format(opt.saved_path, opt.layout)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/gym-pacman_{}".format(opt.saved_path, opt.layout)))
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    game_count = 0
    while game_count <= opt.num_games_to_play:
        if done:
            h_0 = torch.zeros((1, ACTOR_HIDDEN_SIZE), dtype=torch.float)
            c_0 = torch.zeros((1, CRITIC_HIDDEN_SIZE), dtype=torch.float)
            env.reset()
            game_count+=1
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
