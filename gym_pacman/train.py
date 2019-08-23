import os
# To NOT use OpenMP threads within numpy processes
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import Mnih2016ActorCriticWithDropout, SimpleActorCriticWithDropout
#AC_NN_MODEL = Mnih2016ActorCriticWithDropout
AC_NN_MODEL = SimpleActorCriticWithDropout

from src.optimizer import GlobalRMSprop, GlobalAdam
from src.process import local_train, local_test
# For the async policies updates
import torch.multiprocessing as _mp
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--layout", type=str, default='atari')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_processes_to_render", type=int, default=1, help="Renders to a window a color NN input")
    parser.add_argument("--load_previous_weights", type=bool, default=True,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--record", type=bool, default=False,
                        help="Save snapshot from each input frame on ./snaps")
    args = parser.parse_args()
    return args

def train(opt):
    torch.manual_seed(42)
    # Prepare log directory
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    # Prepare saved models directory
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    # Prepare multiprocessing
    mp = _mp.get_context("spawn")
    # Create new training environment just to get number
    # of inputs and outputs to neural network
    _, num_states, num_actions = create_train_env(opt.layout)
    # Create Neural Network model
    global_model = AC_NN_MODEL(num_states, num_actions)
    if opt.use_gpu:
        global_model.cuda()
    # Share memory with processes for optimization later on
    global_model.share_memory()
    # Load trained agent weights
    if opt.load_previous_weights:
        file_ = "{}/gym-pacman_{}".format(opt.saved_path, opt.layout)
        if os.path.isfile(file_):
            print("Loading previous weights for %s..." %opt.layout, end=" ")
            # global_model.load_state_dict(torch.load(file_))
            pretrained_dict = torch.load(file_)
            global_model_dict = global_model.state_dict()
            # 1. filter out unnecessary keys (if trained with different model)
            # for k, v in pretrained_dict.items():
            #     if k in global_model_dict:
            #         print("loading and freezing key: {}".format(k))
            #         #getattr(global_model,k).requires_grad = False
            #     else:
            #         print("key {} do not exist".format(k))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in global_model_dict}
            
            # 2. overwrite entries in the existing state dict
            global_model_dict.update(pretrained_dict) 
            # 3. load the new state dict (only existing keys)
            global_model.load_state_dict(pretrained_dict, strict=False)
            print("Done.")
        else:
            print("Can't load any previous weights for %s! Starting from scratch..." %opt.layout)
    # Define optimizer with shared weights. See 'optimizer.py'
    optimizer = GlobalRMSprop(global_model.parameters(), lr=opt.lr)
    # Create async processes
    processes = []
    for index in range(opt.num_processes):
        # Multiprocessing async agents
        if index == 0:
            # Save weights to file only with this process
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))

        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer))
        process.start()
        processes.append(process)
    # Local test simulation (creates another model = more memory used)
    #process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
    #process.start()
    #processes.append(process)

    for process in processes:
        process.join()

if __name__ == "__main__":
    opt = get_args()
    train(opt)
