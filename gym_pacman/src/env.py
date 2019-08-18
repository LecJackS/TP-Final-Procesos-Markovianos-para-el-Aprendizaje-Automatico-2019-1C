import gym
import gym_pacman

from gym.spaces import Box
from gym import Wrapper

import cv2
import numpy as np
import subprocess as sp
import torch
from torchvision import transforms
from torchvision.utils import save_image

class Monitor:
    def __init__(self, width, height, saved_path):
        # conda install ffmpeg
        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())
        #print(len(image_array))

def process_frame(frame):
    if frame is not None:
        # frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # resize to 84x84px; normalize color num to [0,1)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))

class SameReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(SameReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # Process frame to 84x84px grayscale 
        state = process_frame(state)
        #print("info:", info)
        reward = info["score"] #/ 500. #/1000.

        #print("\naction:",action,"  reward:", reward)
        #print(info, "\n")
        return state, reward, done, info

    def reset(self):
        return process_frame(self.env.reset())

class UnitReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(UnitReward, self).__init__(env)
        #self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # Process frame to 84x84px grayscale 
        #print("info:", info)
        #reward = info["score"]
        if reward > 0:
            reward = 1.
        elif reward ==0:
            reward = 0.
        else:
            #reward < 0
            reward = -1.
        #print("\naction:",action,"  reward:", reward)
        #print(info, "\n")
        return state, reward, done, info

    #def reset(self):
    #    return process_frame(self.env.reset())
class UnitReward_cs188x(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(UnitReward_cs188x, self).__init__(env)
        #self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
         
        #print("info:", info)
        if reward > 0:
            reward = 1.
        else:
            reward = 0.
        #print("\naction:",action,"  reward:", reward)
        #print(info, "\n")
        return state, reward, done, info

    #def reset(self):
    #    return process_frame(self.env.reset())
class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        if self.monitor:
            self.monitor.record(state)
        # Process frame to 84x84px grayscale 
        state = process_frame(state)
        #print("info:", info)
        reward += (info["score"] - self.curr_score) / 10.
        #print("reward:",reward)
        self.curr_score = info["score"]
        # for mario
        #if done:
        #    if info["flag_get"]:
        #        reward += 50
        #    else:
        #        reward -= 50
        # for pacman gym
        #if done:
        #    if info["max_ep"]:
        #        # max episode lenght reached
        #        reward -= 500
        #        print("penalty! info:", info["max_ep"])
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())
    
class NoSkipFrame(Wrapper):
    """ Neural network four frame input:
        [T,
         T,
         T,
         T] """
    def __init__(self, env, skip=4):
        super(NoSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        states = []
        #done = self.done
        state, reward, done, info = self.env.step(action)
        # we need a four channel input
        copies = 4 # instead of copies, do transformations!
        for i in range(copies):
            states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info
    
    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)

class NoSkipFrameFourRotations(Wrapper):
    """ Neural network four frame input:
        [T,
         rot90(T),
         rot180(T),
         rot270(T)] """
    def __init__(self, env, skip=4):
        super(NoSkipFrameFourRotations, self).__init__(env)
        #self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        #self.skip = skip

    def preproc_state(self, state):
        size = (84,84)
        #https://cs231n.github.io/neural-networks-2/#datapre
        mean = (0,0,0) #by channel
        std = (1,1,1)
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        # For whitening
        # compute the covariance
        #X = state.reshape(-1, 3*84*84)
        #print(X.size())
        #print(X.data)
        #cov = np.cov(X.numpy(), rowvar=False)   # cov is (N, N)
        #print(np.shape(cov))
        # singular value decomposition
        #U, S, V = torch.svd(torch.from_numpy(cov))     # U is (N, N), S is (N,1) V is (N,N)
        # build the ZCA matrix which is (N,N)
        #epsilon = 1e-5
        #zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
        transform = transforms.Compose([
                        #transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        #transforms.LinearTransformation(zca_matrix),
                        transforms.ToPILImage(),
                        transforms.Resize(size),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize([0], [1])
                    ])
        return transform(state)[None,:,:,:]

    def step(self, action):
        states = []
        #done = self.done
        state, reward, done, info = self.env.step(action)

        # we need a four channel input 
        # We'll get all four rotations of current fram
        # TODO: mirroring horizontal and diagonal (4 more)
        # Process frame to 84x84px grayscale, to Tensor
        state = torch.from_numpy(state).permute(2,0,1).float()
        states.append(self.preproc_state(state))
        times = 3 # transformations!
        for t in range(1, times+1):
            # rotate +90 degrees each frame 
            states.append(self.preproc_state(torch.rot90(state, t, dims=(1, 2))))
        #states = np.concatenate(states, 0)[None, :, :, :]
        states = torch.cat(states, dim=1)
        return states, reward, done, info
    
    def reset(self):
        state  = self.env.reset()
        state = torch.from_numpy(state).permute(2,0,1).float()
        states = []
        states.append(self.preproc_state(state))
        times = 3
        for t in range(1, times+1):
            # rotate +90 degrees each frame 
            states.append(self.preproc_state(torch.rot90(state, t, dims=(1, 2))))
        #states = np.concatenate(states, 0)[None, :, :, :]
        states = torch.cat(states, dim=1)
        return states

class NoSkipFrameColourFourRotations(Wrapper):
    """ Neural network four frame input:
        [T,
         rot90(T),
         rot180(T),
         rot270(T)] """
    def init(self, env, skip=4):
        super(NoSkipFrameColourFourRotations, self).init(env)
        #self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        #self.skip = skip
        size = (84,84)
        #https://cs231n.github.io/neural-networks-2/#datapre
        mean = [0,0,0] #by channel
        std = [1,1,1]
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        # TODO: whitening
        self.transform_gray = transforms.Compose([
                        #transforms.ToTensor(),
                        transforms.Normalize([0,0,0], [3,3,3]),
                        transforms.ToPILImage(),
                        #transforms.ColorJitter(brightness=[2,2], contrast=[2,2]),
                        transforms.Resize(size, interpolation=2),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize([0],[1])
                    ])
        self.transform_colour = transforms.Compose([
                        #transforms.ToTensor(),
                        transforms.Normalize([0,0,0], [3,3,3]),
                        transforms.ToPILImage(),
                        #transforms.ColorJitter(brightness=[2,2], contrast=[2,2]),
                        transforms.Resize(size, interpolation=2),
                        #transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize([0,0,0], [1,1,1])
                    ])
    def preproc_state(self, state, cha):
        if cha==3:
            return self.transform_gray(state)[None,:,:,:]
        else:
            # Colour channels 0, 1 and 2
            tran_state = self.transform_colour(state)[cha,:,:]
            return tran_state[None,None,:,:]

    def step(self, action):
        states = []
        #done = self.done
        state, reward, done, info = self.env.step(action)

        # we need a four channel input 
        # We'll get all four rotations of current fram
        # TODO: mirroring horizontal and diagonal (4 more)
        # Process frame to 84x84px grayscale, to Tensor
        state = torch.from_numpy(state).permute(2,0,1).float()
        states.append(self.preproc_state(state, cha=0))
        times = 3 # transformations!
        for t in range(1, times+1):
            # rotate +90 degrees each frame 
            states.append(self.preproc_state(torch.rot90(state, t, dims=(1, 2)), cha=t))
        #states = np.concatenate(states, 0)[None, :, :, :]
        states = torch.cat(states, dim=1)
        return states, reward, done, info
    
    def reset(self):
        state  = self.env.reset()
        state = torch.from_numpy(state).permute(2,0,1).float()
        states = []
        states.append(self.preproc_state(state, cha=0))
        times = 3
        for t in range(1, times+1):
            # rotate +90 degrees each frame 
            states.append(self.preproc_state(torch.rot90(state, t, dims=(1, 2)), cha=t))
        #states = np.concatenate(states, 0)[None, :, :, :]
        states = torch.cat(states, dim=1)
        return states

class MinimSkipFrame(Wrapper):
    """ Neural network four frame input:
        [T,
         T,
         T+1,
         T+1] """
    def __init__(self, env, skip=4):
        super(MinimSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        #done = self.done
        state, reward, done, info = self.env.step(action)
        states.append(state)
        states.append(state) # we need a four channel input
        total_reward += reward
        # we pass states in mini groups of 2
        if not done:
            state, reward, done, info = self.env.step(action)
            states.append(state)
            states.append(state)
            total_reward += reward
        else:
            states.append(state)
            states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        print(states)
        print(states.astype(np.float32))
        return states.astype(np.float32), reward, done, info
    
    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)

class SimpleSkipFrame(Wrapper):
    """ Neural network four frame input:
        [T,
         T+1,
         T+2,
         T+3] """
    def __init__(self, env, skip=4):
        super(SimpleSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def preproc_state(self, state):
        size = (84,84)
        mean = (0,0,0) #by channel
        std = (1,1,1)
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        transforms.ToPILImage(),
                        transforms.Resize(size),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize([0], [1])
                    ])
        return transform(state)[None,:,:,:]

    def step(self, action):
        total_reward = 0
        states = []
        #done = self.done
        state, reward, done, info = self.env.step(action)
        states.append(self.preproc_state(state))
        total_reward += reward
        # we pass states in mini groups of skip=4
        for i in range(self.skip-1):
            if not done:
                state, reward, done, info = self.env.step(action)
                # state: [H,W,Channel]
                total_reward += reward
                states.append(self.preproc_state(state))
            else:
                states.append(self.preproc_state(state))
        #states = np.concatenate(states, 0)[None, :, :, :]
        #return states.astype(np.float32), total_reward, done, info
        #states = np.concatenate([state for _ in range(self.skip)], axis=1)

        states = torch.cat(states, dim=1)

        return states, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        p_state = self.preproc_state(state)
        #states = np.concatenate([p_state for _ in range(self.skip)], axis=1)
        states = torch.cat([p_state for _ in range(self.skip)], dim=1)
        return states#.astype(np.float32)
    
class DQNSkipFrame(Wrapper):
    """ # https://github.com/openai/gym/issues/275
        # (tried to be) implemented by jack 
        Neural network four frame input:
        [max(T-1,  T),
         max(T+3,  T+4),
         max(T+7,  T+8),
         max(T+11, T+12)] """
    def __init__(self, env, skip=4):
        super(DQNSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def preproc_state(self, state):
        size = (84,84)
        mean = (0,0,0) #by channel
        std = (1,1,1)
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        transforms.ToPILImage(),
                        transforms.Resize(size),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize([0], [1])
                    ])
        return transform(state)[None,:,:,:]

    def step(self, action):
        total_reward = 0
        states = []

        # we pass states in mini groups of skip=4
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if not done:
                state2, reward, done, info = self.env.step(action)
                total_reward += reward
                # element wise max
                state = np.maximum(state, state2)
            states.append(self.preproc_state(state))
            for j in range(0):
                # dummy steps, but we keep track of reward
                _, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break #this dummy loop
        #total_reward /= 12.
        states = torch.cat(states, dim=1)
        return states, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        p_state = self.preproc_state(state)
        states = torch.cat([p_state for _ in range(self.skip)], dim=1)
        return states

class DQNColourSkipFrame(Wrapper):
    """ # https://github.com/openai/gym/issues/275
        # (tried to be) implemented by jack 
        Neural network four frame input:
        [max(T-1,  T),
         max(T+3,  T+4),
         max(T+7,  T+8),
         max(T+11, T+12)] """
    def __init__(self, env, skip=4):
        super(DQNColourSkipFrame, self).__init__(env)
        #self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip
        # define transformation
        size = (84,84)
        mean = (0,0,0) #by channel
        std = (1,1,1)
        self.transform_gray = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        transforms.ToPILImage(),
                        transforms.Resize(size),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize([0], [1])
                    ])
        self.transform_colour = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        transforms.ToPILImage(),
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])

    def preproc_state(self, state, cha):
        # Color ch1
        if cha==3:
            # Grayscale state
            #print('canal gray')
            return self.transform_gray(state)[None,:,:,:]
        else:
            # loop over colour channels 0,1,2
            #print('canal de color')
            tran_state = self.transform_colour(state)[cha,:,:]
            return tran_state[None,None,:,:]

    def step(self, action):
        total_reward = 0
        states = []

        # we pass states in mini groups of skip=4
        for idx in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if not done:
                state2, reward, done, info = self.env.step(action)
                total_reward += reward
                # element wise max
                state = np.maximum(state, state2)
            states.append(self.preproc_state(state, idx))
            # for j in range(0):
            #     # dummy steps, but we keep track of reward
            #     _, reward, done, _ = self.env.step(action)
            #     total_reward += reward
            #     if done:
            #         break #this dummy loop
        #total_reward /= 12.
        states = torch.cat(states, dim=1)
        return states, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        #p_state = self.preproc_state(state)
        states = torch.cat([self.preproc_state(state, idx) for idx in range(self.skip)], dim=1)

        return states

class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        # skips one frame, and pass the next four to the NN
        state, reward, done, info = self.env.step(action)
        # we pass states in mini groups of skip=4
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)

def create_train_env(layout, output_path=None, index=None):
    if layout=='atari':
        #print("es atari")
        env, nInNN, nOutNN = create_train_env_atari(layout, output_path, index)
    elif layout=='cs188x':
        #print("es cs188x")
        env, nInNN, nOutNN = create_train_env_cs188x(layout, output_path, index)
    else:
        print('layout not understood, using cs188x')
        env, nInNN, nOutNN = create_train_env_cs188x(layout, output_path, index)
    return env, nInNN, nOutNN


def create_train_env_atari(layout, output_path=None, index=None):
    if index is not None:
        print("Process {} - Create train env for {}".format(index, layout))
    else:
        print("Getting number of NN inputs/outputs for", layout)
    env = gym.make('MsPacman-v0')
    #output_path = 'output/un_video.mp4' # Beware! Can freeze training for some reason.
    if output_path:
        #monitor = Monitor(256, 240, output_path)
        monitor = Monitor(150, 150, output_path)
    else:
        monitor = None

    # Pacman Actions https://github.com/Kautenja/nes-py/wiki/Wrap
    actions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'NOOP']
    # Wraps around env:
    #env = CustomReward(env, monitor)
    #env = SameReward(env, monitor)
    #env = CustomSkipFrame(env)
    env = DQNColourSkipFrame(env)
    # positives rewards are 1, others 0
    env = UnitReward(env)
    # Four times same frame input, no skip
    #env = NoSkipFrame(env)
    #env = SimpleSkipFrame(env)
    # Four rotations of same frame input, no skip
    #env = NoSkipFrameFourRotations(env)
    #return env, env.observation_space.shape[0], len(actions)
    num_inputs_to_nn = 4#x84x84
    num_outputs_from_nn = len(actions)
    return env, num_inputs_to_nn, num_outputs_from_nn

def create_train_env_cs188x(layout, output_path=None, index=None):
    if index is not None:
        print("Process {} - Create train env for {}".format(index, layout))
    else:
        print("Getting number of NN inputs/outputs for layout", layout)
    env = gym.make('BerkeleyPacman-v0')
    #output_path = 'output/un_video.mp4' # Beware! Can freeze training for some reason.
    if output_path:
        #monitor = Monitor(256, 240, output_path)
        monitor = Monitor(150, 150, output_path)
    else:
        monitor = None

    # Pacman Actions https://github.com/Kautenja/nes-py/wiki/Wrap
    actions = ['North', 'South', 'East', 'West', 'Stop']
    # Wraps around env:
    #env = CustomReward(env, monitor)
    env = UnitReward_cs188x(env, monitor)
    #env = CustomSkipFrame(env)
    #env = DQNSkipFrame(env)
    # Four times same frame input, no skip
    #env = NoSkipFrame(env)
    # Four rotations of same frame input, no skip
    env = NoSkipFrameColourFourRotations(env)
    #return env, env.observation_space.shape[0], len(actions)
    num_inputs_to_nn = 4#x84x84
    num_outputs_from_nn = len(actions)
    return env, num_inputs_to_nn, num_outputs_from_nn
