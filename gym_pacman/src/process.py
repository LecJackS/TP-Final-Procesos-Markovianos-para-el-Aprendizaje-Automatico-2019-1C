import numpy as np
import queue
import torch
from src.env import create_train_env

from src.model import Mnih2016ActorCriticWithDropout, SimpleActorCriticWithDropout
#AC_NN_MODEL = Mnih2016ActorCriticWithDropout
AC_NN_MODEL = SimpleActorCriticWithDropout
ACTOR_HIDDEN_SIZE=256
CRITIC_HIDDEN_SIZE=256

import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
#import torchvision.transforms as TV
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from collections import deque

import timeit

def preproc_state(np_state):
    #np_img = np.transpose(np_img[0], (2,0,1)) # [C, H, W]
    #size = (3,84,84)
    np_img = np_state[0]
    pil_img = Image.fromarray(np_img.astype('uint8'))

    t_resize = transforms.Resize((84,84))
    pil_img = t_resize(pil_img) # [C, 84, 84]
    t_grayscale = transforms.Grayscale()
    pil_img = t_grayscale(pil_img)
    np_img = np.array(pil_img)
    state = torch.from_numpy(np_img)[None,None,:,:] # [Batch, C, 84, 84]
    return state.repeat(1,4,1,1).float() #repeat 4 times the frame

def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(42 + index)
    if save:
        start_time = timeit.default_timer()
    if index==0:
        # Path for tensorboard log
        process_log_path = "{}/process-{}".format(opt.log_path, index)
        writer = SummaryWriter(process_log_path)#, max_queue=1000, flush_secs=10)
    # Creates training environment for this particular process
    env, num_states, num_actions = create_train_env(opt.layout, index=index)
    # local_model keeps local weights for each async process
    local_model = AC_NN_MODEL(num_states, num_actions)
    if opt.use_gpu:
        local_model.cuda()
    # Tell the model we are going to use it for training
    local_model.train()
    # env.reset and get first state
    if True:#opt.layout == 'atari':
        # Reshape image from HxWxC -to-> CxHxW
        state = env.reset()
        #state = preproc_state(np_state)
    else:
        state = torch.from_numpy(env.reset())
    if opt.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    curr_episode = 0
    # Keep track of min/max Gt and Actor Loss to clamp Critic and Actor
    max_Gt = 5.
    max_AL = 1.
    if index == 0:
        interval = 100
        #reward_hist = np.zeros(interval)
        reward_hist = deque(maxlen=100)
        #queue_rewards = queue.Queue(maxsize=interval)
        record_tag = False
    while True:
        if save:
            # Save trained model at save_interval
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/gym-pacman_{}".format(opt.saved_path, opt.layout))
        if curr_episode%10==0:
            print("Process {}. Episode {}   ".format(index, curr_episode))
        curr_episode += 1
        episode_reward = 0
        
        # Synchronize thread-specific parameters theta'=theta and theta'_v=theta_v
        # (copy global params to local params (after every episode))
        local_model.load_state_dict(global_model.state_dict(), strict=True)
        # Follow gradients only after 'done' (end of episode)
        if done:
            h_0 = torch.zeros((1, ACTOR_HIDDEN_SIZE), dtype=torch.float)
            c_0 = torch.zeros((1, CRITIC_HIDDEN_SIZE), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []
        # Local steps
        for _ in range(opt.num_local_steps):
            curr_step += 1
            # Decay max_Gt over time to adjust to present Gt scale
            max_Gt = max_Gt * 0.9999
            # Model prediction from state. Returns two functions:
            # * Action prediction (Policy function) -> logits (array with every action-value)
            # * Value prediction (Value function)   -> value (single value state-value)
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            
            # Simple estimation: between(-1,1)
            #value = value.clamp(min_Gt, max_Gt)
            # Softmax over action-values
            policy = F.softmax(logits, dim=1)
            # Log-softmax over action-values, to get the entropy of the policy
            log_policy = F.log_softmax(logits, dim=1)
            #print('0. policy----------: \n', policy)
            #print('1. logits----------: \n', logits)
            #print('2. log_policy------: \n', log_policy)
            # Entropy acts as exploration rate
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            # From Async Methods for Deep RL:
            """ We also found that adding the entropy of the policy π to the
                objective function improved exploration by discouraging
                premature convergence to suboptimal deterministic poli-
                cies. This technique was originally proposed by (Williams
                & Peng, 1991), who found that it was particularly help-
                ful on tasks requiring hierarchical behavior."""
            # We sample one action given the policy probabilities
            m = Categorical(policy)
            action = m.sample().item()
            # Perform action_t according to policy pi
            # Receive reward r_t and new state s_t+1
            state, reward, done, _ = env.step(action)
            reward = reward / max_Gt
            episode_reward += reward
            if opt.record and index==0:
                #save animation for each four-frame input
                save_image(state.permute(1,0,2,3),
                           filename='./snaps/process{}-{}.png'.format(index, curr_step),
                           nrow=1)#,normalize=True)

            # Preprocess state:
            #state = preproc_state(np_state)
            # state to tensor
            #state = torch.from_numpy(state)
            # Render as seen by NN, but with colors 
            if index < opt.num_processes_to_render:
                env.render(mode = 'human')
            
            if opt.use_gpu:
                state = state.cuda()
            # If last global step, reset episode
            if curr_step > opt.num_global_steps:
                done = True
            if done:
                curr_step = 0
                state = env.reset()
                #state = preproc_state(np_state)
                print("Process {:2.0f}. acumR: {}     ".format(index, episode_reward))

                if opt.use_gpu:
                    state = state.cuda()
            # Save state-value, log-policy, reward and entropy of
            # every state we visit, to gradient-descent later
            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                # All local steps done.
                break

        # Save history every n episodes as statistics (just from one process)
        if index==0: 
            reward_hist.append(episode_reward)
            if True:#hist_idx==sample_size-1:
                r_mean   = np.mean(reward_hist)
                r_median = np.median(reward_hist)
                r_std    = np.std(reward_hist)
                stand_median = (r_median - r_mean) / (r_std + 1e-9)
                writer.add_scalar("Process_{}/Last100_mean".format(index), r_mean, curr_episode)
                writer.add_scalar("Process_{}/Last100_median".format(index), r_median, curr_episode)
                writer.add_scalar("Process_{}/Last100_std".format(index), r_std, curr_episode)
                writer.add_scalar("Process_{}/Last100_stand_median".format(index), stand_median, curr_episode)
        # Normalize Rewards
        #mean_rewards = np.mean(rewards)
        #std_rewards  = np.std(rewards)
        #rewards = (rewards - mean_rewards) / (std_rewards + 1e-9)
        # Initialize R/G_t: Discounted reward over local steps
        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)
            # Simple state-value estimation: between(-30, 30)
            #R = R.clamp(min_Gt, max_Gt)
        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        # Gradiend descent over minibatch of local steps, from last to first step
        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            # Generalized Advantage Estimator (GAE)
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            # Accumulate discounted reward
            R = reward +  opt.gamma * R

            # For normalization/clamp
            max_Gt = max(max_Gt, abs(R.detach().item()))
            # Accumulate gradients wrt parameters theta'
            #print('log_policy:', log_policy)
            #print('gae:', gae)
            actor_loss = actor_loss + log_policy * gae
            #print('actor_loss:', actor_loss)
            # For normalization/clamp
            max_AL = max(max_AL, abs(actor_loss.detach().item()))
            # Accumulate gradients wrt parameters theta'_v
            critic_loss = critic_loss + ((R - value) ** 2) / 2.
            entropy_loss = entropy_loss + entropy
        # Update and keep track of (min_Gt, max_Gt) for Critic range
        # as an exponential cummulative average

        #max_Gt = 0.495*max_Gt + 0.505*(max(1, R.item())-max_Gt)/(curr_episode)
        
        # Total process' loss
        #print('actor_loss',actor_loss)
        #print('critic_loss',critic_loss)
        #print('entropy_loss',opt.beta * entropy_loss)
        # Make sure that max update is about 1.0 (lr * critic_loss)<1,
        # so updates to weights are not excesive.
        # ie: lr=1e-4; max critic_loss == 1/1e-4 = 1e4 = 10000
        #     lr*loss == 0.0001*10000 == 1 (close to 1)
        critic_loss = critic_loss
        # Normalize actor loss
        actor_loss =  actor_loss#max_AL # 3.*actor_loss funca bien con critic_loss sin modificar
        #print('actor_loss final:', actor_loss)
        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        # Saving logs for TensorBoard
        if index==0:
            writer.add_scalar("Process_{}/Total_Loss".format(index), total_loss, curr_episode)
            writer.add_scalar("Process_{}/actor_Loss".format(index), -actor_loss, curr_episode)
            writer.add_scalar("Process_{}/critic_Loss".format(index), critic_loss, curr_episode)
            writer.add_scalar("Process_{}/entropy_Loss".format(index), -opt.beta*entropy_loss, curr_episode)
            writer.add_scalar("Process_{}/Acum_Reward".format(index), episode_reward, curr_episode)
            writer.add_scalar("Process_{}/max_Gt".format(index), max_Gt, curr_episode)
            writer.add_scalar("Process_{}/max_AL".format(index), max_Gt, curr_episode)
            writer.add_scalar("Process_{}/Gt".format(index), R, curr_episode)
            #writer.add_scalar("actor_{}/Loss".format(index), -actor_loss, curr_episode)
            #writer.add_scalar("critic_{}/Loss".format(index), critic_loss, curr_episode)
            #writer.add_scalar("entropyxbeta_{}/Loss".format(index), opt.beta * entropy_loss, curr_episode)
        # Gradientes a cero
        optimizer.zero_grad()
        # Backward pass
        total_loss.backward()
        # Perform asynchronous update of theta and theta_v
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                # Shared params. No need to copy again. Updated on optimizer.
                break
            # First update to global_param
            global_param._grad = local_param.grad
        # Step en la direccion del gradiente, para los parametros GLOBALES
        optimizer.step()

        # Final del training
        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if index==0:
                writer.close()
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return
    return


def local_test(index, opt, global_model):
    torch.manual_seed(42 + index)
    env, num_states, num_actions = create_train_env(opt.layout, index=index)
    local_model = AC_NN_MODEL(num_states, num_actions)
    # Test model we are going to test (turn off dropout, no backward pass)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            # Copy global model to local model
            local_model.load_state_dict(global_model.state_dict(), strict=False)
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, ACTOR_HIDDEN_SIZE), dtype=torch.float)
                c_0 = torch.zeros((1, CRITIC_HIDDEN_SIZE), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        # Simple estimation: between(-1,1)
        value = value.clamp(-1.,1.)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        # render as seen by NN, but with colors
        render_miniature = True
        if render_miniature: 
            env.render(mode = 'human', id=index)
        actions.append(action)

        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
