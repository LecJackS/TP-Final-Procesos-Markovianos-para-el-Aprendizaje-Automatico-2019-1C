"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
import torch
from src.env import create_train_env

from src.model import Mnih2016ActorCriticWithDropout
AC_NN_MODEL = Mnih2016ActorCriticWithDropout
ACTOR_HIDDEN_SIZE=256
CRITIC_HIDDEN_SIZE=256

import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from collections import deque

import timeit

def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
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
    state = torch.from_numpy(env.reset())
    if opt.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    curr_episode = 0
    while True:
        if save:
            # Save trained model at save_interval
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:

                torch.save(global_model.state_dict(),
                           "{}/gym-pacman_{}".format(opt.saved_path, opt.layout))
        print("Process {}. Episode {}   ".format(index, curr_episode), end="\r")
        curr_episode += 1
        # Synchronize thread-specific parameters theta'=theta and theta'_v=theta_v
        # (copy global params to local params (after every episode))
        local_model.load_state_dict(global_model.state_dict())
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
            # Model prediction from state. Returns two functions:
            # * Action prediction (Policy function) -> logits (array with every action-value)
            # * Value prediction (Value function)   -> value (single value state-value)
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            # Softmax over action-values
            policy = F.softmax(logits, dim=1)
            # Log-softmax over action-values, to get the entropy of the policy
            log_policy = F.log_softmax(logits, dim=1)
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
            # Render as seen by NN, but with colors 
            if index < opt.num_processes_to_render:
                env.render(mode = 'human', id=index)
            # state to tensor
            state = torch.from_numpy(state)
            if opt.use_gpu:
                state = state.cuda()
            # If last local step, reset episode
            if curr_step > opt.num_global_steps:
                done = True
            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset())
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
        # Baseline rewards standarization over episode rewards.
        # Uncomment prints to see how rewards change
        # Should I
        #if index == 0:
        #    print("Rewards before:", rewards)
        mean_rewards = np.mean(rewards)
        std_rewards  = np.std(rewards)
        rewards = (rewards - mean_rewards) / (std_rewards + 1e-9)
        #if index == 0:
        #    print("Rewards after:", rewards)
        # Initialize R/G_t: Discounted reward over local steps
        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)
        # Standarize this reward estimation too
        #mean_rewards = np.mean([R, rewards])
        #std_rewards  = np.std([R, rewards])
        R = (R - mean_rewards) / (std_rewards + 1e-9)
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
            R = reward + opt.gamma * R
            # Accumulate gradients wrt parameters theta'
            actor_loss = actor_loss + log_policy * gae
            # Accumulate gradients wrt parameters theta'_v
            critic_loss = critic_loss + ((R - value)**2) / 2.
            entropy_loss = entropy_loss + entropy
        # Clamp critic loss value if too big
        max_critic_loss = 1./opt.lr
        critic_loss = critic_loss.clamp(-max_critic_loss, max_critic_loss)
        # Total process' loss
        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        # Clamp loss value if too big
        max_loss =  2 * max_critic_loss
        total_loss = total_loss.clamp(-max_loss, max_loss)

        # Saving logs for TensorBoard
        writer.add_scalar("Total_{}/Loss".format(index), total_loss, curr_episode)
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
            writer.close()
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return
    return


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
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
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, ACTOR_HIDDEN_SIZE), dtype=torch.float)
                c_0 = torch.zeros((1, CRITIC_HIDDEN_SIZE), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
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
