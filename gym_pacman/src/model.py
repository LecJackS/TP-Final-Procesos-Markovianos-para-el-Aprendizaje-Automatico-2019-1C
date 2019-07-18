"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# PyTorch input tensor : [batch_size, channel, height, width]

class Mnih2016ActorCriticWithDropout(nn.Module):
    """Same as Mnih2016ActorCritic but with dropout layers for each layer"""
    def __init__(self, num_inputs, num_actions):
        super(Mnih2016ActorCriticWithDropout, self).__init__()
        self.p_drop = 0.0
        self.p_drop_conv = 0.0
        self.conv1_drop = nn.Dropout2d(p=self.p_drop_conv)
        self.conv1      = nn.Conv2d(num_inputs, 16, 8, stride=4) #no padding
        self.conv2_drop = nn.Dropout2d(p=self.p_drop_conv)
        self.conv2      = nn.Conv2d(16, 32, 4, stride=2)
        # Model Representation
        self.lstm = nn.LSTMCell(32 * 9 * 9, 256)
        # Outputs
        self.actor_drop    = nn.Dropout(p=self.p_drop)
        self.actor_linear  = nn.Linear(256, num_actions)
        self.critic_drop   = nn.Dropout(p=self.p_drop)
        self.critic_linear = nn.Linear(256, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        # Dropout after relu for convnets
        #mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf
        x = self.conv1_drop(F.relu(self.conv1(x)))
        x = self.conv2_drop(F.relu(self.conv2(x)))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        # Softmax is applied later, same with log-softmax
        actor  = self.actor_drop(self.actor_linear(hx))
        critic = self.critic_drop(self.critic_linear(cx))
        # Uncomment to print forward values
        #print("Actor: " , actor)
        #print("Critic: ", critic)
        use_dropout = False
        if use_dropout:
            # Changes dropout rate every forward pass
            min_drop = 0.0
            max_drop_conv   = 0.2
            max_drop_linear = 0.6
            random_dropout_rate_conv   = torch.FloatTensor(1).uniform_(min_drop, max_drop_conv).item()
            random_dropout_rate_linear = torch.FloatTensor(1).uniform_(min_drop, max_drop_linear).item()
            self.p_drop_conv   = random_dropout_rate_conv
            self.p_drop_linear = random_dropout_rate_linear
        return actor, critic, hx, cx


class Mnih2016ActorCritic(nn.Module):
    """From: Asynchronous Methods for Deep Reinforcement Learning
    The first hidden layer convolves 16 filters of 8x8 with stride 4
    The second hidden layer convolves 32 filters of 4x4 with stride 2
    The final hidden layer is fully-connected 256 rectifier units.
    The output layer is a fully-connected linear layer with a
    single output for each valid action."""
    def __init__(self, num_inputs, num_actions):
        super(Mnih2016ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4) #no padding
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        # Model Representation
        self.lstm = nn.LSTMCell(32 * 9 * 9, 256)
        # Outputs
        self.actor_linear  = nn.Linear(256, num_actions)
        self.critic_linear = nn.Linear(256, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        # Softmax is applied later, same with log-softmax
        actor  = self.actor_linear(hx) 
        critic = self.critic_linear(hx)
        # Uncomment to print forward values
        #print("Actor: " , actor)
        #print("Critic: ", critic)
        return actor, critic, hx, cx

class Mnih2015Model(nn.Module):
    """From: Human-level control through deep reinforcement learning
    The first hidden layer convolves 32 filters of 8x8 with stride 4
    The second hidden layer convolves 64 filters of 4x4 with stride 2
    The third convolutional layer convolves 64 filters of 3x3 with stride 1
    The final hidden layer is fully-connected 512 rectifier units.
    The output layer is a fully-connected linear layer with a
    single output for each valid action."""
    def __init__(self, num_inputs, num_actions):
        super(Mnih2015Model, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4) #no padding
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.lstm = nn.LSTMCell(64*7*7, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx



class ActorCritic(nn.Module):
    # original model from
    # https://github.com/vietnguyen91/Super-mario-bros-A3C-pytorch
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx