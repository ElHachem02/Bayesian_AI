import torch
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
import torch.nn.functional as F
from copy import deepcopy



warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def _init_(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self)._init_()

        # Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        
        # Define activation function
        activation = nn.ReLU() if activation == "relu" else nn.Tanh()

        # Create a list of layers
        layers = [nn.Linear(input_dim, hidden_size), activation]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        # Add the output layer
        layers.append(nn.Linear(hidden_size, output_dim))

        # Use nn.Sequential for simplicity
        self.model = nn.Sequential(*layers)


    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Implement the forward pass for the neural network you have defined.
        return self.model(s)
    
class Actor:
    def _init_(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self)._init_()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 
        self.actorNetwork = NeuralNetwork(input_dim=self.state_dim, hidden_layers=self.hidden_layers, 
                                         hidden_size=self.hidden_size, output_dim=self.action_dim*2, activation="relu")
                                         
                
        self.optimizer = optim.Adam(self.actorNetwork.parameters(), lr=self.actor_lr)


    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])

        # Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.

        mean = self.actorNetwork(state)[:, :self.action_dim]
        logStd =  self.actorNetwork(state)[:, self.action_dim:]
        std = self.clamp_log_std(logStd).exp()

        # Sample action
        normalDistribution = Normal(mean, std)
        action_inf_supp = mean if deterministic else normalDistribution.rsample()

        # Bound action
        action = torch.tanh(action_inf_supp)

        # Compute log probability in infinite support
        log_prob_inf_supp = normalDistribution.log_prob(action_inf_supp)

        # Compute the log for the bounded action
        log_prob = log_prob_inf_supp - torch.log(1 - action.pow(2))


        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def _init_(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self)._init_()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.criticNetwork = NeuralNetwork(input_dim=self.state_dim + self.action_dim, hidden_layers=self.hidden_layers, 
                                         hidden_size=self.hidden_size, output_dim=1, activation="relu").to(self.device)
        
        self.optimizer = optim.Adam(self.criticNetwork.parameters(), lr=self.critic_lr)


class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def _init_(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def _init_(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        self.hiddenSize = 256
        self.hidden_layers = 2
        self.learningRate = 3e-4
        self.tau = 0.005
        self.gamma = 0.99

        self.setup_agent()

    def setup_agent(self):
        # Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need. 

        # Finetune these parameters
        self.actor = Actor(hidden_size=self.hiddenSize, hidden_layers=self.hidden_layers, actor_lr=self.learningRate)
        self.critic_1 = Critic(hidden_size=self.hiddenSize, hidden_layers=self.hidden_layers, critic_lr=self.learningRate)
        self.critic_2 = Critic(hidden_size=self.hiddenSize, hidden_layers=self.hidden_layers, critic_lr=self.learningRate)
        self.critic1_target = deepcopy(self.critic_1)
        self.critic2_target = deepcopy(self.critic_2)
        self.temp = TrainableParameter(init_param=0.2, lr_param=self.learningRate, train_param=True, device=self.device)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # Don't add these to the computation graph
        with torch.no_grad():
            # Convert state to tensor
            s = torch.from_numpy(np.expand_dims(s, axis=0)).float()

            # Run through actor network to get an action, use deterministic policy on eval and stochastic on train
            action, _ = self.actor.get_action_and_log_prob(s, not train)

            action = action.numpy()[0]

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)


    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        # Only start training if enough samples are in the buffer
        if not self.memory.start_training():
            return

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        # Update code mostly inspired by: https://github.com/denisyarats/pytorch_sac/blob/master/agent/sac.py
        # update_critic, update_actor_and_alpha methods 

        # Update Critic Networks
        with torch.no_grad():
            # Compute the next actions and their log probabilities
            next_action_batch, log_prob_next = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=False)
            
            # Concatenate s_prime_batch and next_action_batch for critic input
            critic_input_next = torch.cat((s_prime_batch, next_action_batch), dim=1)

            # Compute target Q values using both networks in the target critic
            target_q1_next = self.critic_target.q_network_1(critic_input_next)
            target_q2_next = self.critic_target.q_network_2(critic_input_next)
                

            target_q_min = torch.min(target_q1_next, target_q2_next) - self.temperature.get_param() * log_prob_next
            target_q_value = r_batch + 0.99 * target_q_min  # Assuming discount factor gamma = 0.99


        # Concatenate s_batch and a_batch for critic input
        critic_input_current = torch.cat((s_batch, a_batch), dim=1)

        # Get current Q estimates from both networks in the main critic
        current_q1 = self.critic.q_network_1(critic_input_current)
        current_q2 = self.critic.q_network_2(critic_input_current)

        # Compute critic loss
        critic_loss_1 = nn.functional.mse_loss(current_q1, target_q_value)
        critic_loss_2 = nn.functional.mse_loss(current_q2, target_q_value)

        # Update the critics (for both networks)
        self.critic.optimizer = self.critic.optimizer_1 
        self.run_gradient_update_step(self.critic, critic_loss_1)

        self.critic.optimizer = self.critic.optimizer_2
        self.run_gradient_update_step(self.critic, critic_loss_2)

        # Update Policy (Actor)
        new_action_batch, log_prob = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        critic_input_new_action = torch.cat((s_batch, new_action_batch), dim=1)

        q1_new = self.critic.q_network_1(critic_input_new_action)
        q2_new = self.critic.q_network_2(critic_input_new_action)
        q_min_new = torch.min(q1_new, q2_new)

        # Compute actor loss
        actor_loss = (self.temperature.get_param() * log_prob - q_min_new).mean()

        # Update the actor
        self.run_gradient_update_step(self.actor, actor_loss)

        # Soft Update the Target Critic Networks
        self.critic_target_update(self.critic.q_network_1, self.critic_target.q_network_1, tau=0.005, soft_update=True)
        self.critic_target_update(self.critic.q_network_2, self.critic_target.q_network_2, tau=0.005, soft_update=True)

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if _name_ == '_main_':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close