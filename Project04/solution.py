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
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)





warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

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
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

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

        # Compute log probability in inf_init_e support
        log_prob_inf_supp = normalDistribution.log_prob(action_inf_supp)

        # Compute the log for the bounded action
        log_prob = log_prob_inf_supp - torch.log(1 - action.pow(2))


        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
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
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
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
        self.actor = Actor(hidden_size=self.hiddenSize, hidden_layers=self.hidden_layers, actor_lr=self.learningRate)
        # Setup two q values network
        self.critic_1 = Critic(hidden_size=self.hiddenSize, hidden_layers=self.hidden_layers, critic_lr=self.learningRate)
        self.critic_2 = Critic(hidden_size=self.hiddenSize, hidden_layers=self.hidden_layers, critic_lr=self.learningRate)
        # Setup their q values target to help in stabilizing training
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
            s = torch.from_numpy(np.expand_dims(s, axis=0)).float()

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
        # Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        # Only start training if enough samples are in the buffer

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        self.update_critics(s_batch, a_batch, r_batch, s_prime_batch)
        self.update_actor(s_batch)

        # Make the soft update of the critic targets' networks
        self.critic_target_update(self.critic_1.criticNetwork, self.critic1_target.criticNetwork, tau=self.tau, soft_update=True)
        self.critic_target_update(self.critic_2.criticNetwork, self.critic2_target.criticNetwork, tau=self.tau, soft_update=True)

    # gradient step critique
    def update_critics(self, s_batch, a_batch, r_batch, s_prime_batch):
        """Update critic networks."""
        # Step 1: Calculate target Q-values for the next state using target critic networks
        with torch.no_grad():
            predicted_actions, predicted_log_probs = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=False)
            next_critic = torch.cat((s_prime_batch, predicted_actions), dim=1)
            target_q1 = self.critic1_target.criticNetwork(next_critic)
            target_q2 = self.critic2_target.criticNetwork(next_critic)
            # entropy appriximated by log probs
            # q is the value function
            # this way we aim to maximize entropy while maximising value function
            target_q = torch.min(target_q1, target_q2) - self.temp.get_param() * predicted_log_probs
            target_q_values = r_batch + self.gamma * target_q
            target_q_values = target_q_values

        # Step 2: Combine target Q-values with rewards to gives them to critic network
        _state_action_pairs = torch.cat((s_batch, a_batch), dim=1)

        # Step 3: Calculate Q-values for current state-action pairs using main critic networks
        score_q1 = self.critic_1.criticNetwork(_state_action_pairs)
        score_q2 = self.critic_2.criticNetwork(_state_action_pairs)

        # Step 4: Compute loss for each critic network
        critic_loss1 = F.mse_loss(score_q1, target_q_values)
        critic_loss2 = F.mse_loss(score_q2, target_q_values)

        self.run_gradient_update_step(self.critic_1, critic_loss1)
        self.run_gradient_update_step(self.critic_2, critic_loss2)

    # gradient step actor
    def update_actor(self, s_batch):
        """Update actor network."""
        # Step 1: Generate possible actions and their log probabilities for the given batch of states
        possible_actions, log_prob_actions = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        
        # Step 2: Calculate Q-values for these state-action pairs using critic networks
        critic = torch.cat((s_batch, possible_actions), dim=1)
        score_q1 = self.critic_1.criticNetwork(critic)
        score_q2 = self.critic_2.criticNetwork(critic)

        # Step 3: Compute the minimum of the Q-values from the two critic networks
        score_q = torch.min(score_q1, score_q2)

        # Step 4: Calculate actor loss
        actor_loss = (self.temp.get_param() * log_prob_actions - score_q)
        
        self.run_gradient_update_step(self.actor, actor_loss)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

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