import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Config -----
CAPABILITY_DIM = 16  # Dimensionality of capability and task embeddings
NUM_AGENTS = 50
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000

# ----- Utility Functions -----

def cosine_similarity(a, b):
    # a, b: (..., d)
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return (a_norm * b_norm).sum(dim=-1)  # (...,)

# ----- Agent & Task Embeddings -----

class Agent:
    def __init__(self, capability_vector):
        self.capability = capability_vector.to(device)
        self.current_load = 0
        self.max_load = 10  # Arbitrary availability constraint

    def availability(self):
        # Returns a score [0,1] based on load (simple inverse load)
        avail_score = max(0, 1 - (self.current_load / self.max_load))
        return torch.tensor(avail_score, device=device, dtype=torch.float32)

# Dummy task embedding function
def embed_task(task_features):
    # task_features is a vector-like tensor representing the task
    # For simplicity, assume task_features is already a vector in capability space
    return task_features.to(device)

# ----- Confidence Estimator -----

class ConfidenceEstimator(nn.Module):
    def __init__(self, capability_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(capability_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # confidence in [0,1]
        )

    def forward(self, agent_caps, task_embeds):
        # agent_caps, task_embeds: (batch_size, capability_dim)
        x = torch.cat([agent_caps, task_embeds], dim=1)
        conf = self.fc(x).squeeze(-1)
        return conf

# ----- DQN Network for Routing Policy -----

class RoutingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----- Replay Memory -----

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(tuple(args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.memory)

# ----- AWARE System -----

class AWARESystem:
    def __init__(self, num_agents=NUM_AGENTS, capability_dim=CAPABILITY_DIM):
        # Initialize agents with random capability vectors
        self.agents = [
            Agent(torch.randn(capability_dim))
            for _ in range(num_agents)
        ]

        self.capability_dim = capability_dim
        self.num_agents = num_agents

        # Confidence estimator
        self.confidence_estimator = ConfidenceEstimator(capability_dim).to(device)

        # Routing DQN takes as input: state_dim (can be task embedding + global stats)
        # Actions = num_agents (which agent to assign)
        self.state_dim = capability_dim * 2  # task embedding + global load summary (simplified)
        self.action_dim = num_agents
        self.policy_net = RoutingDQN(self.state_dim, self.action_dim).to(device)
        self.target_net = RoutingDQN(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.confidence_estimator.parameters()), lr=LR
        )
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.steps_done = 0

        # Routing weights initialized (alpha, beta, gamma)
        self.routing_weights = torch.tensor([1.0, 1.0, 1.0], device=device)  # can be learned

    def select_action(self, state, task_embed, epsilon=0.1):
        # state: tensor representing current system state
        sample = random.random()
        self.steps_done += 1
        if sample < epsilon:
            return random.randrange(self.num_agents)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0)).squeeze(0)  # (num_agents,)
            # Compute routing scores combining capability similarity, confidence, availability
            scores = []
            for i, agent in enumerate(self.agents):
                cap_sim = cosine_similarity(agent.capability.unsqueeze(0), task_embed.unsqueeze(0)).item()
                conf = self.confidence_estimator(agent.capability.unsqueeze(0), task_embed.unsqueeze(0)).item()
                avail = agent.availability().item()
                score = (self.routing_weights[0].item() * cap_sim +
                         self.routing_weights[1].item() * conf +
                         self.routing_weights[2].item() * avail)
                scores.append(score)
            scores_tensor = torch.tensor(scores, device=device)
            combined = q_values + scores_tensor  # combining learned Q + heuristic score
            action = torch.argmax(combined).item()
            return action

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones, task_embeds = self.memory.sample(BATCH_SIZE)

        states = torch.stack(states).to(device)               # (batch, state_dim)
        actions = torch.tensor(actions, device=device)        # (batch,)
        rewards = torch.tensor(rewards, device=device)        # (batch,)
        next_states = torch.stack(next_states).to(device)     # (batch, state_dim)
        dones = torch.tensor(dones, device=device)             # (batch,)
        task_embeds = torch.stack(task_embeds).to(device)     # (batch, capability_dim)

        # Current Q-values
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values from target network
        next_q_values = self.target_net(next_states).max(1)[0].detach()

        # Compute expected Q values
        expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        # Loss for DQN
        loss = F.mse_loss(state_action_values, expected_q_values)

        # Confidence loss (optional): encourage confidence estimator to predict reward
        conf_preds = []
        for i, action_idx in enumerate(actions):
            agent_cap = self.agents[action_idx].capability.unsqueeze(0)
            conf_pred = self.confidence_estimator(agent_cap, task_embeds[i].unsqueeze(0)).squeeze(0)
            conf_preds.append(conf_pred)
        conf_preds = torch.stack(conf_preds)
        conf_loss = F.mse_loss(conf_preds, rewards)

        total_loss = loss + conf_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_system_state(self, task_embed):
        # Simplified state representation:
        # Concatenate task embedding and mean agent load normalized
        mean_load = torch.tensor([np.mean([agent.current_load / agent.max_load for agent in self.agents])], device=device)
        state = torch.cat([task_embed, mean_load.repeat(self.capability_dim)], dim=0)
        return state

    def route_task(self, task_features):
        # Embed task
        task_embed = embed_task(task_features)

        # Get current state
        state = self.get_system_state(task_embed)

        # Select agent via policy
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        action = self.select_action(state, task_embed, epsilon)

        # Simulate task assignment (increment load)
        self.agents[action].current_load = min(self.agents[action].max_load, self.agents[action].current_load + 1)

        return action, state, task_embed

    def feedback(self, action, reward, next_task_features, done):
        # Get next state
        next_task_embed = embed_task(next_task_features)
        next_state = self.get_system_state(next_task_embed)

        # Store transition
        # For simplicity, assume current state was last used for action
        # Real code should track current state when routing task
        self.memory.push(self.last_state, action, reward, next_state, done, self.last_task_embed)

        # Optimize model
        self.optimize_model()

        if done:
            self.update_target_network()

    def step(self, task_features, reward, done):
        # Step wrapper combining route and feedback
        action, state, task_embed = self.route_task(task_features)
        self.last_state = state
        self.last_task_embed = task_embed
        self.feedback(action, reward, task_features, done)
        return action
