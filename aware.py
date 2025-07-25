import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import time
import threading
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a task to be routed to an agent"""
    id: str
    task_type: str
    complexity: float
    requirements: Dict[str, float]
    embedding: np.ndarray
    priority: int = 1
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class TaskResult:
    """Represents the result of a completed task"""
    task_id: str
    agent_id: str
    success: bool
    completion_time: float
    quality_score: float
    confidence_score: float
    resource_usage: Dict[str, float]

class Agent:
    """Represents an agent in the ecosystem"""
    
    def __init__(self, agent_id: str, capabilities: np.ndarray, max_queue_size: int = 10):
        self.id = agent_id
        self.capabilities = capabilities  # d-dimensional capability vector
        self.max_queue_size = max_queue_size
        self.current_queue = []
        self.processing_task = None
        self.performance_history = []
        self.last_update = datetime.now()
        self.resource_usage = {"cpu": 0.0, "memory": 0.0, "gpu": 0.0}
        self.availability_pattern = self._generate_availability_pattern()
        
    def _generate_availability_pattern(self) -> Dict[int, float]:
        """Generate availability patterns throughout the day"""
        return {hour: random.uniform(0.6, 1.0) for hour in range(24)}
    
    def get_queue_length(self) -> int:
        return len(self.current_queue) + (1 if self.processing_task else 0)
    
    def is_available(self) -> bool:
        return len(self.current_queue) < self.max_queue_size
    
    def add_task(self, task: Task) -> bool:
        if self.is_available():
            self.current_queue.append(task)
            return True
        return False
    
    def process_task(self, task: Task) -> TaskResult:
        """Simulate task processing"""
        start_time = time.time()
        
        # Simulate processing time based on task complexity and agent capability
        capability_match = np.dot(self.capabilities, task.embedding) / (
            np.linalg.norm(self.capabilities) * np.linalg.norm(task.embedding)
        )
        processing_time = task.complexity / max(capability_match, 0.1)
        
        # Simulate actual processing delay
        time.sleep(min(processing_time * 0.001, 0.1))  # Scale down for simulation
        
        # Calculate success probability based on capability match
        success_prob = max(0.3, capability_match)
        success = random.random() < success_prob
        
        completion_time = time.time() - start_time
        quality_score = capability_match if success else 0.0
        confidence_score = capability_match
        
        # Update resource usage
        self.resource_usage["cpu"] = min(1.0, self.resource_usage["cpu"] + 0.1)
        self.resource_usage["memory"] = min(1.0, self.resource_usage["memory"] + 0.05)
        
        result = TaskResult(
            task_id=task.id,
            agent_id=self.id,
            success=success,
            completion_time=completion_time,
            quality_score=quality_score,
            confidence_score=confidence_score,
            resource_usage=self.resource_usage.copy()
        )
        
        self.performance_history.append(result)
        if len(self.performance_history) > 100:  # Keep only recent history
            self.performance_history.pop(0)
            
        return result

class CapabilityProfiler:
    """Continuously updates agent capability vectors based on performance feedback"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.agent_profiles = {}
        
    def update_capabilities(self, agent: Agent, task: Task, result: TaskResult):
        """Update agent capabilities based on task performance"""
        if agent.id not in self.agent_profiles:
            self.agent_profiles[agent.id] = agent.capabilities.copy()
        
        # Update capabilities based on performance
        if result.success:
            # Strengthen capabilities in the direction of successful tasks
            direction = task.embedding / np.linalg.norm(task.embedding)
            self.agent_profiles[agent.id] += self.learning_rate * direction * result.quality_score
        else:
            # Slightly weaken capabilities for failed tasks
            direction = task.embedding / np.linalg.norm(task.embedding)
            self.agent_profiles[agent.id] -= self.learning_rate * 0.5 * direction
        
        # Normalize to maintain unit vector properties
        self.agent_profiles[agent.id] = self.agent_profiles[agent.id] / np.linalg.norm(self.agent_profiles[agent.id])
        agent.capabilities = self.agent_profiles[agent.id]

class ConfidenceEstimator:
    """Estimates agent confidence for task completion using ensemble methods"""
    
    def __init__(self, ensemble_size: int = 5):
        self.ensemble_size = ensemble_size
        self.estimators = {}
        
    def estimate_confidence(self, agent: Agent, task: Task) -> float:
        """Estimate confidence based on historical performance and task similarity"""
        if not agent.performance_history:
            return 0.5  # Default confidence for new agents
        
        # Calculate similarity with previous tasks
        similarities = []
        outcomes = []
        
        for result in agent.performance_history[-20:]:  # Use recent history
            # Find the original task (simplified - in practice would need task storage)
            # For simulation, we'll use a synthetic similarity measure
            sim = random.uniform(0.3, 1.0)  # Placeholder similarity
            similarities.append(sim)
            outcomes.append(1.0 if result.success else 0.0)
        
        if not similarities:
            return 0.5
        
        # Weighted average based on similarity
        similarities = np.array(similarities)
        outcomes = np.array(outcomes)
        weights = similarities / np.sum(similarities)
        
        confidence = np.sum(weights * outcomes)
        
        # Apply calibration based on agent's historical confidence accuracy
        calibration_factor = self._get_calibration_factor(agent)
        confidence = confidence * calibration_factor
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _get_calibration_factor(self, agent: Agent) -> float:
        """Calculate calibration factor based on historical confidence accuracy"""
        if len(agent.performance_history) < 5:
            return 1.0
        
        # Simple calibration based on recent performance
        recent_results = agent.performance_history[-10:]
        success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
        avg_confidence = sum(r.confidence_score for r in recent_results) / len(recent_results)
        
        if avg_confidence > 0:
            return success_rate / avg_confidence
        return 1.0

class AvailabilityMonitor:
    """Tracks real-time system metrics and predicts agent availability"""
    
    def __init__(self):
        self.availability_history = defaultdict(list)
        self.load_patterns = defaultdict(dict)
        
    def get_availability(self, agent: Agent, task: Task) -> float:
        """Calculate agent availability considering multiple factors"""
        # Queue-based availability
        queue_factor = 1.0 - (agent.get_queue_length() / agent.max_queue_size)
        
        # Resource-based availability
        resource_factor = 1.0 - max(agent.resource_usage.values())
        
        # Time-based availability pattern
        current_hour = datetime.now().hour
        time_factor = agent.availability_pattern.get(current_hour, 0.8)
        
        # Combine factors
        availability = 0.4 * queue_factor + 0.3 * resource_factor + 0.3 * time_factor
        
        return np.clip(availability, 0.0, 1.0)
    
    def update_availability_patterns(self, agent: Agent):
        """Update availability patterns based on historical data"""
        current_hour = datetime.now().hour
        current_availability = self.get_availability(agent, None)
        
        if current_hour not in self.load_patterns[agent.id]:
            self.load_patterns[agent.id][current_hour] = []
        
        self.load_patterns[agent.id][current_hour].append(current_availability)
        
        # Update pattern with exponential moving average
        if len(self.load_patterns[agent.id][current_hour]) > 1:
            recent_avg = np.mean(self.load_patterns[agent.id][current_hour][-10:])
            agent.availability_pattern[current_hour] = 0.7 * agent.availability_pattern[current_hour] + 0.3 * recent_avg

class DQNNetwork(nn.Module):
    """Deep Q-Network for routing policy learning"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class RoutingPolicyEngine:
    """Combines insights from other modules to make optimal routing decisions"""
    
    def __init__(self, state_dim: int, max_agents: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.max_agents = max_agents
        self.q_network = DQNNetwork(state_dim, max_agents)
        self.target_network = DQNNetwork(state_dim, max_agents)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.alpha = 0.4  # Capability weight
        self.beta = 0.3   # Confidence weight
        self.gamma = 0.3  # Availability weight
        
        # RL parameters
        self.epsilon = 0.1
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.update_frequency = 100
        self.step_count = 0
        
    def compute_routing_score(self, agent: Agent, task: Task, capability_sim: float, 
                            confidence: float, availability: float) -> float:
        """Compute routing score using learned weights"""
        score = (self.alpha * capability_sim + 
                self.beta * confidence + 
                self.gamma * availability)
        return score
    
    def select_agent(self, agents: List[Agent], task: Task, 
                    capability_similarities: List[float],
                    confidences: List[float], 
                    availabilities: List[float]) -> Agent:
        """Select best agent using either learned policy or scoring function"""
        
        if random.random() < self.epsilon and len(self.memory) > self.batch_size:
            # Use learned policy
            state = self._create_state(task, agents, capability_similarities, confidences, availabilities)
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state).unsqueeze(0))
                available_agents = [i for i, agent in enumerate(agents) if agent.is_available()]
                if available_agents:
                    # Mask unavailable agents
                    masked_q_values = q_values.clone()
                    for i in range(len(agents)):
                        if i not in available_agents:
                            masked_q_values[0, i] = float('-inf')
                    action = masked_q_values.argmax().item()
                    if action < len(agents):
                        return agents[action]
        
        # Fallback to scoring function
        best_agent = None
        best_score = -1
        
        for i, agent in enumerate(agents):
            if not agent.is_available():
                continue
                
            score = self.compute_routing_score(
                agent, task, capability_similarities[i], 
                confidences[i], availabilities[i]
            )
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _create_state(self, task: Task, agents: List[Agent], 
                     capability_similarities: List[float],
                     confidences: List[float], 
                     availabilities: List[float]) -> np.ndarray:
        """Create state representation for DQN"""
        # State includes task features and agent features
        task_features = np.concatenate([
            task.embedding,
            [task.complexity, task.priority]
        ])
        
        # Agent features (for first N agents to maintain fixed size)
        agent_features = []
        for i in range(min(len(agents), self.max_agents)):
            if i < len(agents):
                agent_feat = np.array([
                    capability_similarities[i],
                    confidences[i],
                    availabilities[i],
                    agents[i].get_queue_length() / agents[i].max_queue_size
                ])
            else:
                agent_feat = np.zeros(4)  # Padding for fixed size
            agent_features.extend(agent_feat)
        
        state = np.concatenate([task_features] + [agent_features])
        return state
    
    def update_policy(self, state: np.ndarray, action: int, reward: float, 
                     next_state: np.ndarray, done: bool):
        """Update the routing policy based on outcome feedback"""
        self.memory.append((state, action, reward, next_state, done))
        self.step_count += 1
        
        if len(self.memory) >= self.batch_size and self.step_count % self.update_frequency == 0:
            self._train_step()
    
    def _train_step(self):
        """Perform a training step on the DQN"""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if self.step_count % (self.update_frequency * 10) == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

class AWAREFramework:
    """Main AWARE framework that coordinates all components"""
    
    def __init__(self, capability_dim: int = 64, max_agents: int = 100):
        self.capability_dim = capability_dim
        self.max_agents = max_agents
        
        # Initialize components
        self.capability_profiler = CapabilityProfiler()
        self.confidence_estimator = ConfidenceEstimator()
        self.availability_monitor = AvailabilityMonitor()
        self.routing_policy = RoutingPolicyEngine(
            state_dim=capability_dim + 2 + max_agents * 4,  # Task + agent features
            max_agents=max_agents
        )
        
        # System state
        self.agents = {}
        self.task_queue = deque()
        self.completed_tasks = []
        self.routing_history = []
        self.metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'total_response_time': 0.0,
            'total_routing_time': 0.0,
            'agent_utilization': defaultdict(float)
        }
        
        # Threading for continuous operation
        self.running = False
        self.processing_thread = None
        
    def add_agent(self, agent: Agent):
        """Add an agent to the ecosystem"""
        self.agents[agent.id] = agent
        logger.info(f"Added agent {agent.id} to ecosystem")
    
    def submit_task(self, task: Task):
        """Submit a task for routing"""
        self.task_queue.append(task)
        logger.info(f"Task {task.id} submitted for routing")
    
    def route_task(self, task: Task) -> Optional[Agent]:
        """Route a single task to the most appropriate agent"""
        start_time = time.time()
        
        if not self.agents:
            logger.warning("No agents available for routing")
            return None
        
        available_agents = [agent for agent in self.agents.values() if agent.is_available()]
        if not available_agents:
            logger.warning("No available agents for routing")
            return None
        
        # Compute features for all agents
        capability_similarities = []
        confidences = []
        availabilities = []
        
        for agent in available_agents:
            # Capability similarity
            cap_sim = np.dot(agent.capabilities, task.embedding) / (
                np.linalg.norm(agent.capabilities) * np.linalg.norm(task.embedding)
            )
            capability_similarities.append(cap_sim)
            
            # Confidence estimation
            confidence = self.confidence_estimator.estimate_confidence(agent, task)
            confidences.append(confidence)
            
            # Availability
            availability = self.availability_monitor.get_availability(agent, task)
            availabilities.append(availability)
        
        # Select best agent
        selected_agent = self.routing_policy.select_agent(
            available_agents, task, capability_similarities, confidences, availabilities
        )
        
        if selected_agent:
            # Add task to agent's queue
            if selected_agent.add_task(task):
                routing_time = time.time() - start_time
                self.metrics['total_routing_time'] += routing_time
                
                # Record routing decision
                self.routing_history.append({
                    'task_id': task.id,
                    'agent_id': selected_agent.id,
                    'routing_time': routing_time,
                    'capability_similarity': capability_similarities[available_agents.index(selected_agent)],
                    'confidence': confidences[available_agents.index(selected_agent)],
                    'availability': availabilities[available_agents.index(selected_agent)]
                })
                
                logger.info(f"Task {task.id} routed to agent {selected_agent.id} in {routing_time:.4f}s")
                return selected_agent
        
        logger.warning(f"Failed to route task {task.id}")
        return None
    
    def process_tasks(self):
        """Continuously process tasks from the queue"""
        while self.running:
            try:
                if self.task_queue:
                    task = self.task_queue.popleft()
                    selected_agent = self.route_task(task)
                    
                    if selected_agent:
                        # Process task (in a real system, this would be asynchronous)
                        result = selected_agent.process_task(task)
                        self._handle_task_result(task, result, selected_agent)
                
                # Update availability patterns
                for agent in self.agents.values():
                    self.availability_monitor.update_availability_patterns(agent)
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in task processing: {e}")
    
    def _handle_task_result(self, task: Task, result: TaskResult, agent: Agent):
        """Handle the result of a completed task"""
        self.completed_tasks.append(result)
        
        # Update metrics
        self.metrics['total_tasks'] += 1
        if result.success:
            self.metrics['successful_tasks'] += 1
        self.metrics['total_response_time'] += result.completion_time
        self.metrics['agent_utilization'][agent.id] += 1
        
        # Update agent capabilities
        self.capability_profiler.update_capabilities(agent, task, result)
        
        # Update routing policy (simplified reward)
        reward = 1.0 if result.success else -0.5
        if len(self.routing_history) > 0:
            last_routing = self.routing_history[-1]
            if last_routing['task_id'] == task.id:
                # Create state representation (simplified)
                state = np.random.random(self.routing_policy.state_dim)  # Placeholder
                action = list(self.agents.keys()).index(agent.id)
                next_state = np.random.random(self.routing_policy.state_dim)  # Placeholder
                
                self.routing_policy.update_policy(state, action, reward, next_state, True)
        
        logger.info(f"Task {task.id} completed by {agent.id} - Success: {result.success}")
    
    def start(self):
        """Start the AWARE framework"""
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_tasks)
        self.processing_thread.start()
        logger.info("AWARE framework started")
    
    def stop(self):
        """Stop the AWARE framework"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("AWARE framework stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if self.metrics['total_tasks'] == 0:
            return self.metrics
        
        accuracy = (self.metrics['successful_tasks'] / self.metrics['total_tasks']) * 100
        avg_response_time = self.metrics['total_response_time'] / self.metrics['total_tasks']
        avg_routing_time = self.metrics['total_routing_time'] / self.metrics['total_tasks']
        
        # Calculate resource utilization
        total_utilization = sum(self.metrics['agent_utilization'].values())
        utilization_percentage = (total_utilization / (len(self.agents) * self.metrics['total_tasks'])) * 100 if self.metrics['total_tasks'] > 0 else 0
        
        return {
            'accuracy': accuracy,
            'avg_response_time': avg_response_time,
            'avg_routing_time': avg_routing_time * 1000,  # Convert to milliseconds
            'utilization': utilization_percentage,
            'total_tasks': self.metrics['total_tasks'],
            'successful_tasks': self.metrics['successful_tasks'],
            'total_agents': len(self.agents)
        }

# Example usage and demonstration
def create_sample_ecosystem():
    """Create a sample ecosystem for testing"""
    
    # Initialize AWARE framework
    aware = AWAREFramework(capability_dim=10, max_agents=50)
    
    # Create diverse agents
    agent_types = [
        ("python_dev", [0.9, 0.8, 0.3, 0.6, 0.7, 0.4, 0.5, 0.6, 0.3, 0.4]),
        ("js_dev", [0.3, 0.9, 0.8, 0.5, 0.6, 0.4, 0.5, 0.7, 0.3, 0.4]),
        ("math_expert", [0.4, 0.3, 0.2, 0.9, 0.8, 0.9, 0.7, 0.4, 0.6, 0.5]),
        ("nlp_specialist", [0.5, 0.4, 0.3, 0.6, 0.5, 0.4, 0.9, 0.8, 0.9, 0.7]),
        ("generalist", [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
    ]
    
    # Create multiple agents of each type
    for i, (agent_type, capabilities) in enumerate(agent_types):
        for j in range(10):  # 10 agents of each type
            agent_id = f"{agent_type}_{j}"
            # Add some noise to capabilities
            noisy_caps = np.array(capabilities) + np.random.normal(0, 0.1, len(capabilities))
            noisy_caps = np.clip(noisy_caps, 0, 1)
            # Normalize
            noisy_caps = noisy_caps / np.linalg.norm(noisy_caps)
            
            agent = Agent(agent_id, noisy_caps)
            aware.add_agent(agent)
    
    return aware

def create_sample_tasks(num_tasks: int = 100):
    """Create sample tasks for testing"""
    tasks = []
    task_types = [
        ("python_coding", [0.9, 0.7, 0.2, 0.4, 0.5, 0.3, 0.4, 0.5, 0.2, 0.3]),
        ("javascript_coding", [0.2, 0.9, 0.8, 0.3, 0.4, 0.3, 0.4, 0.6, 0.2, 0.3]),
        ("math_problem", [0.3, 0.2, 0.1, 0.9, 0.8, 0.9, 0.5, 0.3, 0.4, 0.4]),
        ("text_analysis", [0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.9, 0.8, 0.9, 0.6]),
        ("general_query", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ]
    
    for i in range(num_tasks):
        task_type, embedding = random.choice(task_types)
        # Add noise to task embeddings
        noisy_embedding = np.array(embedding) + np.random.normal(0, 0.1, len(embedding))
        noisy_embedding = np.clip(noisy_embedding, 0, 1)
        noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
        
        task = Task(
            id=f"task_{i}",
            task_type=task_type,
            complexity=random.uniform(0.5, 2.0),
            requirements={},
            embedding=noisy_embedding,
            priority=random.randint(1, 5)
        )
        tasks.append(task)
    
    return tasks

def run_demonstration():
    """Run a demonstration of the AWARE framework"""
    print("Creating AWARE ecosystem...")
    aware = create_sample_ecosystem()
    
    print("Creating sample tasks...")
    tasks = create_sample_tasks(50)
    
    print("Starting AWARE framework...")
    aware.start()
    
    print("Submitting tasks...")
    for task in tasks[:20]:  # Submit first 20 tasks
        aware.submit_task(task)
        time.sleep(0.1)  # Small delay between submissions
    
    # Let the system process for a while
    print("Processing tasks...")
    time.sleep(5)
    
    # Submit more tasks
    print("Submitting additional tasks...")
    for task in tasks[20:]:
        aware.submit_task(task)
        time.sleep(0.05)
    
    # Wait for processing
    time.sleep(10)
    
    # Get metrics
    metrics = aware.get_metrics()
    print("\n=== AWARE Framework Performance Metrics ===")
    print(f"Total Tasks: {metrics['total_tasks']}")
    print(f"Successful Tasks: {metrics['successful_tasks']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Average Response Time: {metrics['avg_response_time']:.4f}s")
    print(f"Average Routing Time: {metrics['avg_routing_time']:.2f}ms")
    print(f"Resource Utilization: {metrics['utilization']:.2f}%")
    print(f"Total Agents: {metrics['total_agents']}")
    
    print("\nStopping AWARE framework...")
    aware.stop()
    print("Demonstration complete!")

if __name__ == "__main__":
    run_demonstration()
