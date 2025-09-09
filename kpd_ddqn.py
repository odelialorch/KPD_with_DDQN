import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import subgraph
from datetime import datetime

# ----- COMMAND‐LINE ARGUMENTS -----
parser = argparse.ArgumentParser(description="Run KPD-DQN with customizable horizon and arrival rate")
parser.add_argument("--ts", "--timesteps", dest="timesteps", type=int, default=50,
                    help="Number of timesteps in each simulation (default: 50)")

# DQN Agent Hyperparameters
parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=1e-3,
                    help="Learning rate for the Adam optimizer (default: 1e-3)")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.99,
                    help="Discount factor for future rewards (default: 0.99)")
parser.add_argument("--buffer_cap", dest="buffer_cap", type=int, default=10000,
                    help="Capacity of the replay buffer (default: 10000)")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64,
                    help="Batch size for training (default: 64)")
parser.add_argument("--target_update", dest="target_update", type=int, default=100,
                    help="Frequency of target network updates (default: 100 steps)")

# GNN Model Hyperparameters
parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=64,
                    help="Dimensionality of hidden layers in GNN (default: 64)")
parser.add_argument("--edge_dim", dest="edge_dim", type=int, default=32,
                    help="Dimensionality of edge MLP output (default: 32)")

# Training Loop Hyperparameters
parser.add_argument("--num_episodes", dest="num_episodes", type=int, default=50,
                    help="Number of training episodes (default: 50)")
parser.add_argument("--0", dest="eps_start", type=float, default=1.0,
                    help="Starting epsilon for epsilon-greedy exploration (default: 1.0)")
parser.add_argument("--eps_end", dest="eps_end", type=float, default=0.05,
                    help="Ending epsilon for epsilon-greedy exploration (default: 0.05)")
parser.add_argument("--eps_decay", dest="eps_decay", type=float, default=0.995,
                    help="Epsilon decay rate per episode (default: 0.995)")
parser.add_argument("--log_dir", dest="log_dir", type=str, default="runs/kpd_ddqn",
                    help="Directory for TensorBoard logs (default: runs/kpd_ddqn)")

# New Reward Function Parameters
parser.add_argument("--reward_type", dest="reward_type", type=str, default="simple",
                    choices=["simple", "feature_dist", "mixed"],
                    help="Type of reward function: 'simple' (r_e), 'feature_dist' (r_f), or 'mixed' (r_m) (default: simple)")
parser.add_argument("--feature_name", dest="feature_name", type=str, default="patient_gender",
                    help="Node feature for r_f and r_m (e.g., 'patient_gender', 'donor_blood').")
parser.add_argument("--desired_dist", dest="desired_dist", type=str, default="Male:0.5,Female:0.5",
                    help="Desired distribution for the chosen feature, as 'category1:prob1,category2:prob2,...'. Sum of probs must be 1.")
parser.add_argument("--reward_a", dest="reward_a", type=float, default=1.0,
                    help="Weight 'a' for r_e in the mixed reward function (r_m = a*r_e + b*r_f).")
parser.add_argument("--reward_b", dest="reward_b", type=float, default=1.0,
                    help="Weight 'b' for r_f in the mixed reward function (r_m = a*r_e + b*r_f).")

args = parser.parse_args()

TIMESTEPS     = args.timesteps # horizon length

# ------------------------------------------------------------------
# 1. DEVICE SETUP
# ------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------------------------------------------
# 2. Import timeline generator from generate_data.py
# ------------------------------------------------------------------
from generate_data import simulate_kpd_network_timesteps

# ------------------------------------------------------------------
# 3. Preprocessing Utilities
# ------------------------------------------------------------------

GENDER_MAP   = {"Male": 0, "Female": 1}
RACE_MAP     = {"White": 0, "Black": 1, "Hispanic": 2, "Asian": 3, "Other": 4}
BLOOD_MAP    = {"O": 0, "A": 1, "B": 2, "AB": 3}
HEALTH_MAP   = {"Good": 0, "Fair": 1, "Poor": 2}
INCOMPAT_MAP = {"Compatible": 0, "ABO": 1, "HLA": 2}

# Reverse maps to get category names from numerical features (for r_f)
REVERSE_GENDER_MAP = {v: k for k, v in GENDER_MAP.items()}
REVERSE_RACE_MAP = {v: k for k, v in RACE_MAP.items()}
REVERSE_BLOOD_MAP = {v: k for k, v in BLOOD_MAP.items()}
REVERSE_HEALTH_MAP = {v: k for k, v in HEALTH_MAP.items()}
REVERSE_INCOMPAT_MAP = {v: k for k, v in INCOMPAT_MAP.items()}


# This dictionary maps feature names to their respective reverse maps.
# Only categorical features supported for r_f distribution calculations.
FEATURE_REVERSE_MAP_LOOKUP = {
    "patient_gender": REVERSE_GENDER_MAP,
    "donor_gender": REVERSE_GENDER_MAP,
    "patient_race": REVERSE_RACE_MAP,
    "donor_race": REVERSE_RACE_MAP,
    "patient_blood": REVERSE_BLOOD_MAP,
    "donor_blood": REVERSE_BLOOD_MAP,
    "patient_health": REVERSE_HEALTH_MAP,
    "incompat_reason": REVERSE_INCOMPAT_MAP,
}

def preprocess_timeline(timeline):
    """
    Convert each NetworkX‐based 'graph' in timeline to a GPU‐resident PyG Data object.
    Store original node data for feature lookup.
    Skip if 'pyg_full' already exists.
    """
    for entry in timeline:
        if "pyg_full" in entry:
            continue
        if "graph" not in entry:
            raise KeyError("Expected 'graph' key but not found.")

        G = entry["graph"]
        orig_ids = list(G.nodes())
        orig2local = {nid: idx for idx, nid in enumerate(orig_ids)}

        # Store original node data for later feature lookup
        entry["node_data_map"] = {nid: G.nodes[nid] for nid in orig_ids}

        # Build node features [N, 11] on device
        feats = []
        for nid in orig_ids:
            d = G.nodes[nid]
            feats.append([
                GENDER_MAP[d["patient_gender"]],
                GENDER_MAP[d["donor_gender"]],
                RACE_MAP[d["patient_race"]],
                RACE_MAP[d["donor_race"]],
                d["patient_age"] / 100.0,
                d["donor_age"] / 100.0,
                BLOOD_MAP[d["patient_blood"]],
                BLOOD_MAP[d["donor_blood"]],
                INCOMPAT_MAP[d["incompat_reason"]],
                HEALTH_MAP[d["patient_health"]],
                d["patient_cpra"] / 100.0,
            ])
        x = torch.tensor(feats, dtype=torch.float, device=device)

        # Build directed edges [2, 2E] on device
        edge_idx_list = []
        for (u, v) in G.edges():
            iu = orig2local[u]
            iv = orig2local[v]
            edge_idx_list.append([iu, iv])
            edge_idx_list.append([iv, iu])
        if edge_idx_list:
            edge_index = torch.tensor(edge_idx_list, dtype=torch.long, device=device).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        data_full = Data(x=x, edge_index=edge_index)
        data_full.orig_ids = orig_ids
        entry["pyg_full"] = data_full

    return timeline

def mask_data(data_full, matched_nodes):
    """
    Given a full Data object and matched node IDs, return a masked Data with those nodes removed.
    """
    orig_ids = data_full.orig_ids
    mask = torch.tensor([nid not in matched_nodes for nid in orig_ids],
                        dtype=torch.bool, device=device)
    edge_index_sub, _ = subgraph(mask, data_full.edge_index, relabel_nodes=True)
    x_sub = data_full.x[mask]
    return Data(x=x_sub, edge_index=edge_index_sub)

# ------------------------------------------------------------------
# 4. Environment Definition
# ------------------------------------------------------------------

class KPDEnv:
    """
    Each timeline entry has 'pyg_full': a Data object on GPU.
    matched_nodes tracks which original IDs are matched.
    step(action) → (next_state, reward, done, info).
    """
    def __init__(self, timeline, reward_type, feature_name, desired_dist_str, reward_a, reward_b):
        self.timeline = preprocess_timeline(timeline)
        self.timesteps = len(self.timeline)
        self.current_index = 0
        self.matched_nodes = set() # Set of original IDs of matched nodes
        self.matched_node_orig_ids_list = [] # List of original IDs, for tracking order of matches
        self.done = False

        # Reward Function Configuration
        self.reward_type = reward_type
        self.feature_name = feature_name
        self.reward_a = reward_a
        self.reward_b = reward_b
        self.desired_dist_map = self._parse_desired_dist(desired_dist_str, feature_name)
        self.r_f_epsilon = 1e-6 # Small epsilon for numerical stability in r_f denominator

    def _parse_desired_dist(self, desired_dist_str, feature_name):
        dist_map = {}
        total_prob = 0.0

        if feature_name not in FEATURE_REVERSE_MAP_LOOKUP:
            if feature_name in ["patient_age", "donor_age", "patient_cpra"]:
                raise ValueError(f"Feature '{feature_name}' is numerical and cannot be used with discrete distributions for r_f. Choose a categorical feature like 'patient_gender' or 'donor_blood'.")
            else:
                 raise ValueError(f"Unsupported feature name for r_f: {feature_name}. It must be a categorical feature defined in FEATURE_REVERSE_MAP_LOOKUP.")

        for item in desired_dist_str.split(','):
            if ':' not in item:
                raise ValueError(f"Invalid desired_dist format: {desired_dist_str}. Expected 'category:prob'.")
            category, prob_str = item.split(':')
            prob = float(prob_str)
            dist_map[category.strip()] = prob
            total_prob += prob

        if not np.isclose(total_prob, 1.0):
            print(f"Warning: Desired distribution probabilities sum to {total_prob:.2f}, not 1.0. Normalizing...")
            # Normalize the probabilities if they don't sum to 1
            factor = 1.0 / total_prob
            dist_map = {k: v * factor for k, v in dist_map.items()}
            
        return dist_map

    def reset(self):
        self.current_index = 0
        self.matched_nodes.clear()
        self.matched_node_orig_ids_list = []
        self.done = False
        return self._get_state()

    def _get_state(self):
        entry = self.timeline[self.current_index]
        return mask_data(entry["pyg_full"], self.matched_nodes)

    def _calculate_r_e(self, base_match_reward, current_reward_for_departure):
        """
        Simple reward: +1 for a match, +0 for no matches (including departures).
        """
        if base_match_reward == 1: # A match was made
            return 1.0
        # If no match was made by the action, or if a patient departed, reward is 0.
        # This overrides the -1 for departure in the base env logic if reward_type is simple
        return 0.0

    def _calculate_r_f(self, current_timeline_entry):
        """
        Calculates reward based on MSE between desired feature distribution and matched nodes' distribution.
        Reward = 1 / (MSE + epsilon).
        """
        if not self.matched_node_orig_ids_list:
            # If no matches yet, MSE is effectively infinite, so reward is very small (or 0)
            return 0.0

        current_dist_counts = {cat: 0 for cat in self.desired_dist_map.keys()}
        total_matched_count = 0

        # Get the appropriate reverse map for the chosen feature
        reverse_map = FEATURE_REVERSE_MAP_LOOKUP.get(self.feature_name)
        
        # Iterate through matched nodes to build current distribution
        for orig_id in self.matched_node_orig_ids_list:
            node_data = current_timeline_entry["node_data_map"].get(orig_id)
            if node_data:
                # Extract the feature value using the feature_name as key
                feature_value_str = node_data.get(self.feature_name)
                
                if feature_value_str in current_dist_counts:
                    current_dist_counts[feature_value_str] += 1
                    total_matched_count += 1
            
        if total_matched_count == 0:
            return 0.0 # No relevant matched nodes yet, so reward is 0.0

        # Calculate current distribution
        current_distribution = {cat: count / total_matched_count for cat, count in current_dist_counts.items()}

        # Calculate MSE
        mse = 0.0
        for category, desired_prob in self.desired_dist_map.items():
            current_prob = current_distribution.get(category, 0.0) # Get 0 if category not seen in matched nodes
            mse += (desired_prob - current_prob)**2
        mse /= len(self.desired_dist_map) # Divide by number of categories for mean

        if mse < self.r_f_epsilon: # Treat very small MSE as zero to prevent inf reward
            return 1.0 - self.r_f_epsilon # Return a reward close to 1 if perfect or near-perfect match
        return 1.0 - (mse) # Removed epsilon from denominator as it's handled by the `if mse < epsilon` check


    def step(self, action):
        if self.done:
            raise RuntimeError("Environment finished; call reset().")

        entry = self.timeline[self.current_index]
        state_data = mask_data(entry["pyg_full"], self.matched_nodes)
        ei = state_data.edge_index

        # Build unique undirected edges
        if ei.numel() == 0:
            edges_list = []
        else:
            u_all = ei[0].tolist()
            v_all = ei[1].tolist()
            pairs = set()
            for u, v in zip(u_all, v_all):
                if u < v:
                    pairs.add((u, v))
            edges_list = list(pairs)

        base_reward_from_match = 0 # Will be 1 if a match happens, 0 otherwise
        current_reward_for_departure = 0 # -1 if a departure occurs, 0 otherwise

        # if action == 0: # No-op action
            # num_dep = sum(1 for nid in entry["departures"] if nid not in self.matched_nodes)
            # # if num_dep == 1:
            # #     current_reward_for_departure = -1
        if action != 0: # Attempt to make a match
            idx = action - 1
            if idx < 0 or idx >= len(edges_list):
                raise ValueError("Invalid action index.")
            u_loc, v_loc = edges_list[idx]
            orig_ids = entry["pyg_full"].orig_ids
            mask_list = [nid not in self.matched_nodes for nid in orig_ids]
            valid_orig_ids = [nid for nid, m in zip(orig_ids, mask_list) if m]
            ou = valid_orig_ids[u_loc]
            ov = valid_orig_ids[v_loc]
            if ou not in self.matched_nodes and ov not in self.matched_nodes:
                self.matched_nodes.add(ou)
                self.matched_nodes.add(ov)
                self.matched_node_orig_ids_list.extend([ou, ov]) # Add to list for r_f tracking
                base_reward_from_match = 1

        # Calculate final reward based on chosen type
        if self.reward_type == "simple":
            # r_e: +1 for a match, +0 for no matches (includes departures)
            reward = self._calculate_r_e(base_reward_from_match, current_reward_for_departure)
        elif self.reward_type == "feature_dist":
            # r_f: 1/MSE of desired distribution
            reward = self._calculate_r_f(entry) # Pass the current timeline entry for node data map
        elif self.reward_type == "mixed":
            # r_m = a*r_e + b*r_f
            r_e_val = self._calculate_r_e(base_reward_from_match, current_reward_for_departure)
            r_f_val = self._calculate_r_f(entry)
            reward = self.reward_a * r_e_val + self.reward_b * r_f_val
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        self.current_index += 1
        if self.current_index >= self.timesteps:
            self.done = True

        nxt = None if self.done else self._get_state()
        return nxt, reward, self.done, {}

# ------------------------------------------------------------------
# 5. DQN Components
# ------------------------------------------------------------------

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class GNNQNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim).to(device)
        self.conv2 = GCNConv(hidden_dim, hidden_dim).to(device)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, 1)
        ).to(device)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, 1)
        ).to(device)
        # Store hidden_dim for potential use in handling empty graphs
        self.hidden_dim = hidden_dim


    def forward(self, data):
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        # FIX: Handle empty graphs to prevent the IndexError in GCNConv
        if data.num_nodes == 0:
            # The output of this layer should be node embeddings. For zero nodes,
            # this is an empty tensor with the correct feature dimension.
            return torch.empty((0, self.hidden_dim), device=x.device)

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch.to(device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        return h

    def compute_q_values(self, data):
        h = self.forward(data)
        edge_index = data.edge_index.to(device)

        if edge_index.numel() == 0:
            edges_list = []
        else:
            u_all = edge_index[0].tolist()
            v_all = edge_index[1].tolist()
            pairs = set()
            for u, v in zip(u_all, v_all):
                if u < v:
                    pairs.add((u, v))
            edges_list = list(pairs)

        if hasattr(data, "batch") and data.batch is not None:
            global_emb = global_mean_pool(h, data.batch.to(device))
        elif data.num_nodes > 0: # Handle case with nodes but no batch
            global_emb = h.mean(dim=0).unsqueeze(0) # Ensure it's (1, hidden_dim) for global_mlp
        else: # Empty graph, global_emb should also be empty or a default
            global_emb = torch.zeros(1, self.hidden_dim, device=device) # Or an empty tensor if global_mlp handles it


        q_noop = self.global_mlp(global_emb)

        q_edges = []
        for (u_loc, v_loc) in edges_list:
            feat = torch.cat([h[u_loc], h[v_loc]], dim=0)
            q_uv = self.edge_mlp(feat)
            q_edges.append(q_uv)
        if q_edges:
            q_edges = torch.cat(q_edges, dim=0)
        else:
            q_edges = torch.zeros(0, device=device)

        return q_noop.view(-1), q_edges

class DQNAgent:
    def __init__(self, in_dim, hidden_dim=64, edge_dim=32,
                 lr=1e-3, gamma=0.99, buffer_cap=10000,
                 batch_size=64, target_update=100):
        self.device = device
        self.policy_net = GNNQNetwork(in_dim, hidden_dim, edge_dim).to(device)
        self.target_net = GNNQNetwork(in_dim, hidden_dim, edge_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

        self.buffer = ReplayBuffer(buffer_cap)
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0

    def select_action(self, data, eps):
        # FIX: Handle the case where the graph is empty.
        # If there are no nodes, the only possible action is 0 (no-op).
        if data.num_nodes == 0:
            return 0

        data = data.to(device)
        edge_index = data.edge_index
        if edge_index.numel() == 0:
            num_edges = 0
        else:
            u_all = edge_index[0].tolist()
            v_all = edge_index[1].tolist()
            pairs = set()
            for u, v in zip(u_all, v_all):
                if u < v:
                    pairs.add((u, v))
            num_edges = len(pairs)

        if random.random() < eps:
            return random.randint(0, num_edges)

        with torch.no_grad():
            q_noop, q_edges = self.policy_net.compute_q_values(data)
            if q_edges.numel() > 0:
                q_all = torch.cat([q_noop, q_edges], dim=0)
            else:
                q_all = q_noop
            return q_all.argmax().item()

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return 0.0 # Return 0.0 loss if not enough samples

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        q_vals = []
        q_targs = []
        for s, a, r, ns, d in transitions:
            s = s.to(device)
            # Ensure batch tensor is present even for single graphs
            # This is crucial for GNNConv to correctly interpret the dimensions of 'x'
            if s.num_nodes == 0: # Handle empty graph case
                q_sa = torch.tensor([0.0], device=device) # Assign a default Q-value for empty graphs
            else:
                if not hasattr(s, 'batch') or s.batch is None:
                    s.batch = torch.zeros(s.num_nodes, dtype=torch.long, device=device)
                q_noop, q_edges = self.policy_net.compute_q_values(s)

                # If no edges or invalid index, use q_noop. Otherwise unsqueeze(0).
                if a == 0 or q_edges.numel() == 0 or (a - 1) >= q_edges.numel():
                    q_sa = q_noop
                else:
                    q_sa = q_edges[a - 1].unsqueeze(0)


            if d or ns is None:
                target = torch.tensor([r], device=device)
            else:
                ns_data = ns.to(device)
                # Ensure batch tensor is present for next state data
                if ns_data.num_nodes == 0: # Handle empty next graph case
                    max_qn = torch.tensor([0.0], device=device) # Assign a default Q-value for empty graphs
                else:
                    if not hasattr(ns_data, 'batch') or ns_data.batch is None:
                        ns_data.batch = torch.zeros(ns_data.num_nodes, dtype=torch.long, device=device)

                    with torch.no_grad():
                        # DDQN modification: Use policy net to select action for next state
                        qn_noop_policy, qn_edges_policy = self.policy_net.compute_q_values(ns_data)
                        if qn_edges_policy.numel() > 0:
                            qn_all_policy = torch.cat([qn_noop_policy, qn_edges_policy], dim=0)
                            next_action_from_policy = qn_all_policy.argmax().item()
                        else:
                            next_action_from_policy = 0 # If no edges, only no-op is an option

                        # Use target net to evaluate the selected action
                        qn_noop_target, qn_edges_target = self.target_net.compute_q_values(ns_data)
                        if next_action_from_policy == 0:
                            max_qn = qn_noop_target
                        elif qn_edges_target.numel() > 0 and (next_action_from_policy - 1) < qn_edges_target.numel():
                            max_qn = qn_edges_target[next_action_from_policy - 1]
                        else: # This case should ideally not be hit if policy action is valid
                            max_qn = qn_noop_target # Fallback to no-op if somehow an invalid edge action was chosen

                target = torch.tensor([r], device=device) + self.gamma * max_qn

            q_vals.append(q_sa)
            q_targs.append(target)

        # Handle cases where q_vals or q_targs might be empty due to all empty graphs
        if not q_vals:
            return 0.0 # Return 0.0 loss if no valid Q-values

        q_vals = torch.cat(q_vals).unsqueeze(1) # Ensure q_vals has a consistent shape
        q_targs = torch.cat(q_targs).detach().unsqueeze(1) # Ensure q_targs has a consistent shape
        
        # Filter out NaN/inf values if they appear (shouldn't if previous fixes work)
        valid_indices = ~(torch.isnan(q_vals) | torch.isinf(q_vals) | torch.isnan(q_targs) | torch.isinf(q_targs))
        if valid_indices.sum() == 0:
            print("Warning: All Q-values or targets are NaN/Inf. Skipping optimization step.")
            return 0.0 # Return 0.0 loss if all are invalid
        
        q_vals = q_vals[valid_indices]
        q_targs = q_targs[valid_indices]

        # If after filtering, no valid samples left
        if q_vals.numel() == 0:
            print("Warning: No valid Q-value pairs after filtering NaN/Inf. Skipping optimization step.")
            return 0.0 # Return 0.0 loss if no valid samples

        loss = nn.MSELoss()(q_vals, q_targs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item() # Return the loss value

# ------------------------------------------------------------------
# 6. Training Loop with TensorBoard Logging
# ------------------------------------------------------------------
def train_dqn(num_episodes, eps_start, eps_end, eps_decay, log_dir,
              lr, gamma, buffer_cap, batch_size, target_update,
              hidden_dim, edge_dim,
              reward_type, feature_name, desired_dist_str, reward_a, reward_b):
    # Construct a more informative log directory name with timestamp
    current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_full = f"{log_dir}_ts{TIMESTEPS}_ep{num_episodes}_lr{lr:.0e}_bs{batch_size}_rew-{reward_type}_{current_time_str}"
    writer = SummaryWriter(log_dir=log_dir_full)

    node_feature_dim = 11 # This remains fixed based on your preprocessing
    agent = DQNAgent(in_dim=node_feature_dim,
                     hidden_dim=hidden_dim,
                     edge_dim=edge_dim,
                     lr=lr,
                     gamma=gamma,
                     buffer_cap=buffer_cap,
                     batch_size=batch_size,
                     target_update=target_update)

    eps = eps_start
    global_step = 0

    for ep in range(num_episodes):
        timeline = simulate_kpd_network_timesteps(timesteps=TIMESTEPS)
        # Pass reward function parameters to KPDEnv
        env = KPDEnv(timeline, reward_type, feature_name, desired_dist_str, reward_a, reward_b)
        
        state = env.reset()
        total_reward = 0
        ep_steps = 0
        episode_total_loss = 0.0 # Initialize total loss for the episode
        episode_optim_steps = 0  # Initialize optimization steps for the episode
        
        # Calculate total potential arrivals for match rate calculation
        total_arrivals = sum(len(entry["arrivals"]) for entry in timeline)


        while True:
            if state is None:
                break
            data = state.to(device)
            # Ensure data.batch is always set for the state before passing to agent.select_action
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

            action = agent.select_action(data, eps)
            nxt, reward, done, _ = env.step(action)
            total_reward += reward
            
            writer.add_scalar("Reward/Step", reward, global_step)
            global_step += 1

            agent_loss = agent.optimize_model() # Get the loss from optimize_model
            if agent_loss > 0.0: # Only accumulate if a meaningful loss was computed
                episode_total_loss += agent_loss
                episode_optim_steps += 1

            agent.buffer.push(state, action, reward, nxt, done)

            state = nxt
            ep_steps += 1
            if done:
                # Calculate matches_made based on the final state of matched_nodes
                # Each match involves two nodes.
                matches_made = len(env.matched_nodes) // 2
                break

        eps = max(eps_end, eps * eps_decay)
        match_rate = matches_made / total_arrivals if total_arrivals > 0 else 0.0
        
        # Calculate average loss for the episode
        average_episode_loss = episode_total_loss / episode_optim_steps if episode_optim_steps > 0 else 0.0

        writer.add_scalar("Reward/Episode", total_reward, ep)
        writer.add_scalar("MatchRate/Episode", match_rate, ep)
        writer.add_scalar("Matches/Episode", matches_made, ep)
        writer.add_scalar("Arrivals/Episode", total_arrivals, ep)
        writer.add_scalar("Epsilon/Episode", eps, ep)
        writer.add_scalar("Steps/Episode", ep_steps, ep)
        writer.add_scalar("Loss/Episode", average_episode_loss, ep) # Log average loss per episode

        print(
            f"Episode {ep+1}/{num_episodes}, "
            f"Total Reward: {total_reward}, "
            f"Matches: {matches_made}, Arrivals: {total_arrivals}, "
            f"Match Rate: {match_rate:.4f}, Epsilon: {eps:.3f}, "
            f"Avg Loss: {average_episode_loss:.4f}" # Print average loss
        )

    writer.close()
    return agent

# ------------------------------------------------------------------
# 7. Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    trained_agent = train_dqn(
        num_episodes=args.num_episodes,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        log_dir=args.log_dir,
        lr=args.lr,
        gamma=args.gamma,
        buffer_cap=args.buffer_cap,
        batch_size=args.batch_size,
        target_update=args.target_update,
        hidden_dim=args.hidden_dim,
        edge_dim=args.edge_dim,
        reward_type=args.reward_type, # New
        feature_name=args.feature_name, # New
        desired_dist_str=args.desired_dist, # New
        reward_a=args.reward_a, # New
        reward_b=args.reward_b # New
    )