import torch
import json
import re
import pandas as pd
from torch_geometric.data import Data

# ---- Set device at top ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------- Node Feature Encoding ---------------
def one_hot(val, choices):
    vec = torch.zeros(len(choices), device=device)
    if val in choices:
        vec[choices.index(val)] = 1.0
    return vec

def encode_node_features(node):
    patient_gender = one_hot(node['patient_gender'], ['M', 'F'])
    patient_race = one_hot(node['patient_race'], ['White','Black','Hispanic','Asian','Other'])
    patient_age = torch.tensor([(node['patient_age'] - 18) / (75 - 18)], device=device)
    health = one_hot(node['patient_health'], ['Good', 'Fair', 'Poor'])
    patient_blood = one_hot(node['patient_blood'], ['O', 'A', 'B', 'AB'])
    cpra = torch.tensor([node['patient_cpra'] / 100], device=device)
    incompat = one_hot(node['incompat_reason'], ['ABO','HLA','Both'])
    donor_gender = one_hot(node['donor_gender'], ['M', 'F'])
    donor_race = one_hot(node['donor_race'], ['White','Black','Hispanic','Asian','Other'])
    donor_age = torch.tensor([(node['donor_age'] - 18) / (75 - 18)], device=device)
    donor_blood = one_hot(node['donor_blood'], ['O', 'A', 'B', 'AB'])
    return torch.cat([
        patient_gender, patient_race, patient_age, health, patient_blood,
        cpra, incompat, donor_gender, donor_race, donor_age, donor_blood
    ])  # (31,)

# --------------- Parsing Helper ---------------
def parse_active_nodes(node_str):
    return [int(s.split('(')[0]) for s in node_str.split(';') if s.strip()]

def parse_active_edges(edge_str):
    return [tuple(map(int, re.findall(r'\d+', s))) for s in edge_str.split(';') if s.strip()]

# --------------- Mapping Node IDs to Feature Dicts ---------------
def build_node_map(json_file):
    with open(json_file, 'r') as f:
        node_map = json.load(f)
    node_map = {int(k): v for k, v in node_map.items()}
    return node_map

# --------------- Main Pipeline ---------------
def build_graph_sequence(csv_file, node_map, move_to_gpu=True):
    df = pd.read_csv(csv_file)
    df['ActiveEdges'] = df['ActiveEdges'].fillna("")
    graph_sequence = []
    for idx, row in df.iterrows():
        node_ids = parse_active_nodes(row['ActiveNodes'])
        edge_pairs = parse_active_edges(row['ActiveEdges'])
        node_features = []
        real_ids = []
        for node_id in node_ids:
            if node_id in node_map:
                node_features.append(encode_node_features(node_map[node_id]))
                real_ids.append(node_id)
        if not node_features:
            continue
        # Stack directly as torch tensor on the right device
        x = torch.stack(node_features)
        node_id_to_idx = {nid: i for i, nid in enumerate(real_ids)}
        edge_index = []
        feasible_actions = []
        for i, j in edge_pairs:
            if i in node_id_to_idx and j in node_id_to_idx:
                edge_index.append([node_id_to_idx[i], node_id_to_idx[j]])
                feasible_actions.append((node_id_to_idx[i], node_id_to_idx[j]))
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        else:
            edge_index = torch.empty((2,0), dtype=torch.long, device=device)
        data = Data(x=x, edge_index=edge_index)
        # Optionally ensure everything is on GPU (PyG Data can be moved with .to(device))
        if move_to_gpu:
            data = data.to(device)
        time = float(row['Time'])
        graph_sequence.append((data, feasible_actions, time))
    return graph_sequence

# ----------- Example usage -----------
# node_map = build_node_map("node_map.json")
# graph_sequence = build_graph_sequence("kidney_exchange_sim.csv", node_map)
# Now graph_sequence is ready for RL and all tensors are already on GPU!
