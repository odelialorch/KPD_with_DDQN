# kpd_adp2.py

import os
import argparse
import numpy as np
import networkx as nx
import torch
import cvxpy as cp
from torch.utils.tensorboard import SummaryWriter
from generate_data import simulate_kpd_network_timesteps, blood_type_compatible, create_node

# ----- ARGUMENTS -----
parser = argparse.ArgumentParser()
parser.add_argument("--ts", dest="timesteps", type=int, default=100)
parser.add_argument("--num_samples", type=int, default=50)
args = parser.parse_args()
TIMESTEPS = args.timesteps
NUM_SAMPLES = args.num_samples

# ----- DEVICE -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}, Timesteps: {TIMESTEPS}")

# ----- PARAMETERS -----
DAILY_ARRIVAL = 10
TIMESTEPS_PER_DAY = 8000 / 365.0
ARRIVAL_LAMBDA = DAILY_ARRIVAL / TIMESTEPS_PER_DAY
AVG_DURATION = 200 * TIMESTEPS_PER_DAY
DEPART_PROB = 1.0 / AVG_DURATION

# ----- LOGGING -----
log_dir = f"runs/adp_kpd_samples{args.num_samples}_ts{TIMESTEPS}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# ----- PRECOMPUTE COMPATIBILITY MATRIX -----
TYPES = 128
nodes = [create_node(i) for i in range(TYPES)]
P = np.zeros((TYPES, TYPES))
# Monte Carlo compatibility (reduced trials)
trials = 1000
idx = np.random.randint(0, TYPES, size=(trials,2))
for i,j in idx:
    if i!=j and blood_type_compatible(nodes[i]['donor_blood'], nodes[j]['patient_blood']) \
       and blood_type_compatible(nodes[j]['donor_blood'], nodes[i]['patient_blood']):
        P[i,j] +=1; P[j,i]+=1
P /= trials
mask = np.triu(np.ones((TYPES, TYPES)),1)
P3 = P[:,:,None] * mask[:,:,None]

# ----- OFFLINE ALP -----
def solve_expected_alp():
    lam = np.ones(TYPES)/TYPES * ARRIVAL_LAMBDA
    nu = DEPART_PROB
    # Variables
    x = cp.Variable((TYPES, TYPES, TIMESTEPS), nonneg=True)
    y = cp.Variable((TYPES, TIMESTEPS+1), nonneg=True)
    cons = [y[:,0] == 0]
    for t in range(TIMESTEPS):
        inflow = cp.multiply(y[:,t], 1-nu) + lam
        outflow = cp.sum(x[:,:,t], axis=1)
        cons.append(y[:,t+1] == inflow - outflow)
    # Objective
    obj = cp.Maximize(cp.sum(cp.multiply(P3, x)))
    problem = cp.Problem(obj, cons)
    problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
    # Extract duals
    V = torch.zeros((TIMESTEPS+1, TYPES), device=device)
    # duals of y[:,t+1]== flow constraints are cons[1:] order
    for t in range(TIMESTEPS):
        duals = cons[t+1].dual_value
        V[t,:] = torch.from_numpy(duals)
    V[TIMESTEPS,:] = 0
    return V

# Precompute duals once
V_dual = solve_expected_alp()

# ----- GREEDY POLICY -----
def run_primal_greedy(timeline, V):
    matched = set()
    total = 0
    for t, data in enumerate(timeline, start=1):
        G = data['graph']
        # nodes have 'type' attribute indicating type index
        active = [n for n, attrs in G.nodes(data=True) if n not in matched]
        if not active:
            continue
        H = G.subgraph(active)
        # compute weights using type attributes
        weights = {}
        for u, v in H.edges():
            type_u = G.nodes[u]['type']
            type_v = G.nodes[v]['type']
            weights[(u, v)] = float(V[t, type_u] + V[t, type_v])
        # perform weighted matching
        M = nx.max_weight_matching(H, maxcardinality=False,
                                   weight=lambda u, v, d: weights.get((u, v), 0.0))
        for u, v in M:
            matched.update((u, v))
            total += 1
    return total

# ----- SIMULATION -----
print(f"Running {NUM_SAMPLES} samples...")
arrivals=None; matches=[]
for s in range(NUM_SAMPLES):
    timeline = simulate_kpd_network_timesteps(TIMESTEPS)
    if arrivals is None:
        arrivals = sum(len(e['arrivals']) for e in timeline)
    m = run_primal_greedy(timeline, V_dual)
    matches.append(m)
    rate = m/arrivals
    writer.add_scalar('MatchRate/Episode', rate, s)
    writer.add_scalar("Matches/Episode", m, s)
    print(f"Sample {s+1}: matches={m}, rate={rate:.4f}")
avg = np.mean(matches)
avg_rate = avg/arrivals
print(f"Avg matches: {avg:.1f}, Avg rate: {avg_rate:.4f}")
writer.close()
