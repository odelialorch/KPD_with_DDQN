import random
import networkx as nx
import numpy as np

# Precompute arrival λ and departure probability (unchanged)
DAILY_ARRIVAL = 10
TIMESTEPS_PER_DAY = 8000/365.0
ARRIVAL_LAMBDA = DAILY_ARRIVAL / TIMESTEPS_PER_DAY
AVG_DURATION_DAYS = 200
AVG_DURATION_TIMESTEPS = AVG_DURATION_DAYS * TIMESTEPS_PER_DAY
DEPART_PROB = 1.0 / AVG_DURATION_TIMESTEPS

# Number of distinct types for KPD pairs
TYPES_COUNT = 128

# Distributions (unchanged)
PATIENT_GENDER_DIST = {"Male": 0.502, "Female": 0.498}
DONOR_GENDER_DIST = {"Male": 0.377, "Female": 0.623}
PATIENT_RACE_DIST = {"White":0.608, "Black":0.182, "Hispanic":0.115, "Asian":0.084, "Other":0.011}
DONOR_RACE_DIST = {"White":0.70, "Black":0.105, "Hispanic":0.101, "Asian":0.084, "Other":0.010}
PATIENT_BLOOD_DIST = {"O":0.50, "A":0.30, "B":0.15, "AB":0.05}
DONOR_BLOOD_DIST = {"O":0.398, "A":0.319, "B":0.192, "AB":0.091}
CPRA_CATEGORIES = [(0,20,0.549),(21,80,0.235),(81,98,0.142),(99,100,0.075)]
PATIENT_HEALTH_DIST = {"Good":0.50, "Fair":0.35, "Poor":0.15}
PATIENT_AGE_BUCKETS = [(18,29,0.085),(30,44,0.291),(45,59,0.425),(60,74,0.171),(75,85,0.028)]
DONOR_AGE_BUCKETS = [(18,29,0.119),(30,44,0.443),(45,59,0.335),(60,85,0.136)]
HLA_ANTIGENS = ["A1","A2","A3","A11","B7","B8","B27","B44","DR1","DR4","DR7","DR15","DQ2","DQ6","DQ8"]

# Pre-extract keys & weights
_PAT_KEYS, _PAT_WTS = zip(*PATIENT_BLOOD_DIST.items())
_DON_KEYS, _DON_WTS = zip(*DONOR_BLOOD_DIST.items())
_PG_KEYS, _PG_WTS = zip(*PATIENT_GENDER_DIST.items())
_DG_KEYS, _DG_WTS = zip(*DONOR_GENDER_DIST.items())
_PR_KEYS, _PR_WTS = zip(*PATIENT_RACE_DIST.items())
_DR_KEYS, _DR_WTS = zip(*DONOR_RACE_DIST.items())
_PH_KEYS, _PH_WTS = zip(*PATIENT_HEALTH_DIST.items())


def weighted_choice(keys, weights):
    """Fast weighted choice"""
    return random.choices(keys, weights=weights, k=1)[0]

def sample_age(buckets):
    probs = [b[2] for b in buckets]
    idx = random.choices(range(len(buckets)), weights=probs, k=1)[0]
    low, high, _ = buckets[idx]
    return random.randint(low, high)


def sample_cpra():
    cprs, wts = zip(*[((low,high),prob) for (low,high,prob) in CPRA_CATEGORIES])
    (low,high) = random.choices(cprs, weights=wts, k=1)[0]
    return random.randint(low, high)


def sample_hla_antigens():
    return random.sample(HLA_ANTIGENS, random.choice([2,3,4]))


def sample_patient_antibodies(cpra_score):
    num = int((cpra_score/100)*len(HLA_ANTIGENS))
    return random.sample(HLA_ANTIGENS, num) if num>0 else []


def blood_type_compatible(donor_bt, patient_bt):
    if donor_bt=="O": return True
    if donor_bt=="A": return patient_bt in ("A","AB")
    if donor_bt=="B": return patient_bt in ("B","AB")
    return patient_bt=="AB"


def create_node(node_id):
    """Generate a donor-patient pair with a true 'type' attribute."""
    # Sample attributes as before
    pg = weighted_choice(_PG_KEYS, _PG_WTS)
    dg = weighted_choice(_DG_KEYS, _DG_WTS)
    pr = weighted_choice(_PR_KEYS, _PR_WTS)
    dr = weighted_choice(_DR_KEYS, _DR_WTS)
    pa = sample_age(PATIENT_AGE_BUCKETS)
    da = sample_age(DONOR_AGE_BUCKETS)
    pat_bt = weighted_choice(_PAT_KEYS, _PAT_WTS)
    don_bt = weighted_choice(_DON_KEYS, _DON_WTS)
    cpra = sample_cpra()
    health = weighted_choice(_PH_KEYS, _PH_WTS)
    antigens = sample_hla_antigens()
    antibodies = sample_patient_antibodies(cpra)
    if not blood_type_compatible(don_bt, pat_bt):
        inc = "ABO"
    elif set(antibodies) & set(antigens):
        inc = "HLA"
    else:
        inc = "Compatible"

    # Embed true type
    type_idx = node_id % TYPES_COUNT

    return {
        'id': node_id,
        'type': type_idx,
        'patient_gender': pg,
        'donor_gender': dg,
        'patient_race': pr,
        'donor_race': dr,
        'patient_age': pa,
        'donor_age': da,
        'patient_blood': pat_bt,
        'donor_blood': don_bt,
        'incompat_reason': inc,
        'patient_health': health,
        'patient_cpra': cpra,
        'patient_antibodies': antibodies,
        'donor_antibodies': [],
        'donor_antigens': antigens
    }


def simulate_kpd_network_timesteps(timesteps=8000):
    lam = ARRIVAL_LAMBDA
    depart_p = DEPART_PROB

    active = {}
    node_counter = 1
    timeline = []

    for t in range(timesteps):
        arrivals = []
        for _ in range(np.random.poisson(lam)):
            nd = create_node(node_counter)
            nd['entry_timestep'] = t
            active[node_counter] = nd
            arrivals.append(node_counter)
            node_counter += 1

        to_remove = []
        for nid in list(active):
            if random.random() < depart_p:
                active[nid]['exit_timestep'] = t
                to_remove.append(nid)
        for nid in to_remove:
            del active[nid]
        departures = to_remove

        ids = list(active.keys())
        attrs = [active[i] for i in ids]

        edge_list = []
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                u, v = attrs[i], attrs[j]
                if blood_type_compatible(u['donor_blood'], v['patient_blood']) and \
                   blood_type_compatible(v['donor_blood'], u['patient_blood']) and \
                   not (set(u['donor_antigens']) & set(v['patient_antibodies'])) and \
                   not (set(v['donor_antigens']) & set(u['patient_antibodies'])):
                    edge_list.append((ids[i], ids[j]))

        G = nx.Graph()
        for nid in ids:
            G.add_node(nid, **active[nid])
        if edge_list:
            G.add_edges_from(edge_list)
        timeline.append({'timestep': t, 'graph': G, 'arrivals': arrivals, 'departures': departures})

    return timeline
