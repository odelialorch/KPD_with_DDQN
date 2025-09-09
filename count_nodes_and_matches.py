import pandas as pd
import re

def count_matches(csv_file):
    df = pd.read_csv(csv_file)
    # Assuming the Event column has the word 'Match' for match events
    num_matches = (df['Event'] == 'Match').sum()
    print(f"Total number of matches in 1 year: {num_matches}")
    return num_matches

def count_arrivals_from_generate_data(csv_file):
    df = pd.read_csv(csv_file)
    # Find rows where Event column is 'arrival' (case-insensitive)
    arrivals = df[df['Event'].str.lower() == 'arrival']
    
    # If you have a unique NodeID column:
    if 'NodeID' in arrivals.columns:
        unique_nodes = arrivals['NodeID'].nunique()
    else:
        # Otherwise, parse Node IDs from ActiveNodes column
        unique_ids = set()
        for s in arrivals['ActiveNodes']:
            # Each ActiveNodes entry might have one or more nodes, but for arrival events, usually just one
            ids = [int(sub.split('(')[0]) for sub in str(s).split(';') if sub.strip()]
            unique_ids.update(ids)
        unique_nodes = len(unique_ids)
    
    print(f"Total donor-patient nodes arrived in 1 year: {unique_nodes}")
    return unique_nodes

# Usage:
csv_file = "kidney_exchange_sim.csv"
count_arrivals_from_generate_data(csv_file)
count_matches(csv_file)