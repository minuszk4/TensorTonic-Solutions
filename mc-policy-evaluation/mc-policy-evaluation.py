def mc_policy_evaluation(episodes, gamma, n_states):
    returns_sum = [0.0] * n_states
    returns_count = [0] * n_states
    
    for episode in episodes:
        first_visit = {}
        for t, (state, _) in enumerate(episode):
            if state not in first_visit:
                first_visit[state] = t
        
        for state, t in first_visit.items():
            G = 0
            power = 1
            for k in range(t, len(episode)):
                _, reward = episode[k]
                G += power * reward
                power *= gamma
            
            returns_sum[state] += G
            returns_count[state] += 1
    
    V = [0.0] * n_states
    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = returns_sum[s] / returns_count[s]
    
    return V