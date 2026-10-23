for k in range(len(batch)):
	for j in range(n_data):
		if dones[k][j]:
			q_values[j][k][actions[k][j]] = rewards[k][j]
		else:
			q_values[j][k][actions[k][j]] = rewards[k][j] + GAMMA*np.max(target_q_values[j][k])
