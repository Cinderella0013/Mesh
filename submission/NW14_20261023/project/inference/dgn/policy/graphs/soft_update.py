weights = encoder.get_weights()
target_weights = encoder_t.get_weights()
for w in range(len(weights)):
	target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
encoder_t.set_weights(target_weights)

weights = q_net.get_weights()
target_weights = q_net_t.get_weights()
for w in range(len(weights)):
	target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
q_net_t.set_weights(target_weights)

weights = m1.get_weights()
target_weights = m1_t.get_weights()
for w in range(len(weights)):
	target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
m1_t.set_weights(target_weights)

weights = m2.get_weights()
target_weights = m2_t.get_weights()
for w in range(len(weights)):
	target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
m2_t.set_weights(target_weights)
