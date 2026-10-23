encoder_t = MLP()
m1_t, _ = MultiHeadsAttModel(l=neighbors)
m2_t, _ = MultiHeadsAttModel(l=neighbors)
q_net_t = Q_Net(action_dim = action_space)
In_t= []
for j in range(n_data):
	In_t.append(Input(shape=[len_feature]))
	In_t.append(Input(shape=(neighbors,n_data)))
In_t.append(Input(shape=(1,neighbors)))

feature_t = []
for j in range(n_data):
	feature_t.append(encoder_t(In_t[j*2]))

feature_t_ = Concatenate(axis=1)(feature_t)

relation1_t = []
for j in range(n_data):
	T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In_t[j*2+1],feature_t_])
	relation1_t.append(m1_t([T,T,T,In_t[n_data*2]]))

relation1_t_ = Concatenate(axis=1)(relation1_t)

relation2_t = []
for j in range(n_data):
	T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In_t[j*2+1],relation1_t_])
	relation2_t.append(m2_t([T,T,T,In_t[n_data*2]]))

V_t = []
for j in range(n_data):
	V_t.append(q_net_t([feature_t[j],relation1_t[j],relation2_t[j]]))

model_t = Model(input=In_t,output=V_t)
