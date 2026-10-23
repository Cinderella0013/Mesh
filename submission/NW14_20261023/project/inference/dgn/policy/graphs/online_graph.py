encoder = MLP()
m1, m1_r = MultiHeadsAttModel(l=neighbors)
m2, m2_r = MultiHeadsAttModel(l=neighbors)
q_net = Q_Net(action_dim = action_space)
vec = np.zeros((1,neighbors))
vec[0][0] = 1

In= []
for j in range(n_data):
	In.append(Input(shape=[len_feature]))
	In.append(Input(shape=(neighbors,n_data)))
In.append(Input(shape=(1,neighbors)))
feature = []
for j in range(n_data):
	feature.append(encoder(In[j*2]))

feature_ = Concatenate(axis=1)(feature)

relation1 = []
for j in range(n_data):
	T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In[j*2+1],feature_])
	relation1.append(m1([T,T,T,In[n_data*2]]))

relation1_ = Concatenate(axis=1)(relation1)

relation2 = []
for j in range(n_data):
	T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In[j*2+1],relation1_])
	relation2.append(m2([T,T,T,In[n_data*2]]))

V = []
for j in range(n_data):
	V.append(q_net([feature[j],relation1[j],relation2[j]]))

model = Model(input=In,output=V)
