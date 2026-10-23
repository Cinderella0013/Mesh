Inr= []
for j in range(n_data):
    Inr.append(Input(shape=[len_feature]))
    Inr.append(Input(shape=(neighbors,n_data)))
Inr.append(Input(shape=(1,neighbors)))
featurer = []
for j in range(n_data):
    featurer.append(encoder(Inr[j*2]))

featurer_ = Concatenate(axis=1)(featurer)

relationr1 = []
r2=[]
for j in range(n_data):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([Inr[j*2+1],featurer_])
    relationr1.append(m1([T,T,T,Inr[n_data*2]]))

relationr1_ = Concatenate(axis=1)(relationr1)

for j in range(n_data):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([Inr[j*2+1],relationr1_])
    r2.append(m2_r([T,T,T,Inr[n_data*2]]))

model_r=Model(input=Inr,output=r2)
