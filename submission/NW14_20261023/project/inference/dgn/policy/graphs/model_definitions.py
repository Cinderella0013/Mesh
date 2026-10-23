from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras.layers.core import Activation
from keras.utils import np_utils,to_categorical
from keras.engine.topology import Layer
def MLP():

	In_0 = Input(shape=[len_feature])
	h = Dense(128, activation='relu',kernel_initializer='random_normal')(In_0)
	h = Dense(128, activation='relu',kernel_initializer='random_normal')(h)
	h = Reshape((1,128))(h)
	model = Model(input=In_0,output=h)
	return model
def MultiHeadsAttModel(l=2, d=128, dv=16, dout=128, nv = 8 ):

	v1 = Input(shape = (l, d))
	q1 = Input(shape = (l, d))
	k1 = Input(shape = (l, d))
	ve = Input(shape = (1, l))

	v2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(v1)
	q2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(q1)
	k2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(k1)

	v = Reshape((l, nv, dv))(v2)
	q = Reshape((l, nv, dv))(q2)
	k = Reshape((l, nv, dv))(k2)
	v = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(v)
	k = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(k)
	q = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(q)

	att = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True) / np.sqrt(dv))([q,k])# l, nv, nv
	att_ = Lambda(lambda x: K.softmax(x))(att)
	out = Lambda(lambda x: tf.matmul(x[0], x[1]))([att, v])
	out = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(out)

	out = Reshape((l, dv*nv))(out)

	T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([ve,out])

	out = Dense(dout, activation = "relu",kernel_initializer='random_normal')(T)
	model = Model(inputs=[q1,k1,v1,ve], outputs=out)
	model_ = Model(inputs=[q1,k1,v1,ve], outputs=att_)
	return model,model_
def Q_Net(action_dim):

	I1 = Input(shape = (1, 128))
	I2 = Input(shape = (1, 128))
	I3 = Input(shape = (1, 128))

	h1 = Flatten()(I1)
	h2 = Flatten()(I2)
	h3 = Flatten()(I3)

	h = Concatenate()([h1,h2,h3])
	V = Dense(action_dim,kernel_initializer='random_normal')(h)

	model = Model(input=[I1,I2,I3],output=V)
	return model
