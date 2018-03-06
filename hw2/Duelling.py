import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import argparse
import random
import gym
import sys
from collections import deque
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model


env = gym.make('MountainCar-v0')
state_space=env.observation_space.shape[0]
action_s=env.action_space.n

#Hyperparameters
learning_rate=0.001
episodes=1000000
epsilon_start=0.5
epsilon_end=0.05
#decay=(epsilon_start-epsilon_end)/100000
decay = 0.9
batch_size=32
max_steps=200
gamma=1.0
hidden_layer=50


class QNetwork():
	def __init__(self,learning_rate,action_space,input_dim):
		# self.model= Sequential()
		# self.model.add(Dense(units=30,activation='relu',input_dim=state_space,kernel_initializer='he_uniform'))
		# self.model.add(Dense(units=30,activation='relu',kernel_initializer='he_uniform'))
		# self.model.add(Dense(units=30,activation='relu',kernel_initializer='he_uniform'))
		# self.model.add(Dense(units=action_s,activation='linear',kernel_initializer='he_uniform'))

		self.input  =  Input(shape=(input_dim,))
		self.x=Dense(hidden_layer,activation='relu')(self.input)
		# self.x=keras.layers.BatchNormalization(axis=-1)(self.x)
		self.x=Dense(hidden_layer,activation='relu')(self.x)
		# self.x=keras.layers.BatchNormalization(axis=-1)(self.x)
		self.x=Dense(hidden_layer,activation='relu')(self.x)

		self.value= Dense(1,activation='linear',name='value')(self.x)

		self.value1=self.value
		self.advantage = Dense(action_s,activation='linear',name='advantage')(self.x)
		
		self.advantage_mean = keras.layers.Lambda(lambda x:K.mean(x,axis=-1,keepdims=True))(self.advantage)
		
		self.advantage_mean1 = self.advantage_mean
		# self.value=keras.layers.RepeatVector(2)
		# print('Value',self.value.shape)

		# self.value = keras.layers.Lambda(lambda x:K.equal(x,axis=-1,keepdims=True))(self.value)
		i=1
		while(i<action_s):
			self.value=keras.layers.Lambda(lambda x:K.concatenate(x, axis=-1))([self.value,self.value1])
			self.advantage_mean=keras.layers.Lambda(lambda x:K.concatenate(x,axis=-1))([self.advantage_mean1,self.advantage_mean])
			i+=1
		# print('Adv',self.keras.backend.identity.shape)
		# self.advantage_mean=keras.layers.Lambda(lambda x:K.identity(x))(self.advantage_mean)
		# print('Val1',self.value1.shape)
		self.advantage_subtract_mean = keras.layers.Subtract()([self.advantage,self.advantage_mean])
		# print('Adv su',self.advantage_mean.shape)

		self.added = keras.layers.Add()([self.advantage_subtract_mean,self.value])
		# print("Added",self.added.shape)
		  # equivalent to added = keras.layers.add([x1, x2])
		# self.out = Dense(action_s,activation='linear')(self.added)
		# print("out",self.out.shape)
		self.optimizer=keras.optimizers.Adam(lr=learning_rate)
		self.model = Model(inputs=self.input, outputs=self.added)
		self.model.compile(loss='mse',optimizer=self.optimizer)
		plot_model(self.model, to_file='Duelling2.png')

	def save_model_weights(self, fname):
		self.model.save_weights(fname)

	def load_model(self, model_file):
		self.model.load(model_file)

	def load_model_weights(self,fname):
		self.model.load_weights(fname)

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False):
		self.env = environment_name
		self.net=QNetwork(learning_rate,action_s,state_space)
		self.prev_net=QNetwork(learning_rate,action_s,state_space)
		self.prev_net.model.set_weights(self.net.model.get_weights())
		self.q_values=np.zeros([batch_size,action_s])
		self.memory=Replay_Memory()
		self.burn_in_memory()


	def epsilon_greedy_policy(self, q_values,epsilon):
		if (epsilon>np.random.random()):
			action=random.randrange(action_s)
		else:
			action=np.argmax(q_values[0])
		return action		

	def greedy_policy(self, q_values):
		action=np.argmax(q_values)
		return action

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		epsilon = epsilon_start

		for i in range(1000000):
			state = env.reset()
			state=np.reshape(state,[1,state_space])
			total_reward=0
		    
			step=0
			while step<max_steps:
				env.render()
				step+=1
				q_values = self.net.model.predict(state)
				action=self.epsilon_greedy_policy(q_values,epsilon)
				new_state,reward,done, _ = env.step(action)
				new_state=np.reshape(new_state,[1,state_space])
				self.memory.append([state,action,reward,done,new_state])

				minibatch=self.memory.sample_batch()
				batch_states=np.zeros((batch_size,state_space))
				batch_next_states=np.zeros((batch_size,state_space))

				t_int=0
				for batch_state, batch_action, batch_reward, batch_done, batch_new_state in minibatch:
					batch_states[t_int]=batch_state
					batch_next_states[t_int]=batch_new_state
					t_int+=1

				batch_q_values=self.net.model.predict(batch_states)
				batch_prev_q_values=self.prev_net.model.predict(batch_next_states)

				t_int=0
				for batch_state, batch_action, batch_reward, batch_done, batch_new_state in minibatch:
					if batch_done:
						temp=0
					else: 
						temp=gamma*(np.amax(batch_prev_q_values[t_int]))
					batch_q_values[t_int][batch_action] = batch_reward+temp
					t_int+=1

				self.net.model.fit(batch_states,batch_q_values,batch_size=batch_size,epochs=1,verbose=0)

				epsilon*=decay
				if epsilon<epsilon_end:
					epsilon = epsilon_end
				total_reward+=reward
				state=new_state

				if done:
					break
			self.prev_net.model.set_weights(self.net.model.get_weights())
			print(i,total_reward)

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		pass 

	def burn_in_memory(self):
		state = env.reset()
		state=np.reshape(state,[1,state_space])
		for i in range(self.memory.burn_in):
			action=random.randrange(action_s)
			new_state, reward, done, _ = env.step(action)
			new_state=np.reshape(new_state,[1,state_space])
			self.memory.append([state,action,reward,done,new_state])
			state=new_state
			if done:
				state=env.reset()
				state=np.reshape(state,[1,state_space])


class Replay_Memory():
	def __init__(self, memory_size=10000, burn_in=5000):
		self.transitions =[]
		self.memory_size=memory_size
		self.burn_in = burn_in

	def sample_batch(self, batch_size=32):
		return random.sample(self.transitions,batch_size)

	def append(self, transition):
		if(len(self.transitions)<self.memory_size):
			self.transitions.append(transition)
		else:
			idx=random.randint(1,self.memory_size-1)
			# print(idx)
			del self.transitions[idx]
			self.transitions.append(transition)


def parse_arguments():
	parser = argparse.ArgumentParser(description='Linear Q network parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)
	agent=DQN_Agent(environment_name)
	# print(agent)

	DQN_Agent.train(agent)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)
