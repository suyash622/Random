import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
import numpy as np
import argparse
import random
import gym
import time
import sys
from skimage.transform import resize

env = gym.make('SpaceInvaders-v0')
# state_space=env.observation_space.shape
# action_s=env.action_space.n

# print('StateSpac;', state_space)
# print('Action', action_s)
learning_rate= 0.0001

class QNetwork():
	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output.
	def __init__(self,learning_rate,action_space,input_dim):
		self.action_space=action_space
		self.input_dim=input_dim
		self.state_space=input_dim
		self.learning_rate=learning_rate

	def save_model_weights(self, fname):
		self.model.save_weights(fname)

	def load_model(self, model_file):
		self.model.load(model_file)

	def load_model_weights(self,fname):
		self.model.load_weights(fname)

	def create_DQN_Net(self):
		# model=Sequential()
		# model.add(Dense(units=30,activation='relu',input_dim=self.input_dim,kernel_initializer='he_uniform'))
		# model.add(Dense(units=30,activation='relu',kernel_initializer='he_uniform'))
		# model.add(Dense(units=30,activation='relu',kernel_initializer='he_uniform'))
		# model.add(Dense(units=self.action_space,activation='linear',kernel_initializer='he_uniform'))
		print (self.input_dim)
		input = Input(shape=self.input_dim)
		x = keras.layers.Conv2D(16, (8,8), strides=(1, 1), padding='valid')(input)
		x = keras.layers.Conv2D(32, (4,4), strides=(1, 1), padding='valid')(x)
		x = keras.layers.Flatten()(x)
		x=Dense(256,activation='relu')(x)
		# x=Dense(256,activation='relu')(x)
		out=Dense(self.action_space,activation='linear')(x)
		optimizer=keras.optimizers.Adam(lr=learning_rate)
		model=Model(inputs=input, outputs=out)
		model.compile(loss='mse',optimizer=optimizer)
		return model

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self, environment_name, model_type, render=False):
		self.env = gym.make(environment_name)
		self.state_space=(84,84,1,)
		self.action_space=self.env.action_space.n
		self.render=render

		
		self.network=QNetwork(learning_rate,self.action_space,self.state_space)
		self.net=self.network.create_DQN_Net()
			
		self.prev_network=QNetwork(learning_rate,self.action_space,self.state_space)
		self.prev_net=self.prev_network.create_DQN_Net()

		self.prev_net.set_weights(self.net.get_weights())
		self.memory=Replay_Memory()
		x=time.time()
		self.burn_in_memory()
		print("Burn completed in:",time.time()-x)
		print(len(self.memory.transitions))

		
		self.gamma=0.99
		self.episodes=1000000
		
		self.epsilon_start=0.9
		self.epsilon_end=0.2
		self.epsilon_decay = 0.999
		self.batch_size=32
		self.max_steps=150

	def epsilon_greedy_policy(self, q_values,epsilon):
		if (epsilon>np.random.random()):
			action=random.randrange(self.action_space)
		else:
			action=np.argmax(q_values[0])
		return action

	def greedy_policy(self, q_values):
		action=np.argmax(q_values[0])
		return action

	def cvtbw(self,rgb):
		return np.dot(rgb[...,:3],[0.299,0.587,0.144])

	def res(self,bw):

	  return resize(bw, (84, 84,1))
	  # return np.reshape(x,(84,84,1,1))

	def preprocess(self,x):
		x=self.cvtbw(x)
		x = self.res(x)
		x = np.expand_dims(x, axis=0)

		return x


	def train(self,exp_replay=True):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		if(exp_replay==True):
			epsilon = self.epsilon_start
			tracking_size = 20
			prev_reward = np.zeros([tracking_size])

			for i in range(self.episodes):
				state = self.env.reset()													#start with a state
				state=self.preprocess(state)
				print("state_shape: ", state.shape)
				total_reward=0.0
				step=0
				while step<self.max_steps:
					if self.render==True:
						self.env.render()
					step+=1

					q_values = self.net.predict(state)
					print("qvalue_shape: ", q_values.shape)
					action=self.epsilon_greedy_policy(q_values,epsilon)
					print(action)
					new_state,reward,done, _ = self.env.step(action)
					new_state=self.preprocess(new_state)
					total_reward+=reward
					self.memory.append([state,action,reward,done,new_state])

					minibatch=self.memory.sample_batch()
					batch_states=np.zeros((32,84,84,1))
					batch_next_states=np.zeros((32,84,84,1))
					t_int=0
					for batch_state, batch_action, batch_reward, batch_done, batch_new_state in minibatch:
						batch_states[t_int]=batch_state
						batch_next_states[t_int]=batch_new_state
						t_int+=1

					batch_q_values=self.net.predict(batch_states)
					batch_prev_q_values=self.prev_net.predict(batch_next_states)

					t_int=0
					for batch_state, batch_action, batch_reward, batch_done, batch_new_state in minibatch:
						if batch_done:
							temp=0
						else: 
							temp=self.gamma*(np.amax(batch_prev_q_values[t_int]))
						batch_q_values[t_int][batch_action] = batch_reward+temp
						t_int+=1

					self.net.fit(batch_states,batch_q_values,batch_size=self.batch_size,epochs=1,verbose=0)

					epsilon*=self.epsilon_decay
					if epsilon<self.epsilon_end:
						epsilon=self.epsilon_end

					if done:
						break

					state=new_state


				self.prev_net.set_weights(self.net.get_weights())

				#env = gym.wrappers.Monitor(self.env, directory, video_callable=lambda episode_id: episode_id%10==0)
				prev_reward[:-1]=prev_reward[1:]
				prev_reward[-1]=total_reward
				print(i,total_reward,np.mean(prev_reward))
		else:
			epsilon = self.epsilon_start
			tracking_size = 20
			prev_reward = np.zeros([tracking_size])
			
			for i in range(self.episodes):
				state=self.env.reset()
				state=self.preprocess(state)
				# state=np.reshape(state,[1,self.state_space])
				action=random.randrange(self.action_space)

				total_reward=0.0
				step=0
				while(step<self.max_steps):
					step+=1

					if self.render==True:
						self.env.render()
					
					q_values=self.net.predict(state)

					action=self.epsilon_greedy_policy(q_values,epsilon)
					new_state, reward, done, _=self.env.step(action)
					new_state=self.preprocess(new_state)
					total_reward+=reward
					
					if done:
						temp=0
					else:					
						temp=self.gamma*(np.amax(self.net.predict(new_state)[0]))
					
					q_values[0][action]=reward+temp
					self.net.fit(state,q_values,epochs=1,verbose=0)

					epsilon*=self.epsilon_decay
					if epsilon<self.epsilon_end:
						epsilon=self.epsilon_end

					if done:
						break
					
					state=new_state

				prev_reward[:-1]=prev_reward[1:]
				prev_reward[-1]=total_reward
				print(i,total_reward,np.mean(prev_reward))


	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		pass

	def burn_in_memory(self):
		state = self.env.reset()
		state=self.preprocess(state)
		for i in range(self.memory.burn_in):
			action=random.randrange(self.action_space)
			new_state, reward, done, _ = self.env.step(action)
			new_state=self.preprocess(new_state)
			self.memory.append([state,action,reward,done,new_state])
			state=new_state
			if done:
				state=self.env.reset()
				state=self.preprocess(state)


class Replay_Memory():
	def __init__(self, memory_size=1000000, burn_in=500):
		self.transitions =[]
		self.memory_size=memory_size
		self.burn_in = burn_in

	def sample_batch(self, batch_size=32):
		return random.sample(self.transitions,batch_size)

	def append(self, transition):
		if(len(self.transitions)<self.memory_size):
			self.transitions.append(transition)
		else:
			idx=random.randint(0,self.memory_size-1)
			del self.transitions[idx]
			self.transitions.append(transition)


def parse_arguments():
	parser = argparse.ArgumentParser(description='Q network parser')
	parser.add_argument('--env',dest='env',type=str,default='SpaceInvaders-v0')
	parser.add_argument('--render',dest='render',type=bool,default=False)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str,default="DQN")
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name=args.env
	render_status=args.render
	training_status=args.train
	model_type=args.model_file

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 
	agent=DQN_Agent(environment_name,model_type,render_status)
	if training_status==1:
		if model_type=="Linear_wo_exp_replay":
			DQN_Agent.train(agent,exp_replay=False)
		else:
			DQN_Agent.train(agent,exp_replay=True)

if __name__ == '__main__':
	main(sys.argv)