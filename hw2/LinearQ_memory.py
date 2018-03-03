import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import argparse
import random
import gym
import sys

env = gym.make('CartPole-v0')

class QNetwork():

	

	def __init__(self,learning_rate,action_space):
		self.model= Sequential()
		self.model.add(Dense(hidden_layer,activation='relu',input_dim=4))
		self.model.add(Dense(hidden_layer,activation='relu'))
		self.model.add(Dense(action_s,activation='linear'))

		self.optimizer=keras.optimizers.Adagrad(lr=learning_rate)
		self.model.compile(loss='mse',optimizer=self.optimizer)
		

	def save_model_weights(self, fname):
		# Helper function to save your model / weights. 
		self.model.save_weights(fname)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		self.model.load(model_file)

	def load_model_weights(self,fname):
		# Helper funciton to load model weights. 
		self.model.load_weights(fname)
		


state_space=4
action_s=2
learning_rate=0.0001
episodes=500000
epsilon_start=0.5
epsilon_end=0.05
decay=(epsilon_start-epsilon_end)/100000
batch_size=1
max_steps=200
gamma=0.99
hidden_layer=10

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

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.env = environment_name
		self.net=QNetwork(learning_rate,action_s)
		self.q_values=np.zeros([batch_size,action_s])

		

	def epsilon_greedy_policy(self, q_values,epsilon):
		# Creating epsilon greedy probabilities to sample from.
		if (epsilon>np.random.random()):
			action=random.randrange(action_s)
		else:
			action=np.argmax(q_values[0])
		return action		
	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		
		action=np.argmax(q_values)
		return action
	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

		memory=Replay_Memory()
		
		for i in range(episodes):
			step=0
			state=env.reset()
			# epsilon=epsilon_start+(epsilon_start-epsilon_end)*np.exp(decay*i)
			# action=self.epsilon_greedy_policy(q_values,epsilon)
			# action=random.randrange(action_s)
			state=np.reshape(state,[1,state_space])
			total_reward=0.0


			while(step<max_steps):
					env.render()
					step+=1
					epsilon=epsilon_start+(epsilon_start-epsilon_end)*np.exp(decay*i)
						
					q_values= self.net.model.predict(state)
					action=self.epsilon_greedy_policy(q_values,epsilon)
					new_state, reward, done, _ = env.step(action)
					new_state = np.reshape(new_state, [1, state_space])
					memory.append([state,action,reward,new_state,done])
					total_reward += reward
					


					if done:
						print ("Cummulative reward: ",total_reward, step)
						break
						
						# target = self.net.model.predict(new_state)[0]
						# target_q=reward
					# 	q_values[0][action] = target_q
					# 	self.net.model.fit(state,q_values,epochs=1,verbose=0)
					# 	break

					# else:

					# 	new_state = np.reshape(new_state, [1, state_space])								
						
					# 	target_q = reward + gamma*(np.amax(self.net.model.predict(new_state)[0]))
					# 	q_values[0][action]=target_q
					# 	self.net.model.fit(state,q_values,epochs=1,verbose=0)
							# 	state=new_state
					minibatch = memory.sample_batch()
					for state,action,reward,new_state,done in minibatch:
						if done:
							q_values= self.net.model.predict(state)
							target_q = reward
							q_values[0][action]=target_q
							self.net.model.fit(state,q_values,epochs=1,verbose=0)
						else:

							q_values= self.net.model.predict(state)
							target_q = reward + gamma*(np.amax(self.net.model.predict(new_state)[0]))
							q_values[0][action]=target_q
							self.net.model.fit(state,q_values,epochs=1,verbose=0)


		
	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		pass 

	def burn_in_memory():
		# Initialize your replay memory with a burn_in number of episodes / transitions. 

		pass

	
class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		# self.agent =DQN_Agent()
		self.transitions =[]
		self.memory_size=memory_size
		for i in range(burn_in):
			state=env.reset()
			action=random.randrange(action_s)
			state=np.reshape(state,[1,state_space])
			new_state, reward, done, _ = env.step(action)
			new_state=np.reshape(new_state,[1,state_space])
			self.transitions.append([state,action,reward,new_state,done])




	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		
		return random.sample(self.transitions,batch_size)






	def append(self, transition):
		# Appends transition to the memory. 	
		if(len(self.transitions)<self.memory_size):
			self.transitions.append(transition)
		else:
			idx=random.randint(self.memory_size)
			del sel.transitions[idx]
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

