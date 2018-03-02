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
		self.model.add(Dense(action_space,activation='linear',input_dim=4))

		self.optimizer=keras.optimizers.Adagrad(lr=learning_rate)
		self.model.compile(loss='mse',optimizer=self.optimizer)
		

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		pass

	def load_model(self, model_file):
		# Helper function to load an existing model.
		pass

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		pass



action_space=2
learning_rate=0.0001
max_steps=200
episodes=1000000
epsilon_start=0.5
epsilon_end=0.05
decay=(epsilon_start-epsilon_end)/100000
batch_size=1
max_steps=200


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
		self.net=QNetwork(learning_rate,action_space)
		self.q_values=np.zeros([batch_size,action_space])
		

	def epsilon_greedy_policy(self, q_values,epsilon):
		# Creating epsilon greedy probabilities to sample from.
		if (epsilon>np.random.random()):
			action=env.action_space.sample()
		else:
			action=np.argmax(q_values)
		
	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		
		action=np.argmax(q_values)
	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

		print ("H")
		env.reset()
		for i in range(episodes):
			step=0
			state=np.zeros([1,4])
			action=env.action_space.sample()


			while(step<max_steps):

					total_reward=0
					new_state, reward, done, _ = env.step(action)
					total_reward += reward
					if done:
						print ("Cummulative reward: ",total_reward)
						new_state = np.reshape(new_state, [1, 4])
						target = net.model(predict(new_state)[0])
						q_values = [reward,reward]
						net.model(fit(state,q_values))
						break

					else:

						new_state = np.reshape(new_state, [1, 4])								
						epsilon=epsilon_start+(epsilon_start-epsilon_end)*np.exp(decay*i)
						
						q_values= self.net.model(predict(state))
						action=epsilon_greedy_policy(q_values,epsilon)
						target_q = reward + gamma*(np.amax(net.predict(new_state)[0]))
						q_values[0][action]=target_q
						net.fit(state,q_values)



		
	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		pass 

	def burn_in_memory():
		# Initialize your replay memory with a burn_in number of episodes / transitions. 

		pass

	

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
	print(agent)
	DQN_Agent.train(agent)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)

