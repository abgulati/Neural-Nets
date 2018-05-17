from SoftmaxFP import *			#o/p layer uses softmax function for multi-class classification
from InitializeWeights import *
from batch_mean_squared_error2 import *
from csv_reader import *

#one-hot matrices for digits 0-9, these are the targets
key_values = {0:[1,0,0,0,0,0,0,0,0,0], 4:[0,0,0,0,1,0,0,0,0,0], 7:[0,0,0,0,0,0,0,1,0,0],
			  1:[0,1,0,0,0,0,0,0,0,0], 5:[0,0,0,0,0,1,0,0,0,0], 8:[0,0,0,0,0,0,0,0,1,0],
			  2:[0,0,1,0,0,0,0,0,0,0], 6:[0,0,0,0,0,0,1,0,0,0], 9:[0,0,0,0,0,0,0,0,0,1],
			  3:[0,0,0,1,0,0,0,0,0,0]}									

#obtaining necessary inputs and casting to int type
number_of_input_neurons = int(input("Enter the number of neurons in the i/p layer"))
number_of_hidden_neurons = int(input("Enter the number of hidden neurons"))
number_of_output_neurons = int(input("Enter the number of neurons in the o/p layer"))

#obtain specific network input
pixels, labels = read_input()

#initialize weights in the network
input_to_hidden_layer_wts, hidden_to_output_layer_wts = initialize_weights(number_of_input_neurons, number_of_hidden_neurons,
															number_of_output_neurons)

for i in range(10):
	
	forward_pass_output = []
	for i in range(len(labels)):
		forward_pass_output.append(forward_propagate(pixels[i], number_of_input_neurons, number_of_hidden_neurons, 
									number_of_output_neurons, input_to_hidden_layer_wts, hidden_to_output_layer_wts))

	#print(forward_pass_output)
	error = batch_mean_squared_error(forward_pass_output, key_values, labels, number_of_output_neurons)

	#print(error)