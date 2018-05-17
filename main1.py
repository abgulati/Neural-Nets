from SoftmaxFP import *			#o/p layer uses softmax function for multi-class classification
from InitializeWeights import *
from batch_mean_squared_error import *
from csv_reader import *

#obtaining necessary inputs and casting to int type
number_of_input_neurons = int(input("Enter the number of neurons in the i/p layer"))
number_of_hidden_neurons = int(input("Enter the number of hidden neurons"))
number_of_output_neurons = int(input("Enter the number of neurons in the o/p layer"))

#these are the desired outputs
targets = []
print("Enter targets:")
for i in range(number_of_output_neurons):
	temp = []
	print("enter the target for input", i+1, " one bit at a time:")
	for j in range(number_of_output_neurons):
		temp.append(int(input()))
	targets.append(temp)

#obtain specific network input
network_inputs = []
print("Please provide network inputs one at a time")
for i in range(number_of_inputs):
	temp = []
	print("provide input", i, " ,one bit at a time:")
	for j in range(number_of_input_neurons):
		temp.append(int(input()))
	network_inputs.append(temp)

#initialize weights in the network
input_to_hidden_layer_wts, hidden_to_output_layer_wts = initialize_weights(number_of_input_neurons, number_of_hidden_neurons,
															number_of_output_neurons)

#reporting the generated weights
print("The inputs are: ", network_inputs)
#print("The weights from input to hidden layer are initialised as: ", input_to_hidden_layer_wts)
#print("The weights from the hidden to output layer is: ", hidden_to_output_layer_wts)


for i in range(10):
	
	#running a single forward pass
	forward_pass_output = []
	for i in range(number_of_inputs):
		forward_pass_output.append(forward_propagate(network_inputs[i], number_of_input_neurons, number_of_hidden_neurons, 
									number_of_output_neurons, input_to_hidden_layer_wts, hidden_to_output_layer_wts))

	error = batch_mean_squared_error(forward_pass_output, targets, number_of_output_neurons)

	print(error) 

#print(forward_pass_output)