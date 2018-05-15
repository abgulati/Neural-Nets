#from SigmoidFP import *		#o/p layer uses sigmoid function for binary classification
from SoftmaxFP import *			#o/p layer uses softmax function for multi-class classification
from InitializeWeights import *

#obtaining necessary inputs and casting to int type
number_of_input_neurons = input("Enter the number of neurons in the i/p layer")
number_of_input_neurons = int(number_of_input_neurons)

number_of_hidden_neurons = input("Enter the number of hidden neurons")
number_of_hidden_neurons = int(number_of_hidden_neurons)

number_of_output_neurons = input("Enter the number of neurons in the o/p layer")
number_of_output_neurons = int(number_of_output_neurons)

#obtain specific network input
network_inputs = []
print("Please provide network inputs one at a time")
for i in range(0,number_of_input_neurons):
	temp = input("Provide input ")
	temp = int(temp)
	network_inputs.append(temp)

#initialize weights in the network
input_to_hidden_layer_wts, hidden_to_output_layer_wts = initialize_weights(number_of_input_neurons, number_of_hidden_neurons,
															number_of_output_neurons)

#reporting the generated weights
print("The inputs are: ", network_inputs)
print("The weights from input to hidden layer are initialised as: ", input_to_hidden_layer_wts)
print("The weights from the hidden to output layer is: ", hidden_to_output_layer_wts)

#running a single forward pass
forward_pass_output = forward_propagate(network_inputs, number_of_input_neurons, number_of_hidden_neurons, 
							number_of_output_neurons, input_to_hidden_layer_wts, hidden_to_output_layer_wts)

print(forward_pass_output)