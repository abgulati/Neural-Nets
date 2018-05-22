from SoftmaxFP import *			#o/p layer uses softmax function for multi-class classification
from InitializeWeights import *
from batch_mean_squared_error2 import *
from csv_reader import *
from normalize import *

#one-hot matrices for digits 0-9, these are the targets
key_values = {0:[1,0,0,0,0,0,0,0,0,0], 4:[0,0,0,0,1,0,0,0,0,0], 7:[0,0,0,0,0,0,0,1,0,0],
			  1:[0,1,0,0,0,0,0,0,0,0], 5:[0,0,0,0,0,1,0,0,0,0], 8:[0,0,0,0,0,0,0,0,1,0],
			  2:[0,0,1,0,0,0,0,0,0,0], 6:[0,0,0,0,0,0,1,0,0,0], 9:[0,0,0,0,0,0,0,0,0,1],
			  3:[0,0,0,1,0,0,0,0,0,0]}									

#obtaining necessary inputs and casting to int type
number_of_input_neurons = int(input("Enter the number of neurons in the i/p layer: "))
number_of_hidden_neurons = int(input("Enter the number of hidden neurons: "))
number_of_output_neurons = int(input("Enter the number of neurons in the o/p layer: "))
learning_rate = int(input("Enter the learning rate: "))

#obtain specific network input
pixels, labels = read_input()
pixels_normalized = normalize(pixels)
#print(pixels_normalized)

#initialize weights in the network
input_to_hidden_layer_wts, hidden_to_output_layer_wts = initialize_weights(number_of_input_neurons, number_of_hidden_neurons,
															number_of_output_neurons)

error_over_time = []

for i in range(10):
	
	collective_forward_pass_output = []
	for i in range(len(labels)):
		#obtain output of the forward pass
		hidden_layer_input, hidden_layer_output, forward_pass_output = forward_propagate(pixels_normalized[i], 
									number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons, 
									input_to_hidden_layer_wts, hidden_to_output_layer_wts)

		#to compute batch error:
		collective_forward_pass_output.append(forward_pass_output)

		#otain new weights via back propagation
		hidden_to_output_layer_wts, input_to_hidden_layer_wts = back_propagate(key_values.get(labels[i]), forward_pass_output,
												number_of_output_neurons, number_of_hidden_neurons, number_of_input_neurons, 
												hidden_to_output_layer_wts, input_to_hidden_layer_wts, pixels_normalized[i],
												hidden_layer_input, hidden_layer_output, learning_rate)

	error = batch_mean_squared_error(forward_pass_output, key_values, labels, number_of_output_neurons)
	print(error)

	error_over_time.append(str(error))


#with open('error_rate.csv', 'w') as csv_file:
#	writer = csv.writer(csv_file, delimiter=',')
#	writer.writerow(error_over_time)

with open('error_rate.csv', 'w') as file:
	for line in error_over_time:
		file.write(line)
		file.write('\n')