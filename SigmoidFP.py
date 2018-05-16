import math

def forward_propagate(current_input, number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons,
						input_to_hidden_layer_wts, hidden_to_output_layer_wts):

	hidden_layer_output = []
	for i in range(0,number_of_hidden_neurons):
		sum_of_products_of_input = 0
		for j in range(0,number_of_input_neurons):
			sum_of_products_of_input += current_input[j] * input_to_hidden_layer_wts[j][i]
		neuron_output = (1/(1+math.exp(-sum_of_products_of_input)))
		hidden_layer_output.append(neuron_output)

	network_output = []
	for i in range(0,number_of_output_neurons):
		sum_of_products_of_input = 0
		for j in range(0,number_of_hidden_neurons):
			sum_of_products_of_input += hidden_layer_output[j] * hidden_to_output_layer_wts[j][i]
		neuron_output = (1/(1+math.exp(-sum_of_products_of_input)))
		network_output.append(neuron_output)

	return network_output