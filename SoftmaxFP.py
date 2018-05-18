import math
import functools

def forward_propagate(current_input, number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons, 
						input_to_hidden_layer_wts, hidden_to_output_layer_wts):
	
	#compute output of the hidden layer:
	hidden_layer_output = []
	for i in range(0, number_of_hidden_neurons):
		sum_of_products_of_input = 0
		for j in range(0, number_of_input_neurons):
			sum_of_products_of_input += current_input[j] * input_to_hidden_layer_wts[j][i]
		neuron_output = (1/(1+math.exp(-sum_of_products_of_input)))			#sigmoid function here
		hidden_layer_output.append(neuron_output)

	#now to compute the output of the o/p layer, i.e., the network o/p:

	#first apply math.exp() to all inputs to the o/p layer
	exponents_of_inputs = map(lambda x : math.exp(x), hidden_layer_output)

	#now compute the sum of these exponents:
	sum_of_exponents = functools.reduce(lambda x,y : x+y, exponents_of_inputs)

	network_output = []
	for i in range(0, number_of_output_neurons):
		sum_of_products_of_input = 0
		for j in range(0, number_of_hidden_neurons):
			sum_of_products_of_input += hidden_layer_output[j] * hidden_to_output_layer_wts[j][i]
		neuron_output = math.exp(sum_of_products_of_input)/sum_of_exponents		#softmax function here
		network_output.append(neuron_output)

	return hidden_layer_output, network_output