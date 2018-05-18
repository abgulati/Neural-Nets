def back_propagate(target_output, network_output, number_of_output_neurons, number_of_hidden_neurons, 
					number_of_input_neurons, hidden_to_output_layer_wts, input_to_hidden_layer_wts, network_input,
					hidden_layer_output, learning_rate):

	new_hidden_to_output_layer_wts = []
	delta_output_layer = []

	temp_wts = []
	for i in range(number_of_output_neurons):
		temp = []
		delta = network_output[i] - target_output[i]
		delta_output_layer.append(delta)
		neuron_output = network_output[i]		#to compute a(1-a)
		output_factor = neuron_output * (1 - neuron_output)
		weight_delta = learning_rate * delta  * output_factor	#alpha * delta * a(1-a)
		for j in range(number_of_hidden_neurons):
			temp.append(weight_delta + hidden_to_output_layer_wts[j][i])	#multiplying above product by x, i.e., o/p of previous layer
		temp_wts.append(temp)

	for j in range(len(temp_wts[0])):
		temp = []
		for i in range(len(temp_wts)):
			temp.append(temp_wts[i][j])
		new_hidden_to_output_layer_wts.append(temp)

	
	new_input_to_hidden_layer_wts = []

	#compute delta for the hidden layer using delta of the output layer
	
	delta_hidden_layer = []
	for i in range(number_of_hidden_neurons):
		delta = 0
		for j in range(number_of_output_neurons):
			delta += new_hidden_to_output_layer_wts[i][j] * delta_output_layer[j]
		delta_hidden_layer.append(delta)

	#we can now compute weight changes for the input to hidden layer weights:

	for i in range(number_of_hidden_neurons):
		for j in range(number_of_input_neurons):
			