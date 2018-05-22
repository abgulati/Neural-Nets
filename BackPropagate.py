def back_propagate(target_output, network_output, number_of_output_neurons, number_of_hidden_neurons, 
					number_of_input_neurons, hidden_to_output_layer_wts, input_to_hidden_layer_wts, network_input,
					hidden_layer_input, hidden_layer_output, learning_rate):


	#computing delta terms first:
	delta_output_layer = []				#delta = target_output - actual_neuron_output
	for i in range(number_of_output_neurons):
		delta = 0
		delta = target_output[i] - network_output[i]
		delta_output_layer.append(delta)

	delta_hidden_layer = []				#delta += weight_from_hidden_to_output_layer * output_neurons_delta
	for i in range(number_of_hidden_neurons):
		delta = 0
		for j in range(number_of_output_neurons):
			delta += hidden_to_output_layer_wts[i][j] * delta_output_layer[j]
		delta_hidden_layer.append(delta)

	#to compute weight changes(using the delta rule, computing capital delta here):

	#computing capital theta, i.e., weight change factor for hidden to output ayer weights, i.e., theta2:
	capital_delta_theta2 = []					
	for i in range(number_of_hidden_neurons):
		temp = []
		for j in range(number_of_output_neurons):
			temp.append(learning_rate * delta_output_layer[j] * network_output[j] * (1 - network_output[j]) * 
						hidden_layer_output[i][j])
		capital_delta_theta2.append(temp)

	#computing cumulative output of each neuron in the hidden layer, requried for theta1:
	hidden_neuron_cumulative_output = []
	for i in range(number_of_hidden_neurons):
		sum = 0
		for j in range(number_of_output_neurons):
			sum += hidden_layer_output[i][j]
		hidden_neuron_cumulative_output.append(sum)

	#computing capital theta for input to hidden layer weights, i.e., theta1:
	capital_delta_theta1 = []
	for i in range(number_of_input_neurons):
		temp = []
		for j in range(number_of_hidden_neurons):
			temp.append(learning_rate * delta_hidden_layer[j] * hidden_neuron_cumulative_output[j] * 
						(1 - hidden_neuron_cumulative_output[j]) * hidden_layer_input[i][j])
		capital_delta_theta1.append(temp)

	#computing new weights:		new_weight = capital_delta - old_weight:

	#for theta2:
	new_hidden_to_output_layer_wts = []
	for i in range(number_of_hidden_neurons):
		temp = []
		for j in range(number_of_output_neurons):
			temp.append(hidden_to_output_layer_wts[i][j] - capital_delta_theta2[i][j])
		new_hidden_to_output_layer_wts.append(temp)

	#for theta1:
	new_input_to_hidden_layer_wts = []
	for i in range(number_of_input_neurons):
		temp = []
		for j in range(number_of_hidden_neurons):
			temp.append(input_to_hidden_layer_wts[i][j] - capital_delta_theta1[i][j])
		new_input_to_hidden_layer_wts.append(temp)

	return new_hidden_to_output_layer_wts, new_input_to_hidden_layer_wts
	