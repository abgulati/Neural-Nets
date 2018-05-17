def batch_mean_squared_error(forward_pass_output, key_values, labels, number_of_output_neurons):

	targets = []
	for i in range(len(labels)):
		targets.append(key_values.get(labels[i]))

	squared_error_sum = 0
	for i in range(len(forward_pass_output)):
		error = 0
		for j in range(number_of_output_neurons):
			error += targets[i][j] - forward_pass_output[i][j]
		squared_error_sum += error * error 

	mean_squared_error = squared_error_sum / len(targets)

	return(mean_squared_error)