def batch_mean_squared_error(forward_pass_output, targets, number_of_output_neurons):

	squared_error_sum = 0
	for i in range(len(targets)): 		#length of targets and forward_pass_op is the same, use either here
		error = 0
		for j in range(number_of_output_neurons):
			error += targets[i][j] - forward_pass_output[i][j]
		squared_error_sum += error * error 

	mean_squared_error = squared_error_sum / len(targets)

	return(mean_squared_error)