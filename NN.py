import math

def forward_propagate():

	hidden_layer_output = []
	for i in range(0,number_of_hidden_neurons):
		sum_of_products_of_input = 0
		for j in range(0,number_of_input_neurons):
			sum_of_products_of_input += network_inputs[j] * input_to_hidden_layer_wts[j][i]
		neuron_output = (1/(1+math.exp(-sum_of_products_of_input)))
		hidden_layer_output.append(neuron_output)

	i = j = 0

	network_output = []
	for i in range(0,number_of_output_neurons):
		sum_of_products_of_input = 0
		for j in range(0,number_of_hidden_neurons):
			sum_of_products_of_input += hidden_layer_output[j] * hidden_to_output_layer_wts[j][i]
		neuron_output = (1/(1+math.exp(-sum_of_products_of_input)))
		network_output.append(neuron_output)

	return neuron_output


number_of_input_neurons = input("Enter the number of neurons in the i/p layer")
number_of_hidden_neurons = input("Enter the number of hidden neurons")
number_of_output_neurons = input("Enter the number of neurons in the o/p layer")

#casting to int()
number_of_input_neurons = int(number_of_input_neurons)
number_of_hidden_neurons = int(number_of_hidden_neurons)
number_of_output_neurons = int(number_of_output_neurons)

network_inputs = []										#x
print("Please provide network inputs one at a time")
for i in range(0,number_of_input_neurons):
	temp = input("Provide input ")
	temp = int(temp)
	network_inputs.append(temp)

input_to_hidden_layer_wts = []						#theta
for i in range (0,number_of_input_neurons):
	temp = []
	for j in range(0, number_of_hidden_neurons):
		temp.append(1)
	input_to_hidden_layer_wts.append(temp)

hidden_to_output_layer_wts = []
for i in range(0,number_of_hidden_neurons):
	temp = []
	for j in range(0,number_of_output_neurons):
		temp.append(1)
	hidden_to_output_layer_wts.append(temp)

print("The inputs are: ", network_inputs)
print("The weights from input to hidden layer are initialised as: ", input_to_hidden_layer_wts)
print("The weights from the hidden to output layer is: ", hidden_to_output_layer_wts)

forward_pass_output = forward_propagate()

print(forward_pass_output)