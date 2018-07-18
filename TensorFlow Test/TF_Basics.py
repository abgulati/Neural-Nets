# Disable AX2 warning:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tensorflow provides multiple APIs, of which TensorFlow Core is the lowest level.
# Other APIs sit atop this and are typically easier to learn and use than TensorFlow Core,
# and they make repetitive tasks easier and more consistent between different users, and they
# help you manage data sets, estimators, training and inference.
# Examples of theses higher level APIs are tf.contrib.learn and Keras. Those whose method names
# contain'contrib' are under development and may change or become obsolete with subsequent TF releases.

# The tensor is the central unit of data in TensorFlow, and it consists of a set of primitive values
# shaped into an array of any number of dimensions, or in other words, may be of any 'rank'.
# Examples:
# 3 -> rank 0 tensor; a scalar with shape []
# [1., 2., 3.] -> rank 1 tensor; vector with shape [3]
# [[1,2,3], [4,5,6]] -> rank 2 tensor; matrix with shape [2,3]
# [ [ [1,2,3] ], [ [4,5,6] ] ] -> rank 3 tensor; shape [2,1,3]

# Let's see some code now:

import tensorflow as tf 
# This is the canonical statement for TensorFlow programs, and gives Python access to all of TFs classes, 
# methods and symbols.

# TensorFlow programs consist of two discrete sections:
# 1. Building a computatuional graph
# 2. Running that computational graph

# A computational graph is a series of TensorFlow operations arranged into a graph of nodes, wherein each 
# node takes zero or more tensors as inputs and produces a tensor as an output.

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) #the same as above, as 'tf.float32' is implicit
# So one such node is the constant node, and like all TensorFlow constants, it'll take no inputs and will 
# output the value it stores internally.

print(node1, node2)
# Will output: 
# Tensor("Const:0", shape(), dtype=float32) Tensor("Const_1:0", shape(), dtype=float32)

#Notice this didn't print their values as you may have expected! To print that, they nodes need to be evaluated.

# To evaluate nodes, we must run the computational graph in a session. A session encapsulates the control and state of the TF runtime. 
sess = tf.Session()
print(sess.run([node1, node2])) # prints [3.0, 4.0]
# Above, we create a Session object and then invoke its 'run' method to run the computational graph to evaluate node1 and node2.

# More complicated computations can be built by combining Tensor nodes with operations (operations are also nodes!). For example,
# we can add our two constant nodes and produce a new graph as follows:

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))
# Outputs:
# node3: Tensor("Add_2.0", shape=(), dtype=float32)
# sess.run(node3): 7.0

# This graph isn't particularly interesting as it produces a constant result.
# But we can parameterize a graph to accept external inputs, by using TF "placeholders". A placeholder is a promise to provide a value later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b 	# Here, '+' acts as a shortcut to tf.add(a,b)

# Evaluating this graph:
print(sess.run(adder_node, {a:3, b:4.5})) 	# 7.5
print(sess.run(adder_node, {a:[1,3], b:[2,4]})) # [3. 7.]

# Taking this further:
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a:3, b:4.5})) 	# 22.5

# Now to make the model trainable, we need to be able to modify the graph to get new 
# outputs with the same inputs, and Variables allow us to add trainable parameters to a graph. They are constructed with an intital type and value:
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Constants are initialzed when you call tf.constant, and their value never changes after that. By contrast, variables are not initialized
# at the tf.Variable call. In order to initialize all variables in a TensorFlow program, a special operation needs to be explicitly called as follows:
init = tf.global_variables_initializer()
sess.run(init)
# init is a handle to the TF sub-graph that initializes all global variables once sess.run is called, thus unless sess.run is called, the variables are not yet initialized!

# Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:
print(sess.run(linear_model, {x:[1,2,3,4]}))
# Outputs: [0.	0.3 	0.6 	0.9]

# So we've created a model, but have no idea how well it does. To evaluate our model, we need training data for which we'll use a placeholder 'y' to 
# provide the desired values and provide a loss function.

# A loss function measures how far apart the current model is from the provided data. We'll use a standard loss model for linear regression, which sums the 
# squares of the deltas between the current model and the provided data.
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) #outputs: 23.66
# linear_model - y creates a vector where each element is the corresponding examples error delta.
# tf.square is called to square that error.
# Then, using tf.reduce_sum, we sum all the squared errors to create a single scalar that abstracts the error of all examples

# We can improve this manually by reassigning the values of W and b to the perfect values of -1 and 1!
# A variable initialized to the value provided to tf.Variable but can be changed using operations like tf.assign:
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# prints 0.0

# We "guessed" the correct values, but the whole point of ML is to find the correct model parameters automatically. Lets accomplish this next:

# TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function. The simplest optimizer is gradient descent (GD).
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # resets values to incorrect defaults

for i in range(1000):
	sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))
# outputs: [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]

# Now we have done some actual machine learning!