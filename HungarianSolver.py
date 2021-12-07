import sys
import tensorflow as tf
import Constants
import DebugUtils

class HungarianSolver:
	"""!
			@brief Class that implements a solver for the assignment problem based on
						 an optimization algorithm.
						 
						 In its most general form, the assignment problem is as follows: the
						 problem instance has a number of agents and a number of tasks. Any
						 agent can be assigned to perform any task, incurring some cost that
						 may vary depending on the agent-task assignment. It is required to
						 perform all tasks by assigning exactly one agent to each task and
						 exactly one task to each agent in such a way that the total cost of
						 the assignment is minimized. If the numbers of agents and tasks are
						 equal and the total cost of the assignment for all tasks is equal
						 to the sum of the costs for each agent (or the sum of the costs for
						 each task, which is the same thing in this case), then the problem
						 is called the linear assignment problem.
						 
						 We address this problem via optimization. Given a cost matrix,
						 representing the cost of assigning each variable of a set X with a
						 variable of a set Y, it provides as output the optimal assignment
						 between each variable in X with exactly one variable in Y. For
						 further information please have a look at the Hungarian algorithm.
	"""
	
	def __init__(self, name, batchSize, isRestoringGraph):
		"""!
				@brief Constructor in which we define all operators and variables to be
							 added in the TensorFlow graph.
				
				@param name string representing the TensorFlow scope.
				@param batchSize int representing the size of the batch.
				@param isRestoringGraph boolean representing whether we are restoring a
							 graph or not.
		"""
		
		with tf.name_scope(name):
			## List containing all nodes which can be potentially visualized in
			## TensorBoard.
			self.summaries = []
			
			## Size of the batch.
			self.batchSize = batchSize
			
			## The tensor for computing the maximum column regularization values used
			## for normalizing regularization values during the optimization process.
			self.regularizationColumn = None
			
			## The tensor for computing the maximum row regularization values used for
			## normalizing regularization values during the optimization process.
			self.regularizationRow = None
			
			# We are restoring a pre-trained graph, therefore variables and operators
			# have been already defined.
			if isRestoringGraph == True:
				try:
					self.alpha = tf.get_collection_ref(name + '/alpha')[0]
					
					self.assignmentDefault = tf.get_collection_ref(name +
																												'/assignmentDefault')[0]
					
					self.assignment = tf.get_collection_ref(name + '/assignment')[0]
					
					self.costMatrix = tf.get_collection_ref(name + '/costMatrix')[0]
					
					self.finalAssignment = tf.get_collection_ref(name +
																											 '/finalAssignment')[0]
					
					self.maxRegularizationColumn = tf.get_collection_ref(name +
																								'/maxRegularizationColumn')[0]
					
					self.maxRegularizationRow = tf.get_collection_ref(name +
																										'/maxRegularizationRow')[0]
					
					self.costMatrixPlaceholder = tf.get_collection_ref(name +
																										'/costMatrixPlaceholder')[0]
					
					self.checkAssignment = tf.get_collection_ref(name +
																											 '/checkAssignment')[0]
					
					self.increaseAlpha = tf.get_collection_ref(name + '/increaseAlpha')[0]
					
					self.init = (tf.get_collection_ref(name + '/initAssignment')[0],
											 tf.get_collection_ref(name + '/initAlpha')[0])
					
					self.resetAlpha = tf.get_collection_ref(name + '/resetAlpha')[0]
					
					self.train = tf.get_collection_ref(name + '/train')[0]
					
					self.updateCostMatrix = tf.get_collection_ref(name +
																												'/updateCostMatrix')[0]
					
					self.updateFinalAssignment = tf.get_collection_ref(name +
																										'/updateFinalAssignment')[0]
					
					self.updateMaxRegularization = (tf.get_collection_ref(name +
																					'/updateMaxRegularizationColumn')[0],
																					tf.get_collection_ref(name +
																						'/updateMaxRegularizationRow')[0])
					
					return
				except:
					collections = tf.get_default_graph().get_all_collection_keys()
					
					collectionNotFound = collections[-1]
					
					DebugUtils.Err('', '', True)
					DebugUtils.Err('HungarianSolver', 'Tensor \'', False)
					DebugUtils.Warn('', collectionNotFound, False)
					DebugUtils.Err('', '\' not found in the collections. List ', False)
					DebugUtils.Err('', 'of all collections is:', True)
					
					for i in range(0, len(collections) - 1):
						DebugUtils.Info('HungarianSolver',
														' - %s' %collections[i], True)
					
					DebugUtils.Err('HungarianSolver', 'Exiting...', True)
					
					sys.exit(-1)
			
			# Defining a name scope for the variables so that the code is somehow
			# divided in logical blocks.
			with tf.name_scope('Variables'):
				globalStep = tf.Variable(0, name = 'GlobalStepSolver',
																 trainable = False)
				
				## The tensor which contains the alpha parameter used to balance the
				## regularization factors in the optimization function.
				self.alpha = tf.Variable(1.0, name = 'Alpha', trainable = False)
				
				## The tensor which contains the default assignment used to initialize
				## at every iteration and when no solution has been found.
				self.assignmentDefault = tf.constant(0.01,
																			shape = [self.batchSize,
																							 Constants.MAX_DETECTED_OBJECTS,
																							 Constants.MAX_DETECTED_OBJECTS],
																			dtype = tf.float32)
				
				## The tensor which contains the optimal assignment found after the
				## computation.
				self.assignment = tf.Variable(tf.ones([self.batchSize,
																				Constants.MAX_DETECTED_OBJECTS,
																				Constants.MAX_DETECTED_OBJECTS]) - 0.99,
																			name = 'Assignment',
																			dtype = tf.float32)
				
				## The tensor which represents the cost matrix of assigning one variable
				## in one set with respect to another variable in the other set.
				self.costMatrix = tf.Variable(tf.zeros(
																						[self.batchSize,
																						 Constants.MAX_DETECTED_OBJECTS,
																						 Constants.MAX_DETECTED_OBJECTS]),
																			name = 'CostMatrix', dtype = tf.float32,
																			trainable = False)
				
				## The tensor which contains the final optimal assignment in which we
				## only have 0s and 1s.
				self.finalAssignment = tf.Variable(tf.ones([self.batchSize,
																							Constants.MAX_DETECTED_OBJECTS,
																							Constants.MAX_DETECTED_OBJECTS]),
																					 name = 'FinalAssignment',
																					 dtype = tf.float32,
																					 trainable = False)
				
				## The tensor which contains the maximum column regularization value
				## used for normalizing regularization values during the optimization
				## process. It is set at the first iteration.
				self.maxRegularizationColumn = tf.Variable(1.0,
																									 name = 'MaxRegColumn',
																									 trainable = False)
				
				## The tensor which contains the maximum row regularization value used
				## for normalizing regularization values during the optimization
				## process. It is set at the first iteration.
				self.maxRegularizationRow = tf.Variable(1.0, name = 'MaxRegRow',
																								trainable = False)
			
			# Defining a name scope for the placeholders so that the code is somehow
			# divided in logical blocks.
			with tf.name_scope('Placeholders'):
				## Placeholder used for assigning the new cost matrix for the next
				## optimization iteration. It is a 'float32' 3D tensor having shape
				## [batchSize, maxDetectedObjects, maxDetectedObjects].
				self.costMatrixPlaceholder = tf.placeholder(self.costMatrix.dtype,
																										self.costMatrix.get_shape())
			
			# Defining a name scope for the operators so that the code is somehow
			# divided in logical blocks.
			with tf.name_scope('Operators'):
				## The tensor for checking whether the assignment is valid or not.
				self.checkAssignment = self.checkValidAssignment()
				
				## The tensor for increasing the alpha parameter to give more weight to
				## the constraints of the problem.
				self.increaseAlpha = self.increaseAlphaParameter()
				
				initAssignment, initAlpha = self.initialize()
				
				## The tensor for initializing the assignment matrix and the alpha
				## parameter.
				self.init = (initAssignment, initAlpha)
				
				## The tensor for resetting the alpha parameter to its default value.
				self.resetAlpha = self.resetAlphaParameter()
				
				## The tensor for finding the assignment by having as input the cost
				## matrix.
				self.train = self.solve(globalStep)
				
				## The tensor for updating the cost matrix.
				self.updateCostMatrix = self.assignCostMatrix()
				
				## The tensor for updating the final assignment found during the last
				## iteration.
				self.updateFinalAssignment = self.assignFinalAssignment()
				
				(updateMaxRegularizationColumn,
				 updateMaxRegularizationRow) = self.assignMaxRegularization()
				
				## The tensor for updating the maximum regularization values used for
				## normalizing regularization values during the optimization process.
				self.updateMaxRegularization = (updateMaxRegularizationColumn,
																				updateMaxRegularizationRow)
			
			# Adding variables to TensorFlow collection that will be used when
			# executing a restored pre-trained model.
			tf.add_to_collection(name + '/alpha', self.alpha)
			tf.add_to_collection(name + '/assignmentDefault', self.assignmentDefault)
			tf.add_to_collection(name + '/assignment', self.assignment)
			tf.add_to_collection(name + '/costMatrix', self.costMatrix)
			tf.add_to_collection(name + '/finalAssignment', self.finalAssignment)
			tf.add_to_collection(name + '/maxRegularizationColumn',
													 self.maxRegularizationColumn)
			tf.add_to_collection(name + '/maxRegularizationRow',
													 self.maxRegularizationRow)
			
			# Adding placeholders to TensorFlow collection that will be used when
			# executing a restored pre-trained model.
			tf.add_to_collection(name + '/costMatrixPlaceholder',
													 self.costMatrixPlaceholder)
			
			# Adding tensors to TensorFlow collection that will be used when
			# executing a restored pre-trained model.
			tf.add_to_collection(name + '/checkAssignment', self.checkAssignment)
			tf.add_to_collection(name + '/increaseAlpha', self.increaseAlpha)
			tf.add_to_collection(name + '/initAssignment', initAssignment)
			tf.add_to_collection(name + '/initAlpha', initAlpha)
			tf.add_to_collection(name + '/resetAlpha', self.resetAlpha)
			tf.add_to_collection(name + '/train', self.train)
			tf.add_to_collection(name + '/updateCostMatrix', self.updateCostMatrix)
			tf.add_to_collection(name + '/updateFinalAssignment',
													 self.updateFinalAssignment)
			tf.add_to_collection(name + '/updateMaxRegularizationColumn',
													 updateMaxRegularizationColumn)
			tf.add_to_collection(name + '/updateMaxRegularizationRow',
													 updateMaxRegularizationRow)
	
	def assignCostMatrix(self):
		"""!
				@brief Function that returns an operator for updating the cost matrix
							 from the placeholder to the variable.
				
				@return the operator for updating the cost matrix from the placeholder
								to the variable.
		"""
		
		with tf.name_scope('AssignCostMatrix'):
			return tf.assign(self.costMatrix, self.costMatrixPlaceholder)
	
	def assignFinalAssignment(self):
		"""!
				@brief Function that returns an operator for updating the final optimal
							 assignment.
							 
							 Due to the optimization process, a value clipping step is
							 required to assign 0 to values lower than a threshold and 1 to
							 the others.
				
				@return the operator for updating the final optimal assignment.
		"""
		
		with tf.name_scope('AssignFinalAssignment'):
			return tf.assign(self.finalAssignment,
											 tf.clip_by_value(tf.sign(self.assignment - 0.95), 0, 1))
	
	def assignMaxRegularization(self):
		"""!
				@brief Function that returns an operator for updating the maximum
							 regularization parameters.
							 
							 The regularization guarantees - during the optimization process -
							 that all constraints on the cost matrix are hold.
				
				@return the operator for updating the final optimal assignment.
		"""
		
		with tf.name_scope('MaxRegularization'):
			return (tf.assign(self.maxRegularizationColumn,
												self.regularizationColumn),
							tf.assign(self.maxRegularizationRow, self.regularizationRow))
	
	def checkValidAssignment(self):
		"""!
				@brief Function that returns an operator for checking whether the
							 assignment is valid or not.
							 
							 An assignment is valid if and only if, there is no variable - of
							 one set - assigned to more than one variable of the other set.
							 The only exception is for variables associated to fake agents, if
							 any.
				
				@return the operator for checking whether the assignment is valid or
								not.
		"""
		
		with tf.name_scope('CheckValidAssignment'):
			# We subtract a threshold value to the assignment matrix in such a way,
			# all positive elements in the matrix represents a valid assignment
			# between predictions and targets, while all other elements are 0s.
			finalAssignment = tf.clip_by_value(tf.sign(self.assignment - 0.95), 0, 1)
			
			# Computing the number of agents (i.e., targets) assigned to a given
			# task (i.e., predictions).
			agents = tf.reduce_sum(tf.clip_by_value(tf.reduce_sum(self.costMatrix,
																														axis = 1), 0, 1),
														 axis = 1)
			
			# Computing the number of predictions (i.e., tasks) assigned to a given
			# target (i.e., agent).
			assignedPredictions = tf.reduce_sum(tf.clip_by_value(tf.reduce_sum(
								finalAssignment * self.costMatrix, axis = 1), 0, 1), axis = 1)
			
			return tf.equal(tf.reduce_sum(agents - assignedPredictions), 0)
	
	def increaseAlphaParameter(self):
		"""!
				@brief Function that returns an operator for increasing the alpha
							 parameters to give more weight to the constraints of the problem.
				
				@return the operator for increasing the alpha parameter to give more
								weight to the constraints of the problem.
		"""
		
		with tf.name_scope('IncreaseAlpha'):
			return tf.assign(self.alpha, tf.multiply(self.alpha, 1.1))
	
	def initialize(self):
		"""!
				@brief Function that returns an operator for initializing the assignment
							 matrix and the alpha parameter.
				
				@return the operator for initializing the assignment matrix and the
								alpha parameter.
		"""
		
		with tf.name_scope('Initialize'):
			return (tf.assign(self.assignment, self.assignmentDefault),
							tf.assign(self.alpha, 1.0))
	
	def resetAlphaParameter(self):
		"""!
				@brief Function that returns an operator for resetting the alpha
							 parameter to its default value.
				
				@return the operator for resetting the alpha parameter to its default
								value.
		"""
		
		with tf.name_scope('ResetAlpha'):
			return tf.assign(self.alpha, 1.0)
	
	def solve(self, globalStep):
		"""!
				@brief Function that returns an operator for solving the assignment
							 problem via optimization.
							 
							 In particular, it uses a gradient descent optimizer. The loss
							 function is defined by considering the cost value of assigning
							 two variables and the constraints of the problem. Specifically,
							 given a cost matrix, representing the cost of assigning each
							 variable of a set X with a variable of a set Y, the assignment
							 must assign each variable in X with exactly one variable in Y.
				
				@param globalStep current global step of the training process.
				
				@return the operator for solving the assignment problem via
								optimization.
		"""
		
		with tf.name_scope('Solve'):
			# We clip the assignment matrix by assigning 0 to all negatives numbers
			# and 1 to all numbers greater than 1. Thus, the values will be bounded
			# between [0,1].
			clipAssignment = tf.clip_by_value(self.assignment, 0, 1)
			
			# The cost loss is defined by multiplying element-wise the cost matrix
			# with the assignment matrix being clipped.
			costLoss = tf.abs(tf.reduce_sum(self.costMatrix * self.assignment))
			
			# This regularization factor has the key role in the optimization process
			# to avoid having multiple 1s in one row which violets the constraints.
			self.regularizationRow = tf.norm(tf.matmul(clipAssignment,
				tf.transpose(tf.ones([1, self.costMatrix.shape[1], self.batchSize]))) -
				tf.transpose(tf.ones([self.batchSize, self.costMatrix.shape[1], 1])))
			
			# This regularization factor has the key role in the optimization process
			# to avoid having multiple 1s in one column which violets the constraints.
			self.regularizationColumn = tf.norm(
				tf.matmul(tf.ones([self.batchSize, 1, self.costMatrix.shape[1]]),
									clipAssignment) - tf.ones([self.batchSize, 1,
																						 self.costMatrix.shape[1]]))
			
			# This regularization factor has the key role in the optimization process
			# to avoid having negative values in the assignment matrix.
			regularizationZero = tf.reduce_sum(tf.maximum(-self.assignment, 0))
			
			# The loss function is a linear combination of all factors defined above
			# with a weight for each component.
			loss = (costLoss +
				((self.alpha * (self.regularizationRow / self.maxRegularizationRow)) +
				 (self.alpha * (self.regularizationColumn /
												self.maxRegularizationColumn)) + regularizationZero))
			
			# Gradient descent optimizer with a fixed learning rate. We also tried
			# other optimizers which are way worse for such a particular problem.
			optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
			
			# Minimizing the loss with the given optimizer.
			return optimizer.minimize(loss, global_step = globalStep)
