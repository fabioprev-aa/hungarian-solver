import warnings

warnings.filterwarnings("ignore", message = r"Passing",
												category = FutureWarning)

import numpy as np
import os
import tensorflow as tf
import Constants
import DebugUtils
import HungarianSolver as OptimizationModule
import Utility
from six.moves import xrange

assignmentSolver = None

session = None

def computeCostMatrix(logits, labels):
	"""!
			@brief Function that returns an operator for computing the cost matrix
						 given by the current logits and labels.
						 
						 The cost matrix is computed by subtracting element-wise the x and
						 y coordinates of top left and bottom right corners, thus leading
						 the algorithm to minimize such a distance.
			
			@param logits tensor representing the output logits of the regressor. It
						 must be a 'float32' 2D tensor having shape [batchSize,
						 maxDetectedObjects * detectionRegressionFeatures].
			@param labels tensor representing ground-truth data. It must be a
						 'float32' 2D tensor having shape [batchSize, maxDetectedObjects *
						 detectionBoundingBoxFeatures].
			
			@return a 'float32' 3D tensor having shape [batchSize,
							maxDetectedObjects, maxDetectedObjects] representing the cost
							matrix for associating predictions and ground-truth.
	"""
	
	with tf.name_scope('ComputeCostMatrix'):
		# Reshaping logits so that it can be possible to repeat them for creating
		# a tensor with MAX_DETECTED_OBJECTS elements.
		logitsReshaped = tf.reshape(logits,
													shape = [-1,
																	 Constants.MAX_DETECTED_OBJECTS,
																	 Constants.DETECTION_REGRESSION_FEATURES])
		
		# Extracting first component from the logits (i.e., top left corner x).
		topLeftX = tf.slice(logitsReshaped, [0, 0, 0],
												[-1, int(logitsReshaped.shape[1]), 1])
		
		# Extracting second component from the logits (i.e., top left corner y).
		topLeftY = tf.slice(logitsReshaped, [0, 0, 1],
												[-1, int(logitsReshaped.shape[1]), 1])
		
		# Extracting third component from the logits (i.e., width).
		width = tf.slice(logitsReshaped, [0, 0, 2],
										 [-1, int(logitsReshaped.shape[1]), 1])
		
		# Extracting fourth component from the logits (i.e., height).
		height = tf.slice(logitsReshaped, [0, 0, 3],
											[-1, int(logitsReshaped.shape[1]), 1])
		
		# Computing logits bottom right corner for x coordinate.
		logitsBottomRightX = topLeftX + width
		
		# Computing logits bottom right corner for y coordinate.
		logitsBottomRightY = topLeftY + height
		
		# Repeating logits top left corner for x coordinate in order for creating
		# MAX_DETECTED_OBJECTS elements.
		logitsRepeatTopLeftX = tf.contrib.keras.backend.repeat_elements(topLeftX,
																								 Constants.MAX_DETECTED_OBJECTS,
																								 axis = 2)
		
		# Repeating logits top left corner for y coordinate in order for creating
		# MAX_DETECTED_OBJECTS elements.
		logitsRepeatTopLeftY = tf.contrib.keras.backend.repeat_elements(topLeftY,
																								 Constants.MAX_DETECTED_OBJECTS,
																								 axis = 2)
		
		# Repeating logits bottom right corner for x coordinate in order for
		# creating MAX_DETECTED_OBJECTS elements.
		logitsRepeatBottomRightX = tf.contrib.keras.backend.repeat_elements(
																								 logitsBottomRightX,
																								 Constants.MAX_DETECTED_OBJECTS,
																								 axis = 2)
		
		# Repeating logits bottom right corner for y coordinate in order for
		# creating MAX_DETECTED_OBJECTS elements.
		logitsRepeatBottomRightY = tf.contrib.keras.backend.repeat_elements(
																								 logitsBottomRightY,
																								 Constants.MAX_DETECTED_OBJECTS,
																								 axis = 2)
		
		# Reshaping labels so that it can be possible to repeat them for creating
		# a tensor with MAX_DETECTED_OBJECTS elements.
		labelsReshaped = tf.reshape(labels,
													shape = [-1,
																	 Constants.MAX_DETECTED_OBJECTS,
																	 Constants.DETECTION_BOUNDING_BOX_FEATURES])
		
		# Extracting first component from the labels (i.e., top left corner x).
		topLeftX = tf.slice(labelsReshaped, [0, 0, 0],
												[-1, int(labelsReshaped.shape[1]), 1])
		
		# Extracting second component from the labels (i.e., top left corner y).
		topLeftY = tf.slice(labelsReshaped, [0, 0, 1],
												[-1, int(labelsReshaped.shape[1]), 1])
		
		# Extracting third component from the labels (i.e., width).
		width = tf.slice(labelsReshaped, [0, 0, 2],
											[-1, int(labelsReshaped.shape[1]), 1])
		
		# Extracting fourth component from the labels (i.e., height).
		height = tf.slice(labelsReshaped, [0, 0, 3],
											[-1, int(labelsReshaped.shape[1]), 1])
		
		# Computing labels bottom right corner for x coordinate.
		labelsBottomRightX = tf.reshape(topLeftX + width,
																		shape = [-1, int(labelsReshaped.shape[1])])
		
		# Computing labels bottom right corner for y coordinate.
		labelsBottomRightY = tf.reshape(topLeftY + height,
																		shape = [-1, int(labelsReshaped.shape[1])])
		
		# Reshaping labels top left corner for x coordinate for making next
		# operations easy to be done with standard TensorFlow operations.
		topLeftX = tf.reshape(topLeftX,
													shape = [1, -1, Constants.MAX_DETECTED_OBJECTS])
		
		# Reshaping labels top left corner for y coordinate for making next
		# operations easy to be done with standard TensorFlow operations.
		topLeftY = tf.reshape(topLeftY,
													shape = [1, -1, Constants.MAX_DETECTED_OBJECTS])
		
		# Repeating labels top left corner for x coordinate in order for creating
		# MAX_DETECTED_OBJECTS elements.
		labelsRepeatTopLeftX = tf.tile(topLeftX,
																	 [1, 1, Constants.MAX_DETECTED_OBJECTS])
		
		# Repeating labels top left corner for y coordinate in order for creating
		# MAX_DETECTED_OBJECTS elements.
		labelsRepeatTopLeftY = tf.tile(topLeftY,
																	 [1, 1, Constants.MAX_DETECTED_OBJECTS])
		
		# Reshaping labels top left corner for x coordinate for making next
		# operations easy to be done with standard TensorFlow operations.
		labelsRepeatTopLeftX = tf.reshape(labelsRepeatTopLeftX,
																			shape = [-1,
																							 Constants.MAX_DETECTED_OBJECTS,
																							 Constants.MAX_DETECTED_OBJECTS])
		
		# Reshaping labels top left corner for y coordinate for making next
		# operations easy to be done with standard TensorFlow operations.
		labelsRepeatTopLeftY = tf.reshape(labelsRepeatTopLeftY,
																			shape = [-1,
																							 Constants.MAX_DETECTED_OBJECTS,
																							 Constants.MAX_DETECTED_OBJECTS])
		
		# Repeating labels bottom right corner for x coordinate in order for
		# creating MAX_DETECTED_OBJECTS elements.
		labelsRepeatBottomRightX = tf.tile(labelsBottomRightX,
																			 [1, Constants.MAX_DETECTED_OBJECTS])
		
		# Repeating labels bottom right corner for y coordinate in order for
		# creating MAX_DETECTED_OBJECTS elements.
		labelsRepeatBottomRightY = tf.tile(labelsBottomRightY,
																			 [1, Constants.MAX_DETECTED_OBJECTS])
		
		# Reshaping labels bottom right corner for x coordinate for making next
		# operations easy to be done with standard TensorFlow operations.
		labelsRepeatBottomRightX = tf.reshape(labelsRepeatBottomRightX,
																			 shape = [-1,
																								Constants.MAX_DETECTED_OBJECTS,
																								Constants.MAX_DETECTED_OBJECTS])
		
		# Reshaping labels bottom right corner for y coordinate for making next
		# operations easy to be done with standard TensorFlow operations.
		labelsRepeatBottomRightY = tf.reshape(labelsRepeatBottomRightY,
																			 shape = [-1,
																								Constants.MAX_DETECTED_OBJECTS,
																								Constants.MAX_DETECTED_OBJECTS])
		
		# Computing cost matrix between each prediction (logit) and all targets
		# (labels) by taking advantage of the tensor structure. This is made
		# possible because the labels matrix has a padding of 0-columns for fake
		# targets to reach the same column size of the logits matrix
		# (i.e., MAX_DETECTED_OBJECTS). The difference is computed by subtracting
		# element-wise the top left corner for x coordinate representing a
		# bounding box of a person.
		costMatrixTopLeftX = (tf.abs(tf.subtract(logitsRepeatTopLeftX,
																						 labelsRepeatTopLeftX)) *
																 tf.abs(tf.sign(labelsRepeatTopLeftX)))
		
		# The difference is computed by subtracting element-wise the top left
		# corner for y coordinate representing a bounding box of a person.
		costMatrixTopLeftY = (tf.abs(tf.subtract(logitsRepeatTopLeftY,
																						 labelsRepeatTopLeftY)) *
																 tf.abs(tf.sign(labelsRepeatTopLeftY)))
		
		# The difference is computed by subtracting element-wise the bottom right
		# corner for x coordinate representing a bounding box of a person.
		costMatrixBottomRightX = (tf.abs(tf.subtract(logitsRepeatBottomRightX,
																								 labelsRepeatBottomRightX)) *
																		 tf.abs(tf.sign(labelsRepeatBottomRightX)))
		
		# The difference is computed by subtracting element-wise the bottom right
		# corner for y coordinate representing a bounding box of a person.
		costMatrixBottomRightY = (tf.abs(tf.subtract(logitsRepeatBottomRightY,
																								 labelsRepeatBottomRightY)) *
																		 tf.abs(tf.sign(labelsRepeatBottomRightY)))
		
		# Summing up the differences for both x and y coordinates.
		return (costMatrixTopLeftX + costMatrixTopLeftY +
						costMatrixBottomRightX + costMatrixBottomRightY)

def computeLoss(logitsValue, labelsValue, feedDictionary,
								isPrint = True):
	global assignmentSolver, session
	
	step = 0
	
	costMatrixStep = session.run(computeCostMatrix(logitsValue, labelsValue))
	
	# Updating cost matrix in the AssignmentSolver class.
	session.run(assignmentSolver.updateCostMatrix,
							{ assignmentSolver.costMatrixPlaceholder: costMatrixStep })
	
	# Initializing assignment to a default value in the AssignmentSolver
	# class.
	session.run(assignmentSolver.init)
	
	for step in xrange(Constants.MAX_OPTIMIZATION_STEPS):
		# Performing a step of loss minimization based on the current cost
		# matrix.
		_ = session.run([assignmentSolver.train])
		
		if step == 0:
			# Only for the first step, we set the maximum regularization
			# parameters to their first values.
			session.run(assignmentSolver.updateMaxRegularization)
		
		if step % 100 == 0:
			# Increasing alpha parameter to give more weights to problem
			# constraints during the optimization process.
			session.run(assignmentSolver.increaseAlpha)
		
		if step % 1000 == 0:
			# Checking whether all variables have been correctly assigned.
			res = session.run(assignmentSolver.checkAssignment)
			
			# All variables have been correcly assigned, therefore the
			# optimization process is done.
			if res == True:
				break
	
	finalAssignment = session.run(assignmentSolver.updateFinalAssignment)
	
	if isPrint == True:
		DebugUtils.Err('', '', True)
		DebugUtils.Err('HungarianSolver', 'Cost matrix:', True)
		
		Utility.printArray('HungarianSolver',
											 np.reshape(costMatrixStep[0],
																	[1, Constants.MAX_DETECTED_OBJECTS,
																	 Constants.MAX_DETECTED_OBJECTS]),
											 False, 4, False)
		
		DebugUtils.Err('', '', True)
		DebugUtils.Err('HungarianSolver', 'Assignment:', True)
		
		Utility.printArray('HungarianSolver',
											 np.reshape(finalAssignment[0],
																	[1, Constants.MAX_DETECTED_OBJECTS,
																	 Constants.MAX_DETECTED_OBJECTS]),
											 True, 4, True)
	
	return step

def definePlaceholders():
	with tf.name_scope('Placeholders'):
		logits = tf.placeholder(tf.float32,
														[None, Constants.MAX_DETECTED_OBJECTS *
														 Constants.DETECTION_REGRESSION_FEATURES], 'OutputData')
														 
		labels = tf.placeholder(tf.float32,
														[None, Constants.MAX_DETECTED_OBJECTS *
														 Constants.DETECTION_BOUNDING_BOX_FEATURES], 'Groundtruth')
		
		return logits, labels

def main():
	global assignmentSolver, session
	
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	
	# Creating a session for running Ops on the graph.
	session = tf.Session()
	
	# Defining placeholder which will be used during the computation.
	logits, labels = definePlaceholders()
	
	assignmentSolver = OptimizationModule.HungarianSolver('HungarianSolver', 1,
																												False)
	
	# Adding the variable initializer Op.
	init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
	
	session.run(init)
	
	logitsValue = tf.constant([[ 0., 0., 1., 4., 0.99,  1.,  1.,  2.,  3., 0.99,  2.,  2.,  3.,  4., 0.99, 1., 3., 1., 5., 0.1, 1., 1., 1., 1., 0.1, 2., 1., 2., 4., 0.1],
														 [ 0., 0., 1., 2., 0.99, 10., 10., 20., 30., 0.99, 20., 20., 30., 40., 0.99, 5., 1., 1., 1., 0.1, 1., 1., 1., 1., 0.1, 1., 9., 1., 1., 0.1]])
	
	labelsValue = tf.constant([[ 10., 11., 12., 15., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
														 [  0.,  0.,  1.,  2., 10., 10., 20., 30., 20., 20., 30., 40.,  1., 10.,  9.,  1., -1., -1., -1., -1., -1., -1., -1., -1.]])
	
	# Dictionary containing all data for training.
	feedDictionary = {
		logits: logitsValue,
		labels: labelsValue
	}
	
	for i in range(0,logitsValue.shape[0]):
		DebugUtils.Debug('HungarianSolver', 'Solving assignment problem...', False)
		
		# Computing the loss given by solving the assignment problem.
		steps = computeLoss(logitsValue[i], labelsValue[i], feedDictionary)
		
		DebugUtils.Debug('HungarianSolver', 'done. (%d steps).' %steps, True)

if __name__ == '__main__':
	main()
