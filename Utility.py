"""!
		@brief Set of utility functions which can be used to perform a specific
					 action. They have not been defined in a class because they should be
					 used as static function (i.e., no need to create an object).
"""

import DebugUtils

def printArray(module, array, isAssignment, maxDigits, isInt):
	"""!
			@brief Function that prints a 3D array having shape [batchSize, rows,
						 columns] in a human-readable format.
						 
						 In case the 3D array does not represent an assignment, elements of
						 the rows will be printed by interleaving white and yellow colors,
						 whilst in the other case, all 0 elements of the rows will be
						 printed in red color whilst all 1 elements in yellow color.
						 
						 During the printing, the array is splitted at a given split column
						 character - that can be configured - so that it can be also
						 correctly printed on a paper, if needed. Elements are also padded
						 by considering a maximum digits value, that can be configured as
						 well.
			
			@param module module that is calling the function.
			@param array 3D array having shape [batchSize, rows, columns].
			@param isAssignment boolean representing whether the array is an
						 assignment or not.
			@param maxDigits maximum digits to represent a number.
			@param isInt boolean representing whether the number will be printed as
						 int or float.
			
			@return no value returned.
	"""
	
	SPLIT_COLUMN_CHARACTERS = 100
	
	if isInt == False:
		# In order to consider the dot and two decimals.
		maxDigits += 3
	
	for i in range(0,array.shape[0]):
		if i == 0:
			if isAssignment == True:
				DebugUtils.Err(module, '  [', False)
			else:
				DebugUtils.Info(module, '  [', False)
		else:
			DebugUtils.Info('', '   ', False)
		
		for j in range(0,array.shape[1]):
			if isAssignment == True:
				if i % 2 == 0:
					if j == 0:
						DebugUtils.Err('', '[[', False)
					else:
						DebugUtils.Err(module, '    [', False)
				else:
					if j == 0:
						DebugUtils.Debug('', '[[', False)
					else:
						DebugUtils.Debug(module, '    [', False)
			else:
				if j % 2 == 0:
					if j == 0:
						DebugUtils.Info('', '[[', False)
					else:
						DebugUtils.Info(module, '    [', False)
				else:
					if j == 0:
						DebugUtils.Warn('', '[[', False)
					else:
						DebugUtils.Warn(module, '    [', False)
			
			# We add a value of 10 characters to consider also all open square
			# parentheses.
			totalCharacters = len(module) + 10
			
			for k in range(0,array.shape[2]):
				if isInt == True:
					numberLength = len(str(int(array[i][j][k])))
				else:
					numberLength = len(str("{0:.2f}".format(array[i][j][k])))
				
				totalCharacters += numberLength
				
				if ((totalCharacters + maxDigits - numberLength + 2) >
						SPLIT_COLUMN_CHARACTERS):
					totalCharacters = len(module) + 10
					
					DebugUtils.Info('', '', True)
					DebugUtils.Info(module, '     ', False)
				
				for _ in range(numberLength,maxDigits):
					DebugUtils.Info('', ' ', False)
					
					totalCharacters += 1
				
				if isAssignment == True:
					if array[i][j][k] == 0:
						if i % 2 == 0:
							DebugUtils.Err('', ' %d' %array[i][j][k], False)
						else:
							DebugUtils.Debug('', ' %d' %array[i][j][k], False)
					else:
						DebugUtils.Warn('', ' %d' %array[i][j][k], False)
				else:
					if isInt == True:
						if j % 2 == 0:
							DebugUtils.Info('', ' %d' %array[i][j][k], False)
						else:
							DebugUtils.Warn('', ' %d' %array[i][j][k], False)
					else:
						if j % 2 == 0:
							DebugUtils.Info('', ' %.2f' %array[i][j][k], False)
						else:
							DebugUtils.Warn('', ' %.2f' %array[i][j][k], False)
				
				if (k + 1) < array.shape[2]:
					totalCharacters += 2
					
					if isAssignment == True:
						if i % 2 == 0:
							DebugUtils.Err('', ',', False)
						else:
							DebugUtils.Debug('', ',', False)
					else:
						if j % 2 == 0:
							DebugUtils.Info('', ',', False)
						else:
							DebugUtils.Warn('', ',', False)
			
			if isAssignment == True:
				if i % 2 == 0:
					DebugUtils.Err('', ' ]', False)
				else:
					DebugUtils.Debug('', ' ]', False)
			else:
				if j % 2 == 0:
					DebugUtils.Info('', ' ]', False)
				else:
					DebugUtils.Warn('', ' ]', False)
			
			if (j + 1) < array.shape[1]:
				if isAssignment == True:
					if i % 2 == 0:
						DebugUtils.Err('', ',', True)
					else:
						DebugUtils.Debug('', ',', True)
				else:
					if j % 2 == 0:
						DebugUtils.Info('', ',', True)
					else:
						DebugUtils.Warn('', ',', True)
			else:
				if isAssignment == True:
					if (i + 1) < array.shape[0]:
						if i % 2 == 0:
							DebugUtils.Err('', '],', True)
						else:
							DebugUtils.Debug('', '],', True)
					else:
						if i % 2 == 0:
							DebugUtils.Err('', ']]', True)
						else:
							DebugUtils.Debug('', ']]', True)
				else:
					if (i + 1) < array.shape[0]:
						if j % 2 == 0:
							DebugUtils.Info('', '],', True)
						else:
							DebugUtils.Warn('', '],', True)
					else:
						if j % 2 == 0:
							DebugUtils.Info('', ']]', True)
						else:
							DebugUtils.Warn('', ']]', True)
