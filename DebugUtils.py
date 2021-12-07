"""!
		@brief Set of functions to print on terminal with a specific format. In
					 particular, there is the possibility to specify the module that is
					 printing, the color of the print and whether we want a new line after
					 the print or not.
"""

import sys

def Debug(module, x, isNewLine):
	"""!
			@brief Function that defines a useful print with a blue color.
			
			@param module name of the module that is printing.
			@param x string to be printed on terminal.
			@param isNewLine boolean to whether pushing new line or not.
			
			@return no value returned.
	"""
	
	if module != '':
		print("\033[22;36;1m", "[" + module + "]: \033[0m", sep = '', end = '')
		print("\033[22;34;1m", str(x), "\033[0m", sep = '', end = '')
	else:
		print("\033[22;34;1m", x, "\033[0m", sep = '', end = '')
	
	if (isNewLine == True):
		print()
	else:
		sys.stdout.flush()

def Err(module, x, isNewLine):
	"""!
			@brief Function that defines a useful print with a red color.
			
			@param module name of the module that is printing.
			@param x string to be printed on terminal.
			@param isNewLine boolean to whether pushing new line or not.
			
			@return no value returned.
	"""
	
	if module != '':
		print("\033[22;36;1m", "[" + module + "]: \033[0m", sep = '', end = '')
		print("\033[22;31;1m", str(x), "\033[0m", sep = '', end = '')
	else:
		print("\033[22;31;1m", x, "\033[0m", sep = '', end = '')
	
	if (isNewLine == True):
		print()
	else:
		sys.stdout.flush()

def Info(module, x, isNewLine):
	"""!
			@brief Function that defines a useful print with a white color.
			
			@param module name of the module that is printing.
			@param x string to be printed on terminal.
			@param isNewLine boolean to whether pushing new line or not.
			
			@return no value returned.
	"""
	
	if module != '':
		print("\033[22;36;1m", "[" + module + "]: \033[0m", sep = '', end = '')
		print("\033[22;37;1m", str(x), "\033[0m", sep = '', end = '')
	else:
		print("\033[22;37;1m", x, "\033[0m", sep = '', end = '')
	
	if (isNewLine == True):
		print()
	else:
		sys.stdout.flush()

def Log(module, x, isNewLine):
	"""!
			@brief Function that defines a useful print with a green color.
			
			@param module name of the module that is printing.
			@param x string to be printed on terminal.
			@param isNewLine boolean to whether pushing new line or not.
			
			@return no value returned.
	"""
	
	if module != '':
		print("\033[22;36;1m", "[" + module + "]: \033[0m", sep = '', end = '')
		print("\033[22;38;1m", str(x), "\033[0m", sep = '', end = '')
	else:
		print("\033[22;38;1m", x, "\033[0m", sep = '', end = '')
	
	if (isNewLine == True):
		print()
	else:
		sys.stdout.flush()

def Warn(module, x, isNewLine):
	"""!
			@brief Function that defines a useful print with a yellow color.
			
			@param module name of the module that is printing.
			@param x string to be printed on terminal.
			@param isNewLine boolean to whether pushing new line or not.
			
			@return no value returned.
	"""
	
	if module != '':
		print("\033[22;36;1m", "[" + module + "]: \033[0m", sep = '', end = '')
		print("\033[22;33;1m", str(x), "\033[0m", sep = '', end = '')
	else:
		print("\033[22;33;1m", x, "\033[0m", sep = '', end = '')
	
	if (isNewLine == True):
		print()
	else:
		sys.stdout.flush()
