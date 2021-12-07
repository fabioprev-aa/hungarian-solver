###############################################################################
#														 DETECTION PARAMETERS															#
###############################################################################

# Maximum number of objects that can be detected in an image.
MAX_DETECTED_OBJECTS = 6

# Number of features for the detection problem representing the bounding box
# (i.e., for each bounding box we have [topLeftX, topLeftY, width, height]).
DETECTION_BOUNDING_BOX_FEATURES = 4

# Number of features for the detection feature extraction regression problem
# (i.e., for each bounding box we have [topLeftX, topLeftY, width, height,
# confidency]).
DETECTION_REGRESSION_FEATURES = DETECTION_BOUNDING_BOX_FEATURES + 1

# Maximum number of steps for the optimizer to find a valid assignment.
MAX_OPTIMIZATION_STEPS = 30000
