import pyevaluatedepth_lib as dlib
import numpy as np
import sys
epsilon = sys.float_info.epsilon

image = np.zeros((100,100)).astype(np.float32)

image += epsilon

#image[20,20] = 0.0000001


errors = dlib.depthError(image, image+0.01)

print(errors)

all_errors = [errors, errors, errors]

#results = dlib.evaluateErrors(all_errors)

print(results)
