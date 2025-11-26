# inspect_model.py
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INPUT DETAILS:")
for d in input_details:
    print("  index:", d['index'])
    print("  shape:", d['shape'])  # e.g. [1,224,224,3]
    print("  dtype:", d['dtype'])
    print("  name:", d['name'])
    print()

print("OUTPUT DETAILS:")
for d in output_details:
    print("  index:", d['index'])
    print("  shape:", d['shape'])
    print("  dtype:", d['dtype'])
    print("  name:", d['name'])
    print()
