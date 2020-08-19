import tflite_runtime.interpreter as tflite
import numpy as np

class pose_net:
    def __init__(self):
        model_path = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    
    @staticmethod
    def get_confidence_scores():
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        heatmap = interpreter.get_tensor(output_details[0]['index'])
        offset = interpreter.get_tensor(output_details[1]['index'])
        #the output is a heatmap for every features from the 17 features with an output stride
        print(output_data.shape)