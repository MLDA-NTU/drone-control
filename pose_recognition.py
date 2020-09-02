import tflite_runtime.interpreter as tflite
import numpy as np

class PoseNet:
    def __init__(self,threshold):
        model_path = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.threshold = threshold
    
    def get_confidence_scores(self):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        heatmap = self.interpreter.get_tensor(output_details[0]['index'])
        offset = self.interpreter.get_tensor(output_details[1]['index'])
        #the output is a heatmap for every features from the 17 features with an output stride

        joint_num = heatmap_data.shape[-1]
        pose_kps = np.zeros((joint_num,3), np.uint32)

        for i in range(heatmap_data.shape[-1]):

            joint_heatmap = heatmap_data[...,i]
            max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
            remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
            pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
            pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
            max_prob = np.max(joint_heatmap)

            if max_prob > threshold:
                if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
                    pose_kps[i,2] = 1

    return pose_kps


        return heatmap, offset
    
