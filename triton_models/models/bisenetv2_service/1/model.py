import triton_python_backend_utils as pb_utils
from .resources import BiSeNetV2
from pathlib import Path
import os
import json
import torch
ROOT = Path(__file__).parent

class TritonPythonModel:

    def initialize(self,args):
        
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "output_mask")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
       
        # Load the model
        self.checkpoint = os.path.join(ROOT,"resources","best.pt")
        self.model = BiSeNetV2(use_aux_heads=False)
        self.model.load_checkpoint(self.checkpoint)
        self.model.cuda()
        self.model.eval()

    def execute(self,requests):
        """Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        output0_dtype = self.output0_dtype

        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()

            mask = self.model(self.preprocess(in_0)).cpu().detach().numpy()
            mask = pb_utils.Tensor("output_mask", mask.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[mask])
            responses.append(inference_response)

        return responses

    
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    
    def preprocess(self,image):
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
        image = ( torch.tensor(image) - mean ) / std
        return image.cuda()