import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import AutoConfig
from typing import List
import logging as log
from pathlib import Path
from transformers.generation import GenerationConfig, GenerationMixin
import numpy as np
from openvino.runtime import opset13
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
import PIL
from PIL import Image

from typing import Optional, Tuple, List, Union

import openvino as ov
import nncf
import time

def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [
        key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())
    ]
    key_value_output_names = [
        key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())
    ]
    not_kv_inputs = [
        input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())
    ]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1
    
    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )   

class LlmStatefulModel():
    def __init__(
        self,
        model=None,
        tokenizer=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
        int4_compress=False,
    ):
        self.name = "MoonDream2 LLM Model"
        self.model = model
        self.tokenizer = tokenizer
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.int4_compress = int4_compress
        self.inputs_dict = {}

    def get_model(self):
        return self.model.text_model

    def get_input_names(self):
        inputs = ['attention_mask', 'position_ids']
        for idx in range(24):
            inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        inputs.append('inputs_embeds')
        return inputs

    def get_output_names(self):
        outputs = ['logits']
        for idx in range(len(self.model.text_model.transformer.h)):
            outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])
        return outputs

    def get_dynamic_axes(self):
        pass

    def get_sample_input(self):
            pass
    
    def save_tokenizer(self, tokenizer, out_dir):
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log.error(f'tokenizer loading failed with {e}')

    def convert_sdpa_ov(self):
        llm_model = self.get_model()        
        attention_mask = torch.ones(1, 743)

        llm_input = torch.rand(( 1, 743, 2048), dtype=torch.float32)
        pkv = llm_model(inputs_embeds=llm_input, attention_mask=attention_mask, use_cache=True, return_dict=False)[1]

        attention_mask = torch.ones(1, 743*2)
        import numpy as np
        position_ids = torch.tensor([[743*2-1]])

        llm_model.config.torchscript = True
        ov_model = ov.convert_model(
            llm_model,
            example_input={
                "inputs_embeds":  llm_input,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": pkv,
             },
        )
        # print("stateful model inputs: ", ov_model.inputs)
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        patch_stateful(ov_model)

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_stateful.xml"))
        self.save_tokenizer(self.tokenizer, self.ov_model_path)
        self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1,
            }
            ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(ov_compressed_model, Path(f"{self.ov_model_path}/llm_stateful_int4.xml"))
    
class LlmEmbdModel():
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
    ):
        self.name = "MoonDream2 LLM Embd Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.text_model.transformer.embd

    def get_input_names(self):
        inputs = ['input_ids']
        return inputs

    def get_output_names(self):
        outputs = ['inputs_embeds']
        return outputs

    def get_dynamic_axes(self):
        pass

    def get_sample_input(self):
            pass

    def convert_sdpa_ov(self):
        embd_model = self.get_model()        

        input_ids = torch.tensor([[50256]])

        ov_model = ov.convert_model(
            embd_model,
            example_input={
                "input_ids":  input_ids,
             },
        )

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_embd.xml"))

class VisionEncoderModel():
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
    ):
        self.name = "Vision Encoder Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.vision_encoder.encoder

    def get_input_names(self):
        return ['x']

    def get_output_names(self):
        outputs = ['combined_features']
        return outputs

    def get_dynamic_axes(self):
        return {
            'x': {0:'batch'},
                }

    def get_sample_input(self):
            pass

    def convert_sdpa_ov(self, combined_images=None):
        encoder_model = self.get_model()        
        ov_model = ov.convert_model(
            encoder_model,
            example_input={
                "x": combined_images,
             },
        )
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/vision_encoder.xml"))

        core = ov.Core()
        self.ov_compiled = core.compile_model(ov_model, self.device)
        self.ov_request = self.ov_compiled.create_infer_request()

    def run(self, combined_images=None):
        self.inputs_dict['x'] = combined_images
        self.ov_request.start_async(self.inputs_dict, share_inputs=True)
        self.ov_request.wait()
        return torch.from_numpy(self.ov_request.get_tensor("combined_features").data)
    
class VisionProjectionModel():
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
    ):
        self.name = "Vision Projection Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.vision_encoder.projection

    def get_input_names(self):
        return ['x']

    def get_output_names(self):
        outputs = ['vision_output']
        return outputs

    def get_dynamic_axes(self):
        return {
            'x': {0:'batch'},
                }

    def get_sample_input(self):
            pass

    def convert_sdpa_ov(self, final_features=None):
        encoder_model = self.get_model()        
        ov_model = ov.convert_model(
            encoder_model,
            example_input={
                "x": final_features,
             },
        )
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/vision_projectiton.xml"))

        core = ov.Core()
        self.ov_compiled = core.compile_model(ov_model, self.device)
        self.ov_request = self.ov_compiled.create_infer_request()

    def run(self, final_features=None):
        self.inputs_dict['x'] = final_features
        self.ov_request.start_async(self.inputs_dict, share_inputs=True)
        self.ov_request.wait()
        return torch.from_numpy(self.ov_request.get_tensor("vision_output").data)

def create_patches(image, patch_size=(378, 378)):
    assert image.dim() == 3, "Image must be in CHW format"

    _, height, width = image.shape  # Channels, Height, Width
    patch_height, patch_width = patch_size

    if height == patch_height and width == patch_width:
        return []

    # Iterate over the image and create patches
    patches = []
    for i in range(0, height, patch_height):
        row_patches = []
        for j in range(0, width, patch_width):
            patch = image[:, i : i + patch_height, j : j + patch_width]
            row_patches.append(patch)
        patches.append(torch.stack(row_patches))
    return patches

class Preprocess:
    def __init__(self):
        pass

    def preprocess(self,
                 image: PIL.Image.Image,
                 ):
        self.supported_sizes = [(378, 378), (378, 756), (756, 378), (756, 756)]
        width, height = image.size
        max_dim = max(width, height)
        if max_dim < 512:
            im_size = (378, 378)
        else:
            aspect_ratio = width / height
            im_size = min(
                self.supported_sizes,
                key=lambda size: (
                    abs((size[1] / size[0]) - aspect_ratio),
                    abs(size[0] - width) + abs(size[1] - height),
                ),
            )

        return Compose(
            [
                Resize(size=im_size, interpolation=InterpolationMode.BICUBIC),
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )(image)
        
    def encode(self, images: Union[PIL.Image.Image, list[PIL.Image.Image], torch.Tensor]):
        im_list = None
        if isinstance(images, torch.Tensor):
            # Input must have dimensions (B, C, H, W)
            assert (
                len(images.shape) == 4
            ), "Tensor input must have dimensions (B, C, H, W)"
            im_list = list(images)
        elif isinstance(images, PIL.Image.Image):
            im_list = [images]
        elif isinstance(images, list):
            im_list = images
        else:
            raise ValueError(
                "Input must be a PIL image, list of PIL images, or a tensor"
            )
        
        # Preprocess unless the images are already tensors (indicating that
        # they have already been preprocessed)
        if not isinstance(im_list[0], torch.Tensor):
            im_list = [self.preprocess(im.convert("RGB")) for im in im_list]

        patches = [create_patches(im) for im in im_list]
        flat_patches = [patch for image_patches in patches for patch in image_patches]

        # Images may be variable size, and need to be resized to a common size after
        # creating patches.
        resized_images = [
            F.interpolate(im.unsqueeze(0), size=(378, 378), mode="bilinear")
            for im in im_list
        ]

        combined_images = torch.cat([*resized_images, *flat_patches], dim=0)

        return combined_images, im_list, patches
    
class Middleprocess:
    def __init__(self):
        pass

    def middleprocess(self, combined_features=None, im_list=None, patches=None):
        full_img_features = combined_features[: len(im_list)]
        patch_features = (
            combined_features[len(im_list) :].transpose(1, 2).view(-1, 1152, 27, 27)
        )

        # Reshape patch features back to their original structure
        reshaped_patch_features = []
        patch_idx = 0
        for i, patch_set in enumerate(patches):
            if len(patch_set) == 0:
                reshaped_patch_features.append(
                    full_img_features[i].transpose(0, 1).view(1152, 27, 27)
                )
            else:
                sample_features = []
                for row_patches in patch_set:
                    row_len = len(row_patches)
                    row_features = patch_features[
                        patch_idx : patch_idx + row_len
                    ]  # row_len, T, C
                    row_features = torch.cat(
                        list(row_features), dim=2
                    )  # T, C * row_len
                    patch_idx += row_len
                    sample_features.append(row_features)
                sample_features = torch.cat(sample_features, dim=1)
                sample_features = F.interpolate(
                    sample_features.unsqueeze(0), size=(27, 27), mode="bilinear"
                ).squeeze(0)
                reshaped_patch_features.append(sample_features)
        reshaped_patch_features = (
            torch.stack(reshaped_patch_features).view(-1, 1152, 729).transpose(1, 2)
        )

        final_features = torch.cat([full_img_features, reshaped_patch_features], dim=2)

        return final_features

class MoonDream2_OV:
    def __init__(self, pretrained_model_path=None, model=None, tokenizer=None, ov_model_path='/tmp/moonstream2_ov/', device='CPU', int4_compress=False):

        if model is None and pretrained_model_path:        
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_path,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path, 
                trust_remote_code=True
            )
        elif model and tokenizer and pretrained_model_path is None:
            self.model = model
            self.tokenizer = tokenizer

        self.int4_compress = int4_compress
        self.vision_encoder_model = VisionEncoderModel(model=self.model, ov_model_path=ov_model_path, device=device)
        self.vision_projection_model = VisionProjectionModel(model=self.model, ov_model_path=ov_model_path, device=device)

        self.llm_embd_model = LlmEmbdModel(model=self.model, ov_model_path=ov_model_path, device=device)
        self.llm_stateful_model = LlmStatefulModel(model=self.model, tokenizer= self.tokenizer, ov_model_path=ov_model_path, device=device, int4_compress=self.int4_compress)

        self.vision_pre_process = Preprocess()
        self.vision_middle_process = Middleprocess()

    def export_vision_to_ov(self, image_path):
        combined_images, im_list, patches = self.vision_pre_process.encode(Image.open(image_path))

        self.vision_encoder_model.convert_sdpa_ov(combined_images=combined_images)
        combined_features = self.vision_encoder_model.run(combined_images=combined_images)

        final_features = self.vision_middle_process.middleprocess(combined_features, im_list, patches)

        self.vision_projection_model.convert_sdpa_ov(final_features=final_features)
        self.llm_stateful_model.convert_sdpa_ov()
        self.llm_embd_model.convert_sdpa_ov()

class OVMoonDreamForCausalLM(GenerationMixin):
    def __init__(
        self,
        core=None,
        ov_model_path=None,
        device='CPU',
        int4_compress=False,
        llm_infer_list=[],
    ):
        self.ov_model_path = ov_model_path
        self.core = core
        self.ov_device = device
        self.int4_compress = int4_compress

        if int4_compress and 'CPU' in device:
            self.llm_model = core.read_model(Path(f"{ov_model_path}/llm_stateful_int4.xml"))
            self.llm_compiled_model = core.compile_model(self.llm_model, device)
        else:
            self.llm_model = core.read_model(Path(f"{ov_model_path}/llm_stateful.xml"))
            self.llm_compiled_model = core.compile_model(self.llm_model, device, config = {'INFERENCE_PRECISION_HINT': 'f32'})
            
        self.llm_request = self.llm_compiled_model.create_infer_request()

        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.llm_model.inputs)}
        self.output_names = {idx: key for idx, key in enumerate(self.llm_model.outputs)}
        self.key_value_input_names = [key for key in list(self.input_names) if key not in ["beam_idx", "inputs_embeds", "attention_mask", "position_ids"]]
        self.key_value_output_names = [key for key in list(self.output_names)[1:]]
        self.stateful = len(self.key_value_input_names) == 0
        # self.compiled_model = core.compile_model(self.model, device, config = {'INFERENCE_PRECISION_HINT': 'f32'})

        self.config = AutoConfig.from_pretrained(ov_model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.device = torch.device("cpu")
        self.next_beam_idx = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.past_len = None
        self.main_input_name = "input_ids"
        self._supports_cache_class = False

        self.llm_embd = core.read_model(Path(f"{ov_model_path}/llm_embd.xml"))
        self.llm_embd_compiled_model = core.compile_model(self.llm_embd, 'CPU')
        self.llm_embd_request = self.llm_embd_compiled_model.create_infer_request()
        
        self.tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True)

        self.vision_model_init()

        self.llm_infer_list = llm_infer_list
 

    def vision_model_init(self):
        self.vision_encoder_model = self.core.read_model(Path(f"{self.ov_model_path}/vision_encoder.xml"))
        self.vision_encoder_compiled_model = self.core.compile_model(self.vision_encoder_model, self.ov_device)
        self.vision_encoder_request = self.vision_encoder_compiled_model.create_infer_request()

        self.vision_projectiton_model = self.core.read_model(Path(f"{self.ov_model_path}/vision_projectiton.xml"))
        self.vision_projectiton_compiled_model = self.core.compile_model(self.vision_projectiton_model, self.ov_device)
        self.vision_projectiton_request = self.vision_projectiton_compiled_model.create_infer_request()

        self.vision_pre_process = Preprocess()
        self.vision_middle_process = Middleprocess()

    def vision_encoder_run(self, combined_images=None):
        inputs_dict = {}
        inputs_dict['x'] = combined_images
        self.vision_encoder_request.start_async(inputs_dict, share_inputs=True)
        self.vision_encoder_request.wait()
        return torch.from_numpy(self.vision_encoder_request.get_tensor("combined_features").data)
    
    def vision_projectiton_run(self, final_features=None):
        inputs_dict = {}
        inputs_dict['x'] = final_features
        self.vision_projectiton_request.start_async(inputs_dict, share_inputs=True)
        self.vision_projectiton_request.wait()
        return torch.from_numpy(self.vision_projectiton_request.get_tensor("vision_output").data)

    def vision_model(self, image):
        image = Image.open(image)
        combined_images, im_list, patches = self.vision_pre_process.encode(image)
        combined_features = self.vision_encoder_run(combined_images=combined_images)
        final_features = self.vision_middle_process.middleprocess(combined_features, im_list, patches)
        enc_image = self.vision_projectiton_run(final_features=final_features)

        return enc_image

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True
    
    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def llm_embd_run(self, input_ids):
        llm_embd_inputs = {}
        llm_embd_inputs['input_ids'] = input_ids

        self.llm_embd_request.start_async(llm_embd_inputs, share_inputs=True)
        self.llm_embd_request.wait()

        return torch.from_numpy(self.llm_embd_request.get_tensor("inputs_embeds").data)

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """General inference method"""
        inputs_dict = {}
        if past_key_values is not None:
            inputs_embeds = self.llm_embd_run(input_ids)
            inputs_dict['inputs_embeds'] = inputs_embeds
        else:
            self.past_len = 0
            self.llm_request.reset_state()
            inputs_dict['inputs_embeds'] = inputs_embeds

        inputs_dict["attention_mask"] = attention_mask
        inputs_dict["position_ids"] = position_ids

        batch_size = inputs_embeds.shape[0]
        if "beam_idx" in self.input_names:
            inputs_dict["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        # print('attention_mask: ', inputs_dict['attention_mask'].shape)
        # print('position_ids: ', inputs_dict['position_ids'])
        # print('inputs_embeds: ', inputs_dict['inputs_embeds'])
        start = time.perf_counter()
        self.llm_request.start_async(inputs_dict, share_inputs=True)
        self.llm_request.wait()
        end = time.perf_counter()

        generation_time = (end - start) * 1000
        self.llm_infer_list.append(generation_time)

        past_key_values = ((),)
        self.past_len += inputs_dict["inputs_embeds"].shape[1]

        # print('logits: ', self.request.get_tensor("logits").data)
        return CausalLMOutputWithPast(
            loss=None,
            logits=torch.from_numpy(self.llm_request.get_tensor("logits").data),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )   

    def input_embeds(self, prompt, image_embeds):
        def _tokenize(txt):
            return self.tokenizer(
                txt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
        embeds = []
        embeds.append(self.llm_embd_run(torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)))

        if "<image>" not in prompt:
            embeds.append(self.llm_embd_run(_tokenize(prompt)))
        else:
            assert prompt.count("<image>") == 1
            before, after = prompt.split("<image>")
            if len(before) > 0:
                embeds.append(self.llm_embd_run(_tokenize(before)))
            embeds.append(image_embeds.to(self.device))
            if len(after) > 0:
                embeds.append(self.llm_embd_run(_tokenize(after)))

        return torch.cat(embeds, dim=1)
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            cache_length = past_length = self.past_len
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - self.past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif self.past_len < input_ids.shape[1]:
                input_ids = input_ids[:, self.past_len:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]
        else:
            self.llm_infer_list.clear()

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and (input_ids is None or input_ids.shape[1] == 0):
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:    
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    
