## Update Notes
### 2024/08/01
1. vikhyatk/moondream2 model supports using openvino to accelerate the inference process. Currently only verified on Linux system and only tested on CPU platform.
### 2024/08/08
1. Now supports Intel's ARC770 GPU

## Running Guide
### Installation


```bash
git clone https://github.com/zhaohb/moondream2-ov.git
pip install -r requirements.txt
pip install --pre -U openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```
### Convert moondream2 model to OpenVINO™ IR(Intermediate Representation) and testing:
```shell
cd moondream2-ov
#for cpu
python3 test_ov_moondream2.py -m /path/to/moondream2 -o /path/to/moondream2_ov

#for gpu
python3 test_ov_moondream2.py -m /path/to/moondream2 -o /path/to/moondream2_ov -d GPU.1

#output
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
OpenVINO version 
 2024.5.0-16678-090da7b5376
The attention mask is not set and cannot be inferred from input because pad token is same as eos token.As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
 The image shows a modern, minimalist desk with a laptop on top, a lamp on the left side, and a rug on the right side.
```
### Note:
After the command is executed, the IR of OpenVINO will be saved in the directory /path/to/moondream2_ov. If the existence of /path/to/moondream2_ov is detected, the model conversion process will be skipped and the IR of OpenVINO will be loaded directly.

The commit of our test model is 48be9138e0faaec8802519b1b828350e33525d46: [Model link](https://hf-mirror.com/vikhyatk/moondream2/commit/48be9138e0faaec8802519b1b828350e33525d46)
### Parsing test_ov_moondream2.py's arguments :
```shell
usage: Export moondream2 Model to IR [-h] -m MODEL_ID -o OUTPUT_DIR [-d DEVICE] [-pic PICTURE] [-p PROMPT] [-max MAX_NEW_TOKENS] [-int4]

options:
  -h, --help            show this help message and exit
  -m MODEL_ID, --model_id MODEL_ID
                        model_id or directory for loading
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        output directory for saving model
  -d DEVICE, --device DEVICE
                        inference device
  -pic PICTURE, --picture PICTURE
                        picture file
  -p PROMPT, --prompt PROMPT
                        prompt
  -max MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        max_new_tokens
  -int4, --int4_compress
                        int4 weights compress
```

