## Update Notes
### 2024/08/01
1. vikhyatk/moondream2 model supports using openvino to accelerate the inference process. Currently only verified on Linux system and only tested on CPU platform.

## Running Guide
### Installation


```bash
git clone https://github.com/zhaohb/moondream2-ov.git
pip install openvino_dev 
pip install nncf
pip install transformers==4.43.2
pip install torch
pip install torchvision
```
### Convert moondream2 model to OpenVINO™ IR(Intermediate Representation) and testing:
```shell
cd moondream2-ov
python3 test_ov_moondream2.py -m /path/to/moondream2 -o /path/to/moondream2_ov
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

