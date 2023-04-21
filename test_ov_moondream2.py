import argparse
import openvino as ov
from pathlib import Path
from ov_moondream2 import OVMoonDreamForCausalLM, MoonDream2_OV
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Export minicpm-v2 Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument("-o", "--output_dir", required=True, help="output directory for saving model")
    parser.add_argument('-d', '--device', default='CPU', help='inference device')
    parser.add_argument('-pic', '--picture', default="./moondream.jpg", help='picture file')
    parser.add_argument('-p', '--prompt', default="Describe this image.", help='prompt')
    parser.add_argument('-max', '--max_new_tokens', default=256, help='max_new_tokens')
    parser.add_argument('-int4', '--int4_compress', default=False, help='int4 weights compress')

    args = parser.parse_args()
    model_id = args.model_id
    ov_model_path = args.output_dir
    device = args.device
    max_new_tokens = args.max_new_tokens
    picture_path = args.picture
    question = args.prompt
    int4_compress = args.int4_compress

    if not Path(ov_model_path).exists():
        moondream2_ov = MoonDream2_OV(pretrained_model_path=model_id, ov_model_path=ov_model_path, device=device, int4_compress=int4_compress)
        moondream2_ov.export_vision_to_ov(picture_path)
        del moondream2_ov.model
        del moondream2_ov.tokenizer
        del moondream2_ov

    core = ov.Core()

    moondream2_model = OVMoonDreamForCausalLM(core=core, ov_model_path=ov_model_path, device=device, int4_compress=int4_compress)

    enc_image = moondream2_model.vision_model(picture_path)

    chat_history=""
    prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"

    inputs_embeds=moondream2_model.input_embeds(prompt, enc_image)

    generate_config = {
            "eos_token_id": moondream2_model.tokenizer.eos_token_id,
            "bos_token_id": moondream2_model.tokenizer.bos_token_id,
            "pad_token_id": moondream2_model.tokenizer.bos_token_id,
            "max_new_tokens": max_new_tokens,
        }
    output_ids = moondream2_model.generate( 
                inputs_embeds=inputs_embeds, **generate_config
            )
    
    print(moondream2_model.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
