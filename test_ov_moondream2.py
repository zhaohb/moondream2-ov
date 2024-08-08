import argparse
import openvino as ov
from pathlib import Path
from ov_moondream2 import OVMoonDreamForCausalLM, MoonDream2_OV
from transformers import TextStreamer
import time
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Export moondream2 Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument("-o", "--output_dir", required=True, help="output directory for saving model")
    parser.add_argument('-d', '--device', default='CPU', help='inference device')
    parser.add_argument('-pic', '--picture', default="./moondream.jpg", help='picture file')
    parser.add_argument('-p', '--prompt', default="Describe this image.", help='prompt')
    parser.add_argument('-max', '--max_new_tokens', default=256, help='max_new_tokens')
    parser.add_argument('-int4', '--int4_compress', action="store_true", help='int4 weights compress')

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
    elif Path(ov_model_path).exists() and int4_compress is True and not Path(f"{ov_model_path}/llm_stateful_int4.xml").exists():
        moondream2_ov = MoonDream2_OV(pretrained_model_path=model_id, ov_model_path=ov_model_path, device=device, int4_compress=int4_compress)
        moondream2_ov.export_vision_to_ov(picture_path)
        del moondream2_ov.model
        del moondream2_ov.tokenizer
        del moondream2_ov

    llm_infer_list = []

    core = ov.Core()
    #set cache
    core.set_property({'CACHE_DIR': "moondream2_cache"})

    moondream2_model = OVMoonDreamForCausalLM(core=core, ov_model_path=ov_model_path, device=device, int4_compress=int4_compress, llm_infer_list=llm_infer_list)

    version = ov.get_version()
    print("OpenVINO version \n", version)

    for i in range(2):
        vision_start = time.perf_counter()
        enc_image = moondream2_model.vision_model(picture_path)
        vision_end = time.perf_counter()
        vision_infer_time = ((vision_end - vision_start) * 1000)

        chat_history=""
        prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"


        start = time.perf_counter()
        inputs_embeds=moondream2_model.input_embeds(prompt, enc_image)
        end = time.perf_counter()
        embeds_infer_time = ((end - start) * 1000)

        generate_config = {
                "eos_token_id": moondream2_model.tokenizer.eos_token_id,
                "bos_token_id": moondream2_model.tokenizer.bos_token_id,
                "pad_token_id": moondream2_model.tokenizer.bos_token_id,
                "max_new_tokens": max_new_tokens,
            }
        streamer = TextStreamer(moondream2_model.tokenizer, skip_special_tokens=True, skip_prompt=True)
        output_ids = moondream2_model.generate( 
                    inputs_embeds=inputs_embeds, **generate_config, streamer=streamer
                )
        llm_end = time.perf_counter()
        
        #print(moondream2_model.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        
        ## i= 0 is warming up
        if i != 0:
            print("\n\n")
            if len(llm_infer_list) > 1:
                avg_token = sum(llm_infer_list[1:]) / (len(llm_infer_list) - 1)
                print(f"Inputs len {inputs_embeds.shape[1]}, First token latency: {llm_infer_list[0]:.2f} ms, Output len {len(llm_infer_list) - 1}, Avage token latency: {avg_token:.2f} ms")
                print(f"visin latency: {vision_infer_time:.2f} ms, embeds latency : {embeds_infer_time:.2f} ms")
            print("e2e latency: ", sum(llm_infer_list) + vision_infer_time + embeds_infer_time)
