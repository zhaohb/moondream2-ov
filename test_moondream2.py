from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

#model_id = "vikhyatk/moondream2"
#revision = "2024-07-23"
model = AutoModelForCausalLM.from_pretrained('./moondream2', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('./moondream2', trust_remote_code=True)

image = Image.open('./moondream.jpg')

enc_image = model.encode_image(image)
print(model.answer_question(enc_image, "Describe this image.", tokenizer))


