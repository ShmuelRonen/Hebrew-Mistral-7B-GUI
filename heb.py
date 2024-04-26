import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import torch

model_name = "yam-peleg/Hebrew-Mistral-7B"
cache_dir = "hebrew_mistral_cache"
os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

torch.backends.cudnn.benchmark = True

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, quantization_config=quantization_config)

def generate_response(input_text, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p):
   input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
   outputs = model.generate(
       **input_ids,
       max_new_tokens=max_new_tokens,
       min_length=min_length,
       no_repeat_ngram_size=no_repeat_ngram_size,
       num_beams=num_beams,
       early_stopping=early_stopping,
       temperature=temperature,
       top_p=top_p,
       pad_token_id=tokenizer.eos_token_id,
       do_sample=True
   )
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   return response

def chat(input_text, history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p):
   user_input = f'<div style="text-align: right;">{input_text}</div>'
   response = generate_response(input_text, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p)
   bot_response = f'<div style="text-align: right;">{response}</div>'
   history.append((user_input, bot_response))
   return history, history

with gr.Blocks() as demo:
   gr.Markdown("# Hebrew-Mistral-7B Chatbot", elem_id="title")
   gr.Markdown("Model by Yam Peleg | GUI by Shmuel Ronen", elem_id="subtitle")
   
   chatbot = gr.Chatbot(elem_id="chatbot")
   
   with gr.Row():
       message = gr.Textbox(placeholder="Type your message...", label="User", elem_id="message")
       submit = gr.Button("Send")

   with gr.Accordion("Adjustments", open=False):
       with gr.Row():    
           with gr.Column():
               max_new_tokens = gr.Slider(minimum=10, maximum=1500, value=100, step=10, label="Max New Tokens")
               min_length = gr.Slider(minimum=10, maximum=300, value=100, step=10, label="Min Length")
               no_repeat_ngram_size = gr.Slider(minimum=1, maximum=6, value=4, step=1, label="No Repeat N-Gram Size")
           with gr.Column():
               num_beams = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Num Beams") 
               temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.2, step=0.1, label="Temperature")
               top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Top P")
       early_stopping = gr.Checkbox(value=True, label="Early Stopping")
   
   submit.click(chat, inputs=[message, chatbot, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p], outputs=[chatbot, chatbot])
   
   demo.css = """
       #message, #message *, .label, #title, #subtitle {
           text-align: right !important;
       }
   """

demo.launch()