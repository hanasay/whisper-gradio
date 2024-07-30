from faster_whisper import WhisperModel
from queue import Queue
from threading import Thread
import gradio as gr
import argparse

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
parser.add_argument(
    '--model_path', 
    type=str, 
    required=False, 
    default='openai/whisper-small', 
    help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
)
parser.add_argument(
    '--lang', 
    type=str, 
    required=False, 
    default='zh', 
    help='Language type to inference'
)
parser.add_argument(
    '--beam_size', 
    type=int, 
    required=False, 
    default=5, 
    help='Size of beam search'
)
parser.add_argument(
    '--repetition_penalty', 
    type=float, 
    required=False, 
    default=1.1, 
    help='Penalty of repeat in output'
)
parser.add_argument(
    '--temperature', 
    type=float, 
    required=False, 
    default=0,
)

args = parser.parse_args()

model = WhisperModel(args.model_path, device="cpu", compute_type="auto")

def transcribe(audio):
    utt = audio.split('/')[-1]
    print(f'Processing: {utt}')
    segments, info = model.transcribe(
        audio=audio,
        language=args.lang,
        beam_size=5,
        best_of=5,
        patience=1,
        repetition_penalty=1.1,
        temperature=0,
        compression_ratio_threshold=2.0,
        without_timestamps=True,
    )
    text = ''
    for seg in segments:
        text += seg.text
    return text

with gr.Blocks() as demo:
    gr.HTML(f'<div style="display:inline; text-align:center"><h1>Taigi Whisper ckpt-17000</h1></div>')
    with gr.Row():
        with gr.Column(scale=4):
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath")
            submit_btn = gr.Button(value="Submit")
            
        with gr.Column(scale=2):
            output = gr.Textbox(label = "output")
    with gr.Row():
        submit_btn.click(
            transcribe, inputs=audio, outputs=output, api_name=False
        )
        examples = gr.Examples(
            examples=[
            ],
            inputs=audio,
        )
            
demo.queue().launch(
        share=True,
        server_name='0.0.0.0',
        server_port=19324)
