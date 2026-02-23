import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

# 1. HuggingFace Space Deployment Settings
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct" # Base model
# To use your fine-tuned model from Kaggle:
# 1. model.push_to_hub("your-name/med-qwen-vl-adapter")
# 2. Add adapter load here for PEFT
ADAPTER_ID = "hssling/med-qwen-vl-adapter"

# Initialize Model and Processor globally
print("Starting App Engine...")
print(f"Loading {MODEL_ID}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

if ADAPTER_ID:
    print(f"Loading custom fine-tuned LoRA weights: {ADAPTER_ID}")
    model.load_adapter(ADAPTER_ID)

# 2. Main API Function called by our Next App
def diagnose_api(history: str, examination: str, image: Image.Image = None, audio_path: str = None, temp: float = 0.2, max_tokens: int = 1500):
    try:
        if image is None:
            # Fallback if no image is passed
            return "Error: Qwen-VL requires at least one image/diagnostic input to function accurately."

        # Re-construct the specific structured prompt our diagnostic copilot uses
        system_prompt = "You are a highly advanced Multi-Modal Diagnostic Co-Pilot Medical AI. Provide ## Integrated Analysis, ## Decision Making, and ## Management & Treatment Plan."
        user_prompt = f"History: {history}\nExamination: {examination}\nAnalyze the provided scan and history."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=int(max_tokens), temperature=float(temp), top_p=0.9, do_sample=True)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return output_text

    except Exception as e:
        return f"Model Error: {str(e)}"

# 3. Create the Gradio interface
# This acts as the visual UI for the HF Space, but more importantly,
# exposes an API endpoint via `/api/predict` that our React app can connect to securely.
demo = gr.Interface(
    fn=diagnose_api,
    inputs=[
        gr.Textbox(lines=5, label="Patient History (String)", placeholder="Age, symptoms, past medical history..."),
        gr.Textbox(lines=5, label="Examination Findings (String)", placeholder="Vitals, systemic exam..."),
        gr.Image(type="pil", label="Diagnostic Scan / Image"),
        gr.Audio(type="filepath", label="Optional Dictation Audio", visible=False),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Temperature (Creativity)"),
        gr.Slider(minimum=256, maximum=4096, value=1500, step=256, label="Max Output Tokens")
    ],
    outputs=gr.Markdown(label="Clinical Report Output"),
    title="Multi-Modal Diagnostic Co-Pilot API (Trained via Kaggle)",
    description="This Space hosts the fine-tuned medical vision-language model for the Diagnostic Co-Pilot ecosystem."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
