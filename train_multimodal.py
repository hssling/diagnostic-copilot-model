import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

# 1. Configuration for Kaggle/HuggingFace Fine-Tuning
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct" # Small, highly capable multimodal model perfect for medical VQA
DATASET_ID = "flaviagiammarino/vqa-rad" # Example Medical VQA dataset (Radiology)
OUTPUT_DIR = "./med-qwen-vl-adapter"

def main():
    print(f"Loading processor and model: {MODEL_ID}")
    
    # Load processor and model with memory-efficient 4-bit quantization
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Apply LoRA (Low-Rank Adaptation)
    print("Applying LoRA parameters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Attention layers
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format the dataset
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train[:50%]") # Use subset for demonstration
    
    def format_data(example):
        # We need to format the inputs as required by the specific model
        # For Qwen2-VL:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example["answer"]}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text, "image": example["image"]}
    
    formatted_dataset = dataset.map(format_data, remove_columns=dataset.column_names)

    # Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100, # Set low for quick Kaggle demonstration
        save_strategy="steps",
        save_steps=50,
        fp16=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none" # Disable wandb for seamless Kaggle runs
    )

    # Custom Data Collator for Vision-Language Models
    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        images = [ex["image"] for ex in examples]
        
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        # Labels are the same as input_ids for standard causal LM training
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # Train using TRL's SFT Trainer
    print("Starting fine-tuning...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        data_collator=collate_fn,
        dataset_text_field="text" # SFTTrainer requires this, though we use a custom collator
    )

    trainer.train()
    
    # Save the adapter
    print(f"Saving fine-tuned adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("Done! You can now merge this adapter or upload it directly to the Hugging Face Hub (e.g. via model.push_to_hub())")

if __name__ == "__main__":
    main()
