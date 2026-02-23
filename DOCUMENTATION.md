# AI Copilot Backend & Pipeline Management

**Multi-Modal Continuous Learning Pipeline**

This repository handles the intensive GPU backend for the Diagnostic Co-Pilot web application.

Rather than sending 100% of our patient inputs to generic APIs (like Gemini or OpenAI), this ecosystem trains a tailored Medical Vision-Language Model efficiently on limited memory (like Kaggle's T4x2 GPU).

## Architecture Components:

1. **`app.py`**: A Hugging Face Space Gradio inference API. It hosts our specialized code and safely answers WebSocket requests from our main React frontend with heavily formatted Multi-Modal Diagnostics.
2. **`train_multimodal.py`**: The fully automated Python training pipeline. It clones this repository inside a runtime environment (like Kaggle), dynamically loads `Qwen2-VL` into an explosive 4-bit footprint, slices off a tiny Low-Rank Adaptation (LoRA) layer (`adapter_model.safetensors`), trains it on medical images (`flaviagiammarino/vqa-rad`), and automatically pushes the weights back out to the Hugging Face Hub.
3. **`.github/workflows/sync_to_hub.yml`**: Automates pushing GitHub commits straight to Hugging Face Spaces seamlessly!

## Setting Up Your Ecosystem:

### Part 1: Linking GitHub to Hugging Face

Any time you update the `app.py` code here, you want it to push to Hugging Face instantly:

1. Log into your [Hugging Face Settings -> Access Tokens](https://huggingface.co/settings/tokens) and generate a **Write** Token (e.g. `hf_...`).
2. Go to your GitHub repository -> Settings -> Secrets and Variables -> Actions -> **New Repository Secret**. Name it `HF_TOKEN` and paste the token from Step 1.
3. Create an empty Hugging Face Space: Head over to [Hugging Face Spaces](https://huggingface.co/spaces), Create a New Space called `diagnostic-copilot-api`, specify `Gradio` SDK, and leave it blank. Wait 1 min, and GitHub will auto-sync.

### Part 2: Training the Multimodal AI (Kaggle)

Because this relies on heavy GPU access and requires careful navigation around Kaggle's internal root, use the exact sequence of code below inside your Notebook.

1. Add a **Secret** to your Kaggle Notebook named `HF_TOKEN` and input the write token you generated above. toggle it `ON`.
2. Add a **T4x2 GPU** accelerator to the environment.
3. Run this exact sequence of Python inside a **single code cell**:

```python
import os
import shutil

# 1. ALWAYS start at the absolute global root of the container
os.chdir('/kaggle/working')

# 2. Clean out old attempts safely (forces a fresh sync with GitHub updates)
if os.path.exists('diagnostic-copilot-model'):
    shutil.rmtree('diagnostic-copilot-model')

# 3. Clone fresh repository
!git clone https://github.com/hssling/diagnostic-copilot-model.git

# 4. Install requirements from the newly downloaded folder
!pip install -r /kaggle/working/diagnostic-copilot-model/requirements.txt

# 5. Enter directory and run the fully automated training script!
os.chdir('/kaggle/working/diagnostic-copilot-model')
!python train_multimodal.py
```

### What to expect from Kaggle

The notebook will execute the training run. You will see progress bars from PyTorch indicating loss reduction over `max_steps` bounds. Upon hitting 100%, the automated python script will intercept your Kaggle Secret, package the `med-qwen-vl-adapter` module, and silently upload it directly to your live Hugging Face Account.

Wait about 1 Minute. Connect the React frontend Settings to `hf-space:hssling/diagnostic-copilot-api`. Your bespoke medical multimodal model is now receiving diagnostic scans securely from the web!
