# Child Trauma Assessment AI - Gemma Fine-tuning

A fine-tuned Gemma 3N model specialized for conducting trauma assessments with children from conflict zones, supporting multilingual conversations and generating professional psychological reports.

## Model Availability

The fine-tuned model is available on Hugging Face:
- Model: [SoufianeDahimi/child_trauma_gemma_finetune](https://huggingface.co/SoufianeDahimi/child_trauma_gemma_finetune)
- GGUF format: [SoufianeDahimi/child_trauma_assessment_gemma-GGUF](https://huggingface.co/SoufianeDahimi/child_trauma_assessment_gemma-GGUF)

## Overview

This repository contains the training pipeline for fine-tuning Gemma 3N to:
- Conduct empathetic, culturally-sensitive conversations with parents/caregivers
- Generate standardized trauma assessment reports based on conversations
- Support multiple languages (Arabic dialects, Ukrainian, English)
- Understand cultural contexts from conflict zones (Gaza, Ukraine, Sudan, Syria)

## Quick Start

### 1. Dataset Generation
```bash
pip install -r requirements.txt
python dataset_generation.py
```

This generates realistic trauma assessment conversations and corresponding professional reports.

### 2. Fine-tuning
Run the Jupyter notebook `trauma_assessment_gemma_finetune.ipynb` to fine-tune Gemma 3N on your generated dataset.

## Dataset Structure

The training data consists of two types of examples:

**Conversation Examples**: Multi-turn dialogues between parents and AI
```json
{
  "messages": [
    {"role": "user", "content": "أنا قلقان على ابني..."},
    {"role": "assistant", "content": "أفهم قلقك..."}
  ]
}
```

**Report Generation Examples**: Converting conversations to professional assessments
```json
{
  "messages": [
    {"role": "user", "content": "Based on this conversation, generate a report..."},
    {"role": "assistant", "content": "TRAUMA ASSESSMENT REPORT\n\nPARENT OBSERVATIONS:..."}
  ]
}
```

The generated dataset can be found at `trauma_assessment_training.jsonl`
## Key Features

- **Multilingual Support**: Conversations in Arabic (Palestinian/Syrian/Sudanese dialects), Ukrainian, English
- **Cultural Sensitivity**: Understands region-specific trauma expressions and family dynamics
- **Professional Output**: Generates standardized reports with severity scores and risk indicators
- **Efficient Training**: Uses LoRA adapters for memory-efficient fine-tuning

## Model Capabilities

The fine-tuned model can:
- Ask appropriate follow-up questions during assessments
- Identify trauma indicators across different cultural contexts
- Generate structured reports with severity scoring (1-10)
- Provide culturally-appropriate responses and recommendations
- Handle sensitive conversations with empathy and professionalism

## Files

- `dataset_generation.py` - Generates training conversations and reports
- `models.py` - Pydantic models for structured data
- `trauma_assessment_gemma_finetune.ipynb` - Fine-tuning notebook
- `requirements.txt` - Python dependencies

## Training Configuration

- **Base Model**: Gemma 3N (unsloth/gemma-3n-E2B-it)
- **Method**: LoRA fine-tuning (r=16, alpha=16)
- **Context Length**: 2048 tokens
- **Batch Size**: 1 with gradient accumulation (4 steps)
- **Learning Rate**: 2e-4 with linear scheduler
- **Training Focus**: Conversation responses only (loss masked on user inputs)

## Dataset Format

The training data follows Gemma's chat format with strictly alternating roles:

```json
{
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "أنا قلقان على ابني..."}]},
    {"role": "model", "content": [{"type": "text", "text": "أفهم قلقك..."}]}
  ]
}
```

Note: The model responds in the same language/dialect as the user's input.

## Usage Example

```python
from transformers import TextStreamer
from unsloth import FastModel

# Load model
model, tokenizer = FastModel.from_pretrained(
    "your-model-path",
    max_seq_length = 2048,
    load_in_4bit = True,
)

def generate_response(prompt):
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
        tokenize = True,
        return_dict = True,
    ).to("cuda")
    
    return model.generate(
        **inputs,
        max_new_tokens = 512,
        temperature = 0.7,
        top_p = 0.95,
        top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

# Example in Arabic
response = generate_response("أنا قلقان على ابني عمره 8 سنوات...")

# Example in Ukrainian
response = generate_response("Я хвилююся за свого сина...")
```

## Language Support

The model:
- Automatically detects the input language
- Responds in the same language/dialect as the user
- Supports:
  * Palestinian/Levantine Arabic
  * Sudanese Arabic
  * Ukrainian
  * English
- Generates all assessment reports in professional English regardless of conversation language

## Model Output

The model outputs two types of responses:
1. **Conversational**: Empathetic questions and guidance during assessment
2. **Report Generation**: Structured professional assessments with:
   - Parent observations summary
   - AI analysis of trauma indicators
   - Severity score (1-10)
   - Risk indicators array
   - Cultural context notes

## Integration

This model integrates with the other components of the project (the web platform for psychologist/volunteers, the mobile app and OllamaForge)

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for training
- HuggingFace transformers, unsloth, datasets