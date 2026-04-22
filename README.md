# Singhmaar Corp AI — Fine-Tuning Gemma 4 E2B with QLoRA

A weekend experiment in domain-specific LLM fine-tuning.

I fine-tuned Google's Gemma 4 E2B model to act as an internal AI assistant for a fictional Indian conglomerate — A. Singhmaar Corp. The goal was to see how well a small, carefully written dataset could teach a general-purpose model to understand a specific company's tone, structure, and domain knowledge.

**Training time: 3.4 minutes. Dataset size: 100 examples. Cost: ₹0.**

---

## What Is This

A. Singhmaar Corp is a fictional mid-sized Indian conglomerate with five divisions:

- **SinghmaarTech** — B2B SaaS and cloud infrastructure
- **SinghmaarProperties** — commercial real estate and smart buildings
- **SinghmaarCapital** — venture investing and startup advisory
- **SinghmaarHealth** — hospital chain, diagnostics, and healthtech
- **SinghmaarEdu** — edtech platform and skills training

The fine-tuned model handles employee queries, client-facing Q&A, HR policy questions, legal compliance, investor relations, and operational support across all five divisions.

---

## Results

The base model responded to every company-specific prompt with:
> "I don't have access to internal company policies."

The fine-tuned model, on prompts it had never seen during training, responded with structured, division-aware answers in the right tone and context.

It did not just memorize the training examples. It generalised to unseen queries — which was the most interesting part of this experiment.

---

## Model Details

| Field | Value |
|---|---|
| Base model | google/gemma-4-E2B-it (4.36B parameters) |
| Fine-tuning method | QLoRA via Unsloth |
| Quantization | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Trainable parameters | 25.3M (only 0.49% of total) |
| Training time | 3.4 minutes |
| Hardware | Google Colab, Tesla T4 (free tier) |
| Adapter size | 101MB |
| Framework | Unsloth 2026.4.6 + TRL 0.24.0 + Transformers 5.5.0 |

---

## Training Details

| Field | Value |
|---|---|
| Trainer | SFTTrainer |
| Epochs | 3 |
| Batch size | 2 per device |
| Gradient accumulation | 4 steps (effective batch size 8) |
| Learning rate | 2e-4 with cosine scheduler |
| Warmup steps | 5 |
| Max sequence length | 512 tokens |
| Total steps | 36 |

### Loss Curve

| Epoch | Training Loss | Validation Loss |
|---|---|---|
| 1 | 13.43 | 5.73 |
| 2 | 6.42 | 5.07 |
| 3 | 4.64 | 4.93 |

---

## Dataset

100 hand-crafted examples in Alpaca format covering 10 business domains:

1. HR and employee policy
2. SinghmaarTech product, engineering, SLA, and API queries
3. SinghmaarProperties leasing, facilities, and tenant support
4. SinghmaarCapital startup evaluation, investment, and founder communications
5. SinghmaarHealth appointments, diagnostics, packages, and support
6. SinghmaarEdu courses, certifications, and learner support
7. Executive strategy, board, and leadership summaries
8. Legal and compliance, contracts, DPDP Act, and procurement
9. Finance, accounts, invoices, reimbursements, and budgets
10. Customer support, complaints, and escalations

Each example follows this format:

```json
{
  "instruction": "What is the annual leave policy for employees at A. Singhmaar Corp?",
  "input": "",
  "output": "Employees receive 18 days of paid annual leave per year, accrued monthly from the date of joining. Requests must be submitted via the HR portal at least 48 hours in advance, and up to 10 unused days may be carried forward to the next calendar year."
}
```

---

## Repo Structure

```
singhmaar-gemma4-finetune/
├── Singhmaar_Corp_Gemma4_E2B_FineTune_v3.ipynb   # Training notebook (run on Google Colab)
├── singhmaar_corp_finetune_dataset.json            # 100-example Alpaca-format dataset
├── sample_outputs.md                               # Before vs after comparisons
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file
```

---

## How to Run

### Step 1 — Open the notebook in Google Colab

Upload `Singhmaar_Corp_Gemma4_E2B_FineTune_v3.ipynb` to Google Colab.

Set runtime to **T4 GPU** via Runtime → Change runtime type.

### Step 2 — Add your HuggingFace token

In Cell 1, replace the placeholder with your own HF token:

```python
HF_TOKEN = 'YOUR_HUGGINGFACE_TOKEN_HERE'
```

Get a token at huggingface.co/settings/tokens (Read permission is enough to load the model, Write is needed if you want to push your own adapter).

### Step 3 — Upload the dataset

Upload `singhmaar_corp_finetune_dataset.json` via the Files panel in Colab (left sidebar, folder icon).

### Step 4 — Run all cells top to bottom

Do not skip any cell. Do not re-run individual cells out of order. If the runtime crashes, go to Runtime → Disconnect and delete runtime, then start fresh.

---

## Requirements

All dependencies are installed automatically in Cell 2 of the notebook. For reference:

```
unsloth[colab-new]
trl==0.24.0
datasets==4.3.0
transformers==5.5.0
accelerate
bitsandbytes
sentencepiece
torch
```

---

## What I Learned

The hardest part of this project was not the code — it was writing 100 examples that were varied enough to teach the model a domain rather than just memorise answers.

The tooling (Unsloth especially) has gotten remarkably accessible. If you can write a good dataset, you can fine-tune a model. That feels like an important shift in who can actually build with LLMs.

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — made fine-tuning on a free T4 actually practical
- [Google DeepMind](https://huggingface.co/google/gemma-4-E2B-it) — for releasing Gemma 4 openly
- [Hugging Face](https://huggingface.co) — for the ecosystem that makes all of this composable
