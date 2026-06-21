"""
finetune.py — D3 / D4.

QLoRA fine-tune of meta-llama/Llama-3.2-1B on the corpus Q/A task.

Pipeline
--------
1. Load 10 gold Q/A pairs from data/gold_qa.json.
2. For each pair, fetch the grounding chunks from MongoDB by expected_doc_ids.
3. Use Groq (llama-3.1-8b-instant) to generate 6 STYLE-DIVERSE variants per
   gold pair, grounded in the same context. The six styles are fixed and
   different (casual / academic / short / multi-part / adversarial / follow-up)
   to avoid self-distillation collapse from i.i.d. paraphrases.
4. Combine the 10 originals (unmodified) with 60 variants = 70 training triples
   in the {instruction, input, output} format.
5. QLoRA fine-tune: r=16, alpha=32, target q_proj+v_proj, 3 epochs,
   lr=2e-4, batch 4 with gradient accumulation 4.
6. Save adapter to outputs/lora_adapter/.

Usage:
  python -m src.finetune
"""

import json
import logging
import os
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pymongo import MongoClient
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

load_dotenv(".env.local", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
MONGO_URI    = os.getenv("MONGO_URI", "mongodb://admin:changeme@localhost:27017/?authSource=admin")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")

BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GOLD_PATH   = Path("data/gold_qa.json")
OUTPUT_DIR  = Path("outputs/lora_adapter")

VARIANTS_PER_GOLD  = 6
MAX_CONTEXT_CHARS  = 4000
SEED               = 42

random.seed(SEED)


# ── 1. load gold pairs and fetch grounding chunks ─────────────────────────────

def load_gold_pairs() -> list[dict]:
    with open(GOLD_PATH) as f:
        pairs = json.load(f)
    log.info("Loaded %d gold Q/A pairs from %s", len(pairs), GOLD_PATH)
    return pairs


def fetch_context(mongo, doc_ids: list[str]) -> str:
    chunks = list(mongo.chunks.find({"doc_id": {"$in": doc_ids}}, {"_id": 0, "text": 1}))
    if not chunks:
        log.warning("No chunks found for doc_ids=%s", doc_ids)
        return ""
    text = "\n\n".join(c["text"] for c in chunks)
    return text[:MAX_CONTEXT_CHARS]


# ── 2. style-diverse Q/A variant generation ───────────────────────────────────

STYLE_PROMPTS: list[tuple[str, str]] = [
    (
        "casual",
        "Rewrite the question in a casual, conversational tone — the way a curious "
        "non-expert would ask a friend at lunch. Contractions are fine. Keep the "
        "underlying question the same; only the register changes."
    ),
    (
        "academic",
        "Rewrite the question in a formal academic register, the way it would appear "
        "in a peer-reviewed paper or a graduate-level exam. Use precise terminology. "
        "Keep the underlying question the same; only the register changes. "
        "Do NOT invent specific names, acronyms, dataset names, or numbers that do "
        "not appear in the context. Use generic terms like 'the proposed system' or "
        "'the authors' method' if a name is not given."
    ),
    (
        "short",
        "Rewrite the question as tersely as possible, ideally under 10 words. Strip "
        "every unnecessary word but preserve the exact information being requested."
    ),
    (
        "multi_part",
        "Rewrite the question as a TWO-PART question where the FIRST clause is a "
        "true factual setup drawn directly from the context, and the SECOND clause "
        "is the original question restated. Example: 'The paper introduces X for Y. "
        "What specific mechanism does it use to achieve Z?' The answer must still "
        "match the original answer exactly."
    ),
    (
        "adversarial",
        "Rewrite the question so it contains a subtle TRAP by swapping one key term "
        "for a near-miss (e.g. 'single-vector' instead of 'multivector', 'increase' "
        "instead of 'reduce', 'training' instead of 'inference'). The answer should "
        "then be: 'The context does not address this; the paper instead discusses "
        "[one-sentence summary of the real answer].' This trains the model to push "
        "back on wrong-premise questions rather than confabulate."
    ),
    (
        "follow_up",
        "Rewrite the question as a follow-up turn in a conversation, starting with a "
        "phrase like 'And what about...' or 'Building on that,...' or 'You mentioned "
        "earlier — '. The follow-up should still be answerable on its own and admit "
        "the same answer."
    ),
]


def _groq_client():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set; needed for augmentation.")
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def generate_variant(
    client,
    style_name: str,
    style_instruction: str,
    original_question: str,
    original_answer: str,
    context: str,
) -> dict | None:
    system = (
        "You generate Q/A training pairs for a fine-tuning dataset. Output ONLY a "
        "single JSON object with keys 'question' and 'answer'. No prose, no code "
        "fences, no explanation. The answer must be grounded ONLY in the provided "
        "context — do not use outside knowledge. If the rewritten question can no "
        "longer be answered from the context, return the answer verbatim from the "
        "original answer field."
    )
    user = (
        f"STYLE INSTRUCTION ({style_name}):\n{style_instruction}\n\n"
        f"ORIGINAL QUESTION:\n{original_question}\n\n"
        f"ORIGINAL ANSWER (the rewritten Q must admit this same answer):\n{original_answer}\n\n"
        f"CONTEXT (ground the answer in this text):\n{context}\n\n"
        "Return JSON: {\"question\": \"...\", \"answer\": \"...\"}"
    )

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.7,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        data = json.loads(raw)
        if "question" in data and "answer" in data:
            return {"question": data["question"].strip(), "answer": data["answer"].strip()}
        log.warning("[%s] missing keys in response: %s", style_name, raw[:200])
        return None
    except json.JSONDecodeError as e:
        log.warning("[%s] JSON parse failed: %s | raw=%s", style_name, e, raw[:200])
        return None
    except Exception as e:
        log.warning("[%s] Groq call failed: %s", style_name, e)
        return None


def augment_dataset(gold_pairs: list[dict], mongo) -> list[dict]:
    client  = _groq_client()
    triples: list[dict] = []

    for i, gold in enumerate(gold_pairs):
        q       = gold["question"]
        a       = gold["reference_answer"]
        doc_ids = gold["expected_doc_ids"]
        context = fetch_context(mongo, doc_ids)

        triples.append({"instruction": q, "input": context, "output": a})

        for style_name, style_instr in STYLE_PROMPTS:
            variant = generate_variant(client, style_name, style_instr, q, a, context)
            if variant is None:
                continue
            triples.append({
                "instruction": variant["question"],
                "input":       context,
                "output":      variant["answer"],
            })
            time.sleep(0.4)

        log.info("[%d/%d] gold pair processed (%d triples so far)",
                 i + 1, len(gold_pairs), len(triples))

    log.info("Augmentation done: %d total triples (target ~70)", len(triples))
    return triples


# ── 3. prompt template (import this in eval scripts — do not duplicate) ───────

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Context:\n{input}\n\n"
    "### Response:\n{output}"
)

_PROMPT_PREFIX = (
    "### Instruction:\n{instruction}\n\n"
    "### Context:\n{context}\n\n"
    "### Response:\n"
)

_TRAIN_MAX_CONTEXT_CHARS = 1500


def format_example(ex: dict) -> dict:
    return {
        "prompt":   _PROMPT_PREFIX.format(
            instruction=ex["instruction"],
            context=ex["input"][:_TRAIN_MAX_CONTEXT_CHARS],
        ),
        "response": ex["output"],
    }


# ── 4. tokenise ───────────────────────────────────────────────────────────────

def tokenise(dataset: Dataset, tokenizer, max_length: int = 1024) -> Dataset:
    def _tok(batch):
        full_texts = [p + r for p, r in zip(batch["prompt"], batch["response"])]
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        labels = []
        for prompt, input_ids in zip(batch["prompt"], tokenized["input_ids"]):
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            label = [-100] * prompt_len + list(input_ids[prompt_len:])
            label = (label + [-100] * max_length)[:max_length]
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

    return dataset.map(_tok, batched=True, remove_columns=dataset.column_names)


# ── 5. QLoRA fine-tune ────────────────────────────────────────────────────────

def train(triples: list[dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading tokenizer and 4-bit base model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = Dataset.from_list(triples).map(format_example)
    ds = tokenise(ds, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
    )

    collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    log.info("Starting training...")
    trainer.train()

    log.info("Saving adapter to %s", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log.info("Done.")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    cached = Path("outputs/train_triples.json")
    if cached.exists():
        log.info("Loading existing triples from %s (skipping augmentation)", cached)
        with open(cached) as f:
            triples = json.load(f)
    else:
        mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000).d2
        try:
            gold    = load_gold_pairs()
            triples = augment_dataset(gold, mongo)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(cached, "w") as f:
                json.dump(triples, f, indent=2)
            log.info("Saved augmented dataset: %s", cached)
        finally:
            mongo.client.close()

    train(triples)


if __name__ == "__main__":
    main()
