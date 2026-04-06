"""Seed script: populate the database with 50 synthetic tasks and realistic feedback.

Usage:
    python scripts/seed.py                   # default: DATABASE_URL from env
    DATABASE_URL=... python scripts/seed.py  # explicit connection
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import uuid

# Ensure backend modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import text
from core.database import async_session, init_db, engine
from models import (
    Annotator,
    AnnotationType,
    Dataset,
    FeedbackItem,
    Task,
    TaskAssignment,
    TaskStatus,
    TrainingRun,
    TrainingStatus,
)

random.seed(42)

# ── Prompt templates ──────────────────────────────────────────────────

PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in simple terms.",
    "Write a Python function that {task}.",
    "Compare and contrast {topic_a} with {topic_b}.",
    "Summarize the key points of {topic}.",
    "What are the pros and cons of {topic}?",
    "Generate a creative story about {topic}.",
    "Debug the following code:\n```python\n{code}\n```",
    "Translate the following to {language}: {text}",
    "Provide step-by-step instructions for {task}.",
    "What would happen if {scenario}?",
]

TOPICS = [
    "machine learning", "quantum computing", "climate change", "blockchain",
    "neural networks", "reinforcement learning", "natural language processing",
    "computer vision", "distributed systems", "cybersecurity", "gene editing",
    "autonomous vehicles", "microservices", "containerization", "edge computing",
    "federated learning", "transformer architecture", "attention mechanism",
    "gradient descent", "backpropagation", "transfer learning", "data augmentation",
    "model compression", "knowledge distillation", "prompt engineering",
]

TASKS = [
    "sorts a list using quicksort", "implements binary search",
    "calculates the Fibonacci sequence", "validates email addresses",
    "parses JSON from a string", "implements a simple LRU cache",
    "finds the shortest path in a graph", "merges two sorted arrays",
    "computes the edit distance between two strings",
    "implements a basic tokenizer",
]

MODEL_NAMES = ["gpt-4", "claude-3", "llama-70b", "gemini-pro", "mixtral-8x7b"]

ANNOTATOR_NAMES = [
    ("Alice Chen", "alice@example.com"),
    ("Bob Martinez", "bob@example.com"),
    ("Carol Wang", "carol@example.com"),
    ("David Kim", "david@example.com"),
    ("Eve Johnson", "eve@example.com"),
    ("Frank Brown", "frank@example.com"),
    ("Grace Lee", "grace@example.com"),
    ("Henry Taylor", "henry@example.com"),
]

TAGS = [
    ["coding", "python"], ["explanation", "beginner"], ["comparison", "analysis"],
    ["creative", "writing"], ["debugging", "python"], ["translation"],
    ["tutorial", "step-by-step"], ["reasoning", "logic"],
    ["math", "algorithms"], ["nlp", "text"],
]

EVALUATION_CRITERIA = [
    ["accuracy", "clarity", "completeness"],
    ["helpfulness", "harmlessness", "honesty"],
    ["relevance", "coherence", "fluency"],
    ["correctness", "efficiency", "readability"],
]


def _uid() -> str:
    return str(uuid.uuid4())


def _generate_prompt(idx: int) -> str:
    template = PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)]
    return template.format(
        topic=random.choice(TOPICS),
        topic_a=random.choice(TOPICS),
        topic_b=random.choice(TOPICS),
        task=random.choice(TASKS),
        code="def fib(n): return fib(n-1) + fib(n-2)",
        language=random.choice(["Spanish", "French", "Japanese", "German"]),
        text="Hello, how are you today?",
        scenario=f"{random.choice(TOPICS)} became mainstream overnight",
    )


def _generate_responses(n: int = 2) -> list[dict]:
    models = random.sample(MODEL_NAMES, min(n, len(MODEL_NAMES)))
    return [
        {
            "model_id": model,
            "text": f"This is a synthetic response from {model}. "
            f"It demonstrates a {random.choice(['detailed', 'concise', 'thorough', 'creative'])} "
            f"approach to answering the question with "
            f"{random.choice(['examples', 'analogies', 'step-by-step reasoning', 'code snippets'])}.",
        }
        for model in models
    ]


async def seed():
    await init_db()

    async with async_session() as db:
        # Check if data already exists
        result = await db.execute(text("SELECT count(*) FROM tasks"))
        count = result.scalar()
        if count and count > 0:
            print(f"Database already has {count} tasks. Skipping seed.")
            return

        # ── Create annotators ─────────────────────────────────────────
        annotators = []
        for name, email in ANNOTATOR_NAMES:
            a = Annotator(
                id=_uid(),
                email=email,
                name=name,
                role=random.choice(["annotator", "annotator", "annotator", "senior_annotator"]),
                expertise_tags=random.choice(TAGS),
                reliability_score=round(random.uniform(0.7, 1.0), 2),
            )
            db.add(a)
            annotators.append(a)
        await db.flush()
        print(f"Created {len(annotators)} annotators.")

        # ── Create 50 tasks ───────────────────────────────────────────
        tasks = []
        for i in range(50):
            n_responses = random.choice([2, 2, 2, 3, 4])
            annotation_type = random.choice(list(AnnotationType))
            status = random.choices(
                [TaskStatus.PENDING, TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS, TaskStatus.FLAGGED],
                weights=[0.15, 0.55, 0.2, 0.1],
            )[0]

            task = Task(
                id=_uid(),
                prompt=_generate_prompt(i),
                responses=_generate_responses(n_responses),
                annotation_type=annotation_type,
                status=status,
                min_annotations=random.choice([3, 3, 3, 5]),
                tags=random.choice(TAGS),
                evaluation_criteria=random.choice(EVALUATION_CRITERIA),
            )
            db.add(task)
            tasks.append(task)
        await db.flush()
        print(f"Created {len(tasks)} tasks.")

        # ── Create feedback ───────────────────────────────────────────
        feedback_count = 0
        for task in tasks:
            if task.status in (TaskStatus.PENDING,):
                n_feedback = random.randint(0, 1)
            elif task.status == TaskStatus.IN_PROGRESS:
                n_feedback = random.randint(1, task.min_annotations - 1)
            else:
                n_feedback = random.randint(task.min_annotations, task.min_annotations + 2)

            task_annotators = random.sample(annotators, min(n_feedback, len(annotators)))

            for annotator in task_annotators:
                n_resp = len(task.responses) if task.responses else 2

                fb = FeedbackItem(
                    id=_uid(),
                    task_id=task.id,
                    annotator_id=annotator.id,
                    confidence=round(random.uniform(0.5, 1.0), 2),
                    flagged=random.random() < 0.05,
                )

                if task.annotation_type == AnnotationType.RANKING:
                    fb.ranking = random.sample(range(1, n_resp + 1), n_resp)
                elif task.annotation_type == AnnotationType.SCALAR:
                    fb.scalar_reward = round(random.uniform(0.0, 1.0), 4)
                elif task.annotation_type == AnnotationType.BINARY:
                    fb.binary_label = random.choice([True, False])
                elif task.annotation_type == AnnotationType.CRITIQUE:
                    fb.critique_text = random.choice([
                        "Response A is more detailed and accurate.",
                        "Response B provides better examples.",
                        "Both responses are adequate but A has fewer errors.",
                        "Neither response fully addresses the question.",
                        "Response A is clearly superior in clarity and depth.",
                    ])
                    fb.scalar_reward = round(random.uniform(0.3, 0.9), 4)

                db.add(fb)
                feedback_count += 1

                # Create assignment
                assignment = TaskAssignment(
                    id=_uid(),
                    task_id=task.id,
                    annotator_id=annotator.id,
                    time_spent_sec=random.randint(30, 600),
                )
                db.add(assignment)

        await db.flush()
        print(f"Created {feedback_count} feedback items.")

        # ── Compute quality scores for completed tasks ────────────────
        from itertools import combinations
        import math

        for task in tasks:
            result = await db.execute(
                FeedbackItem.__table__.select().where(
                    FeedbackItem.task_id == task.id,
                    FeedbackItem.flagged == False,  # noqa: E712
                )
            )
            all_fb_rows = result.fetchall()

            if not all_fb_rows:
                continue

            # Compute IAA from rankings
            rankings = [row.ranking for row in all_fb_rows if row.ranking]
            if len(rankings) >= 2:
                n_raters = len(rankings)
                n_items = len(rankings[0])
                if n_items >= 2:
                    pairs = list(combinations(range(n_items), 2))
                    total_agreement = 0
                    total_pairs = len(pairs) * n_raters * (n_raters - 1) / 2
                    for ii, jj in pairs:
                        prefer_i = sum(1 for r in rankings if r[ii] < r[jj])
                        prefer_j = n_raters - prefer_i
                        total_agreement += prefer_i * (prefer_i - 1) / 2 + prefer_j * (prefer_j - 1) / 2
                    if total_pairs > 0:
                        p_o = total_agreement / total_pairs
                        kappa = (p_o - 0.5) / 0.5
                        task.iaa = round(max(-1.0, min(1.0, kappa)), 4)

            # Consensus reward
            rewards = []
            for row in all_fb_rows:
                if row.scalar_reward is not None:
                    rewards.append(row.scalar_reward)
                elif row.binary_label is not None:
                    rewards.append(1.0 if row.binary_label else 0.0)
            if rewards:
                task.consensus_reward = round(sum(rewards) / len(rewards), 4)

            # Quality score
            iaa_val = max(0.0, task.iaa) if task.iaa is not None else 0.0
            stdev = 0.0
            if len(rewards) >= 2:
                mean = sum(rewards) / len(rewards)
                variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
                stdev = math.sqrt(variance)
            non_flagged = [r for r in all_fb_rows if not r.flagged]
            coverage = min(1.0, len(non_flagged) / max(1, task.min_annotations))
            task.quality_score = round(0.4 * iaa_val + 0.4 * (1.0 - min(1.0, stdev)) + 0.2 * coverage, 4)

        await db.flush()
        print("Quality scores computed.")

        # ── Create sample datasets ────────────────────────────────────
        for name, fmt, filters in [
            ("DPO Training Set v1", "jsonl", {"min_quality": 0.5}),
            ("High Agreement Set", "parquet", {"min_quality": 0.6, "min_iaa": 0.3}),
            ("Full Export", "huggingface", None),
        ]:
            ds = Dataset(
                id=_uid(),
                name=name,
                filters=filters,
                export_format=fmt,
                task_count=0,
            )
            db.add(ds)
        await db.flush()
        print("Created 3 sample datasets.")

        # ── Create sample training run ────────────────────────────────
        datasets_result = await db.execute(Dataset.__table__.select().limit(1))
        first_ds = datasets_result.fetchone()
        if first_ds:
            tr = TrainingRun(
                id=_uid(),
                dataset_id=first_ds.id,
                algorithm="DPO",
                config={"learning_rate": 1e-5, "beta": 0.1, "epochs": 3},
                reward_history=[round(random.uniform(0.3, 0.8) + i * 0.02, 4) for i in range(20)],
                kl_history=[round(random.uniform(0.01, 0.05) + i * 0.001, 4) for i in range(20)],
                loss_history=[round(0.8 - i * 0.03 + random.uniform(-0.02, 0.02), 4) for i in range(20)],
                status=TrainingStatus.COMPLETED,
            )
            db.add(tr)
            await db.flush()
            print("Created 1 sample training run.")

        # ── Update annotator stats ────────────────────────────────────
        for annotator in annotators:
            fb_result = await db.execute(
                FeedbackItem.__table__.select().where(
                    FeedbackItem.annotator_id == annotator.id
                )
            )
            fb_rows = fb_result.fetchall()
            if fb_rows:
                non_flagged = [r for r in fb_rows if not r.flagged]
                annotator.reliability_score = round(len(non_flagged) / max(1, len(fb_rows)), 2)

        await db.commit()
        print("\nSeed complete!")
        print(f"  - {len(annotators)} annotators")
        print(f"  - {len(tasks)} tasks")
        print(f"  - {feedback_count} feedback items")
        print(f"  - 3 datasets")
        print(f"  - 1 training run")


if __name__ == "__main__":
    asyncio.run(seed())
