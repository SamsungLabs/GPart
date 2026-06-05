"""
Utility functions for model evaluation and text processing
Shared between different evaluation scripts
"""

import re

import torch
from tqdm import tqdm


def extract_first_sentence(text):
    """Extract the first sentence from text based on sentence-ending punctuation"""
    # Split on sentence-ending punctuation (., !, ?) followed by whitespace or end of string
    sentences = re.split(r"[.!?]+\s+", text.strip())
    return sentences[0] if sentences else text.strip()


def extract_last_sentence(text):
    """Extract the last sentence from text based on sentence-ending punctuation"""
    # Split on sentence-ending punctuation (., !, ?) followed by whitespace or end of string
    sentences = re.split(r"[.!?]+\s+", text.strip())
    return sentences[-1] if sentences else text.strip()


def format_prompt(
    entity_labels,
    relation_labels,
    entity_ids=None,
    relation_ids=None,
    adjacency=None,
    question=None,
    include_adjacency=True,
):
    """Format a prompt with graph information, question, and op"""
    # Create a more readable representation of the graph with entity IDs
    if entity_ids and len(entity_ids) == len(entity_labels):
        # Every entity has an ID - use the clean format
        entities_str = "\n".join(
            [
                f"  {i}. {label} ({entity_ids[i]})"
                for i, label in enumerate(entity_labels)
            ]
        )
    else:
        # Fallback to basic format (shouldn't happen with current dataset)
        entities_str = "\n".join(
            [f"  {i}. {entity}" for i, entity in enumerate(entity_labels)]
        )

    # Create a more readable representation of the graph with relation IDs
    if relation_ids and len(relation_ids) == len(relation_labels):
        # Every relation has an ID - use the clean format
        relations_str = "\n".join(
            [
                f"  {i}. {label} ({relation_ids[i]})"
                for i, label in enumerate(relation_labels)
            ]
        )
    else:
        # Fallback to basic format (shouldn't happen with current dataset)
        relations_str = "\n".join(
            [f"  {i}. {relation}" for i, relation in enumerate(relation_labels)]
        )

    # Format adjacency matrix if provided and requested
    adjacency_str = ""
    if include_adjacency and adjacency is not None:
        adjacency_str = "\nAdjacency Matrix:\n"
        # Create rows for each entity
        for i, row in enumerate(adjacency):
            adjacency_str += f" ".join([f"{val:2}" for val in row]) + "\n"

    # Adjust the prompt text based on whether adjacency matrix is included
    if include_adjacency:
        graph_description = "The graph contains the following entities, relations, and adjacency matrix."
        adjacency_instruction = "Each row of the adjacency matrix represents the following connection: entity -> relationship -> entity."
    else:
        graph_description = "The graph contains the following entities and relations."
        adjacency_instruction = ""

    prompt = f"""Below is a question related to a knowledge graph.
{graph_description}
{adjacency_instruction}
Answer the question based on the graph information.
Output just the response to the question, without any extra details.

### Entities:
{entities_str}

### Relations:
{relations_str}
{adjacency_str}
### Question:
{question}

### Answer:\n
"""

    return prompt


def format_instruction(sample, include_adjacency=True):
    """Format the Graphext-QA dataset into instruction format"""
    # Extract graph information
    subgraph = sample.get("subgraph", {})
    entity_labels = subgraph.get("entity_labels", [])
    relation_labels = subgraph.get("relation_labels", [])
    entity_ids = subgraph.get("entities", [])
    relation_ids = subgraph.get("relations", [])
    adjacency = subgraph.get("adjacency", [])
    answer = sample["answers"][0]

    return (
        format_prompt(
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            adjacency=adjacency,
            question=sample["question"],
            include_adjacency=include_adjacency,
        ),
        answer,
    )


def format_dataset(sample, include_adjacency=True):
    """Format the Graphext-QA dataset into prompt-completion format for training"""
    prompt, completion = format_instruction(sample, include_adjacency=include_adjacency)
    return {"prompt": prompt, "completion": completion}


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    use_tqdm=False,
    device_map="auto",
    include_adjacency=True,
    use_chat_template=False,
    batch_size=32,
):
    """Evaluate the model on the test dataset

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        test_dataset: The test dataset
        use_tqdm: Whether to show progress bar
        device_map: Device mapping for the model
        include_adjacency: Whether to include adjacency matrix in the prompt
        use_chat_template: Whether to use chat template for formatting
        batch_size: Number of samples to process at once
    """
    print("\n" + "=" * 60)
    print("Evaluating model on test set...")
    print("=" * 60)

    model.eval()
    correct = 0
    total = 0
    results_data = []

    dataset_size = len(test_dataset)
    device = next(model.parameters()).device

    # Left-padding is required for batched generation so all sequences
    # are right-aligned and new tokens are generated at the end.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batch_starts = range(0, dataset_size, batch_size)
    if use_tqdm:
        batch_starts = tqdm(
            batch_starts,
            total=(dataset_size + batch_size - 1) // batch_size,
            unit="batch",
        )

    for batch_start in batch_starts:
        batch_samples = [
            test_dataset[i]
            for i in range(batch_start, min(batch_start + batch_size, dataset_size))
        ]

        prompts = []
        for sample in batch_samples:
            # Format the prompt using the instruction formatter
            raw_prompt, answer = format_instruction(
                sample, include_adjacency=include_adjacency
            )

            if use_chat_template and tokenizer is not None:
                messages = [{"role": "user", "content": raw_prompt}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = raw_prompt

            if "### Answer:" in prompt:
                prompt = prompt.split("### Answer:")[0] + "### Answer:"
            prompts.append(prompt)

        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False,  # Use greedy decoding for evaluation
            )

        # Decode only the newly generated tokens (skip the input)
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for j, (sample, response) in enumerate(zip(batch_samples, responses)):
            generated_answer = response.strip()
            correct_answer = sample["answers"][0].strip()
            first_sentence = extract_first_sentence(generated_answer)
            is_correct = correct_answer.lower() in first_sentence.lower()
            if is_correct:
                correct += 1
            total += 1

            # Store results data - fix the undefined variable 'i'
            global_i = batch_start + j
            results_data.append(
                {
                    "sample_id": global_i,
                    "question": sample["question"],
                    "correct_answer": correct_answer,
                    "generated_answer": generated_answer,
                    "first_sentence": first_sentence,
                    "is_correct": is_correct,
                    "prompt": prompts[
                        j
                    ],  # Use prompts[j] instead of undefined 'prompt'
                }
            )

            # Print first 5 examples for inspection
            if global_i < 5:
                print(f"\nExample {global_i+1}:")
                print(f"Question: {sample['question']}")
                print(f"Correct Answer: {correct_answer}")
                print(f"First Generated Sentence: {first_sentence}")
                print(f"Match: {is_correct}")

    tokenizer.padding_side = original_padding_side

    accuracy = correct / total if total > 0 else 0
    print(f"\nTest Results ({dataset_size} samples):")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    return accuracy, results_data


def evaluate_model_coconut(
    model, tokenizer, test_dataset, mode="implicit_cot", use_tqdm=False, batch_size=32
):
    """Evaluate a COCONUT-trained model on the test dataset.

    Args:
        model: The PEFT model to evaluate
        tokenizer: Tokenizer loaded from the coconut checkpoint (includes <bot>/<eot> tokens)
        test_dataset: The test dataset
        mode: "explicit_cot" (stage 0) or "implicit_cot" (stage 1+)
        use_tqdm: Whether to show a progress bar
        batch_size: Number of samples to process at once

    Prompt format follows the chat template used during COCONUT training.
    For implicit_cot the assistant turn is primed with "<bot> <eot>\\n" so
    generation starts from the state the model was trained to answer from.
    Answer extraction: last sentence for explicit_cot (answer follows CoT),
    first sentence for implicit_cot.
    """
    assert mode in ("explicit_cot", "implicit_cot"), f"Unknown mode: {mode}"
    max_new_tokens = 500 if mode == "explicit_cot" else 150

    print("\n" + "=" * 60)
    print(f"Evaluating COCONUT model ({mode}) on test set...")
    print("=" * 60)

    model.eval()
    correct = 0
    total = 0
    results_data = []  # Add results data collection

    dataset_size = len(test_dataset)
    device = next(model.parameters()).device

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batch_starts = range(0, dataset_size, batch_size)
    if use_tqdm:
        batch_starts = tqdm(
            batch_starts,
            total=(dataset_size + batch_size - 1) // batch_size,
            unit="batch",
        )

    for batch_start in batch_starts:
        batch_samples = [
            test_dataset[i]
            for i in range(batch_start, min(batch_start + batch_size, dataset_size))
        ]

        prompts = []
        for sample in batch_samples:
            subgraph = sample.get("subgraph", {})
            user_content = format_prompt(
                entity_labels=subgraph.get("entity_labels", []),
                relation_labels=subgraph.get("relation_labels", []),
                entity_ids=subgraph.get("entities", []),
                relation_ids=subgraph.get("relations", []),
                adjacency=subgraph.get("adjacency", []),
                question=sample["question"],
            )
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if mode == "implicit_cot":
                prompt += "<bot> <eot>\n"
            prompts.append(prompt)

        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for j, (sample, response) in enumerate(zip(batch_samples, responses)):
            generated_answer = response.strip()
            correct_answer = sample["answers"][0].strip()
            extracted = (
                extract_last_sentence(generated_answer)
                if mode == "explicit_cot"
                else extract_first_sentence(generated_answer)
            )
            is_correct = correct_answer.lower() in extracted.lower()
            if is_correct:
                correct += 1
            total += 1

            # Add results data collection
            global_i = batch_start + j
            results_data.append(
                {
                    "sample_id": global_i,
                    "question": sample["question"],
                    "correct_answer": correct_answer,
                    "generated_answer": generated_answer,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "prompt": prompts[j],
                }
            )

            if global_i < 5:
                print(f"\nExample {global_i + 1}:")
                print(f"Question: {sample['question']}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Extracted Answer: {extracted}")
                print(f"Match: {is_correct}")

    tokenizer.padding_side = original_padding_side

    accuracy = correct / total if total > 0 else 0
    print(f"\nTest Results ({dataset_size} samples):")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    return accuracy, results_data


def evaluate_model_inherent_knowledge(
    model,
    tokenizer,
    test_dataset,
    use_tqdm=False,
    device_map="auto",
    include_adjacency=True,
    batch_size=32,
):
    """Evaluate the model on the test dataset with only questions (no context) to test inherent knowledge

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        test_dataset: The test dataset
        use_tqdm: Whether to show progress bar
        device_map: Device mapping for the model
        include_adjacency: Whether to include adjacency matrix in the prompt (ignored for inherent knowledge test)
        batch_size: Number of samples to process at once
    """
    print("\n" + "=" * 60)
    print("Evaluating model inherent knowledge on test set (questions only)...")
    print("=" * 60)

    model.eval()
    correct = 0
    total = 0
    results_data = []

    dataset_size = len(test_dataset)
    device = next(model.parameters()).device

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batch_starts = range(0, dataset_size, batch_size)
    if use_tqdm:
        batch_starts = tqdm(
            batch_starts,
            total=(dataset_size + batch_size - 1) // batch_size,
            unit="batch",
        )

    for batch_start in batch_starts:
        batch_samples = [
            test_dataset[i]
            for i in range(batch_start, min(batch_start + batch_size, dataset_size))
        ]

        prompts = [
            f"""Below is a question. Answer the question to the best of your ability.
Output just the response to the question, without any extra details.

### Question:
{sample["question"]}

### Answer:""" for sample in batch_samples
        ]

        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False,  # Use greedy decoding for evaluation
            )

        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for j, (sample, response) in enumerate(zip(batch_samples, responses)):
            generated_answer = response.strip()
            correct_answer = sample["answers"][0].strip()
            first_sentence = extract_first_sentence(generated_answer)
            is_correct = correct_answer.lower() in first_sentence.lower()
            if is_correct:
                correct += 1
            total += 1

            # Store results data
            global_i = batch_start + j
            results_data.append(
                {
                    "sample_id": global_i,
                    "question": sample["question"],
                    "correct_answer": correct_answer,
                    "generated_answer": generated_answer,
                    "first_sentence": first_sentence,
                    "is_correct": is_correct,
                    "prompt": prompts[j],
                }
            )

            # Print first 5 examples for inspection
            if global_i < 5:
                print(f"\nExample {global_i+1}:")
                print(f"Question: {sample['question']}")
                print(f"Correct Answer: {correct_answer}")
                print(f"First Generated Sentence: {first_sentence}")
                print(f"Match: {is_correct}")

    tokenizer.padding_side = original_padding_side

    accuracy = correct / total if total > 0 else 0
    print(f"\nTest Results ({dataset_size} samples):")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    return accuracy, results_data
