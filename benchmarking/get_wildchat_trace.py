import datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import json, os, argparse
from dataclasses import asdict, dataclass, field
from typing import List, Optional



@dataclass
class TraceEntry:
    prompt: str
    response: str
    prompt_length: int
    response_length: int

@dataclass
class TraceMetadata:
    num_warmup_requests: int
    avg_entries_per_partition: float
    max_prompt_length: int
    min_prompt_length: int
    avg_prompt_length: float
    max_response_length: int
    min_response_length: int
    avg_response_length: float
    max_total_length: int

@dataclass
class Trace:
    entries: List[TraceEntry] = field(default_factory=list)
    metadata: TraceMetadata = field(default_factory=lambda: TraceMetadata(0, 0, 0, 0, 0, 0, 0, 0,0))

def build_trace(
    dataset: datasets.Dataset, model_name: str, num_entries: int, max_length: int, seed: int, apply_chat_template: bool = False
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = (
        dataset["train"]
        .filter(
            lambda x: x["model"] == "gpt-4"
            and x["turn"] == 1
            and x["language"] == "English"
        )
        .shuffle(seed=seed)
        .select(range(num_entries*3))
    )
    pairs = []
    for row in dataset:
        assert len(row["conversation"]) == 2
        assert row["conversation"][0]["role"] == "user"
        assert row["conversation"][1]["role"] == "assistant"
        pairs.append(
            (
                row["conversation"][0]["content"],
                row["conversation"][1]["content"],
            )
        )

    trace = Trace()
    trace_metadata = TraceMetadata(
        num_warmup_requests=0,
        avg_entries_per_partition=0,
        max_prompt_length=0,
        min_prompt_length=float("inf"),
        avg_prompt_length=0,
        max_response_length=0,
        min_response_length=float("inf"),
        avg_response_length=0,
        max_total_length=0,
    )

    for prompt, response in tqdm(pairs, desc="Processing HF trace"):
        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        prompt_length = len(tokenizer(prompt)["input_ids"])
        response_length = len(tokenizer(response)["input_ids"])
        if prompt_length + response_length > max_length:
            continue
        new_entry = TraceEntry(prompt, response, prompt_length, response_length)
        trace.entries.append(new_entry)
        trace_metadata.max_prompt_length = max(trace_metadata.max_prompt_length, prompt_length)
        trace_metadata.min_prompt_length = min(trace_metadata.min_prompt_length, prompt_length)
        trace_metadata.avg_prompt_length += prompt_length
        trace_metadata.max_response_length = max(trace_metadata.max_response_length, response_length)
        trace_metadata.min_response_length = min(trace_metadata.min_response_length, response_length)
        trace_metadata.avg_response_length += response_length
        trace_metadata.max_total_length = max(trace_metadata.max_total_length, prompt_length + response_length)
        if len(trace.entries) == num_entries:
            break
    trace_metadata.avg_prompt_length /= len(trace.entries)
    trace_metadata.avg_response_length /= len(trace.entries)
    trace_metadata.avg_entries_per_partition = len(trace.entries)

    trace.metadata = trace_metadata

    return trace

def save_trace(trace: Trace, output_path: str):
    """
    Save a Trace instance to a JSON file.
    
    Args:
    trace (Trace): The trace to save.
    output_path (str): The path where the JSON file will be saved.
    """
    # Convert the Trace instance to a dictionary
    trace_dict = asdict(trace)
    
    # Save the dictionary as a JSON file
    with open(output_path, 'w') as f:
        json.dump(trace_dict, f, indent=2)
    
    print(f"Trace saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build WildChat trace")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="Model name")
    parser.add_argument("-m", "--max-length", type=int, default=5000, help="Maximum prompt + response length")
    parser.add_argument("-n", "--num_entries", type=int, default=250, help="Number of entries")
    parser.add_argument("-s", "--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("-o", "--output_file", type=str, default="./traces/wildchat.json", help="Output file name")
    args = parser.parse_args()

    # Change directory to that holding this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    dataset = datasets.load_dataset("allenai/WildChat")
    trace = build_trace(dataset, args.model_name, args.num_entries, args.max_length, args.seed, apply_chat_template=False)
    print("Build trace with the following metadata:")
    print(trace.metadata)
    
    # Save prompts list to a json file
    num_above_2048 = 0
    for entry in trace.entries:
        if entry.prompt_length + entry.response_length > 2048:
            num_above_2048 += 1
    print(f"Number of entries above 2048 tokens: {num_above_2048}")
    save_trace(trace, args.output_file)
