import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_SCRIPT = "./data/openwebtext.py"
DATA_DIR = "./data/subsets"
OUTPUT_DIR = "./data/tokenized"
MAX_WORKERS = os.cpu_count() or 4
TOKENIZER_BATCH_SIZE = 1000  
TARGET_TOKENS = {
    'train': 18_000_000_000,
    'val': 2_000_000_000,
    'test': 2_000_000_000
}
INITIAL_SIZE_FACTOR = 1.2


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

def batch_tokenize_documents(docs):
    tokenizer = get_tokenizer()
    eot_token = tokenizer.eot_token
    
    tokenized_docs = tokenizer.encode_batch(docs, allowed_special="all")
    
    for tokens in tokenized_docs:
        tokens.append(eot_token)
        
    return [np.array(tokens, dtype=np.uint16) for tokens in tokenized_docs]

def process_and_write():
    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        DATASET_SCRIPT, 
        data_dir=DATA_DIR, 
        name="plain_text",
        trust_remote_code=True, 
        cache_dir="/media/prxneeth/Elements/Cache/",
        streaming=True,
        split='train'
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = {}
    memmaps = {}
    current_sizes = {}
    indices = {}

    for split in ['train', 'val', 'test']:
        output_file = os.path.join(OUTPUT_DIR, f"{split}.bin")
        initial_size = int(TARGET_TOKENS[split] * INITIAL_SIZE_FACTOR)
        
        files[split] = output_file
        memmaps[split] = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(initial_size,))
        current_sizes[split] = initial_size
        indices[split] = 0

    total_tokens_processed = 0
    current_split = 'train'
    
    dataset_iter = iter(dataset)
    
    with tqdm(total=sum(TARGET_TOKENS.values()), unit="tokens", desc="Overall Progress") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            while True:
                batch_texts = []
                try:
                    for _ in range(TOKENIZER_BATCH_SIZE * MAX_WORKERS):
                        batch_texts.append(next(dataset_iter)['text'])
                except StopIteration:
                    if not batch_texts:
                        break

                future = executor.submit(batch_tokenize_documents, batch_texts)
                
                token_arrays = future.result()
                
                for tokens in token_arrays:
                    if current_split == 'train' and total_tokens_processed >= TARGET_TOKENS['train']:
                        print("\nSwitching to validation set...")
                        current_split = 'val'
                    elif current_split == 'val' and total_tokens_processed >= TARGET_TOKENS['train'] + TARGET_TOKENS['val']:
                        print("\nSwitching to test set...")
                        current_split = 'test'
                    elif current_split == 'test' and total_tokens_processed >= sum(TARGET_TOKENS.values()):
                        break

                    arr = memmaps[current_split]
                    idx = indices[current_split]
                    size = current_sizes[current_split]
                    
                    num_tokens = len(tokens)
                    
                    if idx + num_tokens > size:
                        new_size = int((idx + num_tokens) * 1.2)
                        print(f"\nResizing {current_split}.bin from {size:,} to {new_size:,} tokens...")
                        arr.flush()
                        memmaps[current_split] = np.memmap(files[current_split], dtype=np.uint16, mode='r+', shape=(new_size,))
                        current_sizes[current_split] = new_size
                        arr = memmaps[current_split]

                    arr[idx : idx + num_tokens] = tokens
                    indices[current_split] += num_tokens
                    total_tokens_processed += num_tokens
                    pbar.update(num_tokens)

                if current_split == 'test' and total_tokens_processed >= sum(TARGET_TOKENS.values()):
                    break
                
                del batch_texts, token_arrays
                gc.collect()

    print("\nFinalizing files...")
    for split in ['train', 'val', 'test']:
        if memmaps[split] is not None:
            memmaps[split].flush()
            final_idx = indices[split]
            final_arr = np.memmap(files[split], dtype=np.uint16, mode='r+', shape=(final_idx,))
            final_arr.flush()
            print(f"Successfully created {files[split]} with {final_idx:,} tokens.")

if __name__ == "__main__":
    process_and_write()
    print("All splits processed.")