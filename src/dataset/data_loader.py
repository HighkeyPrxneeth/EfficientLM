import numpy as np
import torch
import os

class DataLoader:
    def __init__(self, file_path, batch_size, block_size, device='cuda', pin_memory: bool = False):
        """Simple memmap loader with lightweight random slicing.

        Args:
            file_path: path to uint16 .bin memmap file
            batch_size: number of sequences per batch
            block_size: tokens per sequence
            device: 'cuda' or 'cpu'. If 'cuda', tensors are moved to GPU.
            pin_memory: when device is 'cpu', return pinned CPU tensors to enable fast H2D copies.
        """
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.pin_memory = bool(pin_memory)
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')

    def get_batch(self):
        """Return a random batch of token ids (x) and next-token targets (y).

        Returns torch.int64 tensors if moved to CUDA device; otherwise returns torch.uint16 CPU tensors
        to reduce CPU-side copies. Cast to long on GPU for embedding lookup.
        """
        N = len(self.data)
        B = self.batch_size
        T = self.block_size
        # Sample B random start positions in [0, N - T - 1]
        starts = np.random.randint(0, max(1, N - T - 1), size=B, dtype=np.int64)
        # Build [B, T] indices using broadcasting via outer addition
        seq = starts[:, None] + np.arange(T, dtype=np.int64)[None, :]
        x_np = self.data[seq.reshape(-1)].reshape(B, T)  # uint16
        y_np = self.data[(seq + 1).reshape(-1)].reshape(B, T)  # uint16
        # CPU tensors referencing the numpy memory (no extra copy initially)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        if 'cuda' in self.device:
            # Move to GPU as int64 for embedding lookup
            return (
                x.pin_memory().to(self.device, dtype=torch.long, non_blocking=True),
                y.pin_memory().to(self.device, dtype=torch.long, non_blocking=True),
            )
        else:
            # For CPU training, ensure int64 dtype for embedding and loss
            x = x.long()
            y = y.long()
            if self.pin_memory:
                return x.pin_memory(), y.pin_memory()
            return x, y


def diagnose(file_path, chunk_size=1024*1024):
    print(f"--- Running Safe Diagnostics on {file_to_check} ---")
    try:
        data = np.memmap(file_path, dtype=np.uint16, mode='r')
        total_tokens = len(data)
        file_size_gb = data.nbytes / (1024**3)
        
        print(f"File size: {file_size_gb:.2f} GB")
        print(f"Total tokens in file: {total_tokens:,}")

        if total_tokens == 0:
            print("\nResult: The file is empty.")
            return

        print("Scanning for non-zero data in chunks...")
        found_data = False
        for i in range(0, total_tokens, chunk_size):
            chunk = data[i:i+chunk_size]
            if np.any(chunk > 0):
                first_nonzero_in_chunk = np.argmax(chunk > 0)
                first_nonzero_idx = i + first_nonzero_in_chunk
                print(f"\nResult: Found the first non-zero token at index: {first_nonzero_idx}")
                print("This indicates the file is at least partially filled.")
                print("\nSample of first 20 tokens from where data begins:")
                print(data[first_nonzero_idx : first_nonzero_idx + 20])
                found_data = True
                break
        
        if not found_data:
            print("\nResult: The file contains only zeros.")
            print("This means the tokenization script did not write any data.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    file_to_check = 'data/tokenized/train.bin'

    if not os.path.exists(file_to_check) or os.path.getsize(file_to_check) == 0:
        print(f"Error: The file '{file_to_check}' is missing or empty.")
        print("Please run the tokenizer script to generate the data file.")
    else:
        diagnose(file_to_check)

        print("\n--- Attempting to fetch a random batch ---")
        try:
            train_loader = DataLoader(file_to_check, batch_size=4, block_size=1024)
            x, y = train_loader.get_batch()
            print("Input batch shape:", x.shape)
            print("Target batch shape:", y.shape)
            print("\nFirst sequence (input):")
            print(x[0])
            print("\nFirst sequence (target):")
            print(y[0])
        except Exception as e:
            print(f"An error occurred while getting a batch: {e}")