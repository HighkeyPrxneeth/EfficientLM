# EfficientLM

EfficientLM is an experimental large language model project focused on maximizing efficiency and performance on limited hardware. The model was designed and trained with a strong emphasis on optimization, leveraging state-of-the-art techniques to push the boundaries of what is possible on consumer-grade GPUs.

## Project Highlights

- **High Efficiency:**
  - The model was trained for nearly **1000 hours** on a single NVIDIA RTX 3060 12GB GPU, demonstrating the feasibility of large-scale language modeling on affordable hardware.
  - Every aspect of the pipeline, from data loading to model architecture, was optimized for memory and compute efficiency.

- **Cutting-Edge Technologies:**
  - Utilizes advanced transformer architectures and memory-efficient attention mechanisms.
  - Employs mixed-precision training (FP16/AMP) to reduce memory usage and accelerate computation.
  - Implements gradient checkpointing and smart batching to fit larger models and sequences into limited VRAM.
  - Data pipeline leverages fast, compressed formats and on-the-fly tokenization for minimal I/O bottlenecks.

- **Open Dataset Handling:**
  - Supports large, open datasets with efficient streaming and sharding, enabling scalable training even on a single GPU.

- **Modular and Extensible:**
  - The codebase is organized for easy experimentation with new architectures, optimizers, and training strategies.

## Project Status

Due to limited compute resources, the full training run could not be completed. However, the project serves as a proof-of-concept for efficient large language model training on consumer hardware, and as a foundation for further research and development.


## Model Architecture (For Research Reference Only)

> **Note:** This model is not intended for general use or to be run on typical consumer systems. It is a research prototype designed to explore the limits of efficiency and optimization in large language models.

### Overview

EfficientLM is built around a highly optimized transformer-based architecture, integrating several advanced techniques to maximize performance and minimize memory usage:

- **Transformer Backbone:**
  - The core of the model is a stack of custom transformer blocks (`TransformerBlockMTLA`), each using a novel attention mechanism called **Multihead Temporal Latent Attention (MTLA)**.
  - MTLA combines content-based and position-based attention, with memory-efficient computation and support for long sequences.

- **Attention Innovations:**
  - **MultiheadTemporalLatentAttention** uses a combination of rotary and non-rotary positional encodings, with learnable scaling and decoupled normalization for improved expressiveness.
  - **HyperNetwork Downsampling:** A hypernetwork generates dynamic weights for downsampling key/value representations, enabling efficient handling of long contexts and reducing memory footprint.
  - **Chunked and Stride-Aware Masking:** Custom masking strategies allow the model to process long sequences in chunks, maintaining causality and efficiency.

- **Feedforward and Adapter Layers:**
  - **SwiGLU Feedforward:** Each block uses a SwiGLU (Swish-Gated Linear Unit) feedforward network for improved non-linearity and parameter efficiency.
  - **Adapters:** Lightweight adapter modules allow for external embeddings and efficient transfer learning.

- **Optimization Techniques:**
  - **Mixed-Precision Training:** The model is trained with FP16/AMP for reduced memory usage and faster computation.
  - **Gradient Checkpointing:** Reduces memory usage by recomputing intermediate activations during backpropagation.
  - **KV Caching:** Supports efficient autoregressive generation with per-block key/value state caching.

- **Other Features:**
  - **Learned Positional Embeddings** for flexible sequence handling.
  - **Modular Design:** All components are implemented as independent, extensible PyTorch modules for easy experimentation.

### Key Modules

- `SwiGLU`: Swish-Gated Linear Unit feedforward block for efficient non-linearity.
- `Adapter`: Bottleneck adapter for external embeddings and transfer learning.
- `HyperNetwork`: Generates dynamic weights for attention downsampling.
- `MultiheadTemporalLatentAttention`: Custom multi-head attention with advanced positional encoding and memory optimizations.
- `TransformerBlockMTLA`: Transformer block integrating MTLA, SwiGLU, and adapters.
- `LMModel`: The full language model, stacking multiple transformer blocks and providing efficient forward and generation methods.

For further details, see the code in `src/model.py`.


## Using External Embedding Models

EfficientLM supports the integration of external embedding models to enhance the quality and expressiveness of its representations. For example, you can use a powerful embedding model such as **Qwen3-0.6B-Embedding** to generate high-quality input embeddings before passing them to EfficientLM's adapter modules.

- **Why Use External Embeddings?**
  - External embedding models are often trained on vast and diverse datasets, capturing rich semantic and contextual information.
  - By leveraging embeddings from models like Qwen3-0.6B-Embedding, EfficientLM can benefit from improved input representations, which can lead to better downstream performance and generalization.

- **How It Works:**
  - The `Adapter` module in EfficientLM is designed to accept external embeddings (e.g., from Qwen3-0.6B-Embedding) and project them into the model's internal hidden space.
  - This allows the main language model to focus on higher-level reasoning and generation, while the external model provides robust, information-rich token or sentence embeddings.

- **Example Workflow:**
  1. Use Qwen3-0.6B-Embedding to encode your input text into dense vectors.
  2. Pass these vectors as `x_external` to EfficientLM's `LMModel`.
  3. The adapter layer will transform these embeddings for use in the transformer stack.

This modular approach enables experimentation with different embedding models and can significantly boost the quality of the language model, especially when compute resources for full end-to-end training are limited.


## Acknowledgements

This project is inspired by the open-source AI community and aims to make large language models more accessible to everyone, regardless of hardware limitations.

---

*Note: Training large models on limited hardware is a significant challenge. EfficientLM demonstrates that with careful engineering and the latest techniques, it is possible to achieve impressive results even with modest resources.*
