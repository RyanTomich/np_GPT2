# np_GPT2

`np_GPT2` is a non-sampling implementation of GPT-2 using only Numpy for tensor operations.

## Overview

This project aims to replicate the functionality of GPT-2 using Numpy, without relying on sampling mechanisms. It provides a lightweight alternative for understanding and working with GPT-2.

## Running the Comparison Script

To compare the outputs of different GPT-2 implementations, run the `got2_comparison.py` script with Python 3.12. The script will display responses from:

- GPT-2 directly from the [HuggingFace transformer library](https://huggingface.co/transformers/)
- GPT-2 from a debugging fork of the HuggingFace transformer library
- `np_GPT2` using only Numpy

The goal is for all responses to match, demonstrating the correctness of the `np_GPT2` implementation.

## Requirements

- Python 3.12
- Numpy
- [HuggingFace Transformers library](https://huggingface.co/transformers/) (for comparison purposes)
- [HuggingFace Transformers fork](https://github.com/RyanTomich/huggingface_transformers.git) (for debug purposes)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RyanTomich/np_GPT2.git
