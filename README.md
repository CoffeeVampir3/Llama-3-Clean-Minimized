Paths are hardcoded - to run yourself clone <https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct> to the repo directory.

`hf-control.py` prints debugging info and runs the huggingface control with the same prompt and settings. The outcomes should be identicle, or there's an issue.
On my 3090 TI the hf control takes around (Uncompiled because compiled ver never completes): `Execution time: 1.92 seconds` `Execution time: 1.81 seconds` `Execution time: 2.29 seconds`

`verify-modeling.py` prints debugging info and runs the modeling files in this repo.
Same hardware, modeling took: 
`Compiling took: 0.50 seconds -- Execution time: 0.57 seconds` x2 `Compiling took: 0.50 seconds -- Execution time: 0.58 seconds`

There's about a 100% speedup if you count the compile time and a 200% speedup in terms of expected speedup.

### Test Training:

Use `test-train.py` -- assumes you have a dataset for text completion in jsonl format structured like `"text":"This is one sample of training text..."\n"text":"..."`
`inference.py` can be used to validate the resulting models. Due to weight tying, the models can't be saved as .safetensors right now as safetensors does not support it.

### Changelog:
2/4/25 -- Added test training scripts and validated that training succeeds with 3 epochs on 1GB of data reaching a loss of about ~3. The outputs are coherent and the modeling and training are sound.
2/2/25 -- Added liger rope kernel. This is 20% slower in the singular test (because of JIT) but about 20% faster in only 10 iterations, so a significant speedup.
