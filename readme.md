### Getting Started
To get started, install the same dependencies:
```bash
conda env create -f environment.yml
```
Note that pytorch must be installed seperately.

### File Structure
* ```engrams_kv_moe.py``` - Implementation of Engrams with KV-caching with MoE and pre-norm residual units. Not optimized for multiple GPUs.
* ```test_engrams.py``` - Basic set of tests to ensure that later optimized algorithms still perform the same operations

### Testing
To test, run
```bash
test_engrams.py
```