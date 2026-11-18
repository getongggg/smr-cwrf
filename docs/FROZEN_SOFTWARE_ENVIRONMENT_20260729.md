# Frozen Detector Software Environment

The main `requirements.txt` describes the current audit, analysis, and plotting
environment. It uses Ultralytics 8.4.107 and is suitable for the included unit
tests, cached-table analyses, manuscript audit, and figure regeneration that
does not require detector inference.

The three detector checkpoints used by the paper embed the following training
provenance:

- Ultralytics: 8.3.252
- Model: YOLOv8n
- Image size: 640
- Batch size: 16
- Epoch budget: 40
- Seeds: 0, 1, and 2

The reported SMR-CWRF runtime artifact independently records Python 3.13.5,
PyTorch 2.9.1+cu128, Ultralytics 8.3.252, and CUDA 12.8. Its checkpoint hashes
are:

- seed 0: `6751DA5756682398E647522FA5C6472247B77387731EB1140C37883E86CE8570`
- seed 1: `B87399056F19726A5BEC62825B1C08AB7AEB17A43B4B8332A1567DD75B787952`
- seed 2: `0385F84AFD24653BF4E9A6863FD15399ACC254224105C43A67F923F08045A539`

For a strict end-to-end rerun from the original weights and images, create a
separate environment with Ultralytics 8.3.252 and match the recorded runtime
stack. Do not reinterpret a rerun under 8.4.107 as bitwise reproduction of the
frozen detector outputs. The redistributed cached predictions, route tables,
statistics, and figures remain the authoritative artifacts for checking the
reported paper values.
