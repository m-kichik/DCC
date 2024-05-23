# DCC
Original code and trained model for paper ['Learning Selective Communication for Multi-Agent Path Finding'](https://arxiv.org/abs/2109.05413).

# POGEMA
This repository contains the implementation of the POGEMA \[[github](https://github.com/AIRI-Institute/pogema) | [paper](https://arxiv.org/abs/2206.10944)\] wrapper for the DCC algorithm and corresponding functions.

Find demo by following [link](https://github.com/m-kichik/DCC/blob/main/pogema_demo.ipynb).

To run the provided in [test_set](https://github.com/m-kichik/DCC/tree/main/test_set) tests with POGEMA environment execute command

```bash
python pogema_test.py \
  --weights=saved_models/128000.pth \
  --test_set=test_set/40length_4agents_0.3density.pth \
  --on_target="nothing" \
  --collision_system="soft"
```
