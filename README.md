# Deep Reinforcement Learning for Multi-Period Facility Location: $p_k$-median Dynamic Location Problem
This is the official code for the published paper 'Deep Reinforcement Learning for Multi-Period Facility Location: $p_k$-median Dynamic Location Problem'.
<img width="1144" alt="image" src="https://github.com/user-attachments/assets/ddfc30c3-358c-4dcc-ab17-41d6e3a6942d">
The paper was accepted by 32nd ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems **(SIGSPATIAL 2024)** as **Full Papar**.
## Dependencies
All experiments are conducted on the machine with two GTX 3090 GPU and Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz, and the code is implemented in Python.

* Python==3.7.0
* pytorch=1.13.1
* SciPy=1.7.3
* tqdm
* tensorboardx

## Quick start
```bash
python run.py --graph_size 20 --pk 2 3 4 --r 0.32 --run_name 'MultiPM20'
python run.py --graph_size 20 --pk 2 3 4 6 8 --r 0.32 --run_name 'MultiPM50'
python run.py --graph_size 20 --pk 2 4 7 9 10 13 15 --r 0.32 --run_name 'MultiPM100'
```

## Usage
### Generating datasets
Generating 1000 MultiPM instances with 20 nodes with $p_k = [2, 3, 4]$
```bash
python gen_data.py --graph_size 20 --pk 2 3 4 --dataset_size 1000
```

### Training
For training the MultiPM instances with 20 nodes with $p_k = [2, 3, 4]$ and using rollout as REINFORCE baseline:
```bash
python run.py --graph_size 20 --pk 2 3 4 --r 0.32 --run_name 'MultiPM20'
```

### Evalution
To evaluate the Multi instances with 20 nodes by greedy strategy:
```bash
python eval.py {dataset_path} --decode_strategy greedy --model {model_path}
```

To evaluate the Multi instances with 20 nodes by sample1280 strategy:
```bash
python eval.py {dataset_path} --decode_strategy sample --width 1280 --model {model_path}
```

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).

#### Warm start
You can initialize a run using a pretrained model by using the `--load_path` option:
```bash
python run.py --graph_size 100 --load_path {pre-trained path}
```

## Solution Visualization
<img width="1596" alt="image" src="https://github.com/user-attachments/assets/f99a5efd-7a3b-4124-9b2c-48282a6e6568">

## Real Application
<img width="1509" alt="image" src="https://github.com/user-attachments/assets/c04a726c-5b71-4236-b778-62c59183df0c">


## Acknowledgements
This repository includes adaptions of the following repositories as baselines:
* https://github.com/wouterkool/attention-learn-to-route/
* https://github.com/kevin031060/CSP_Attention
* https://github.com/CO-RL/SpoNet



