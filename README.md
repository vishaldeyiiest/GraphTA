
# Enhancing Molecular Property Prediction with Auxiliary Learning and Task-Specific Adaptation

## Installation
Follow the [instructions](https://python-poetry.org/docs/#installation) to install poetry. Use `poetry` to install the code and dependencies as follows:
```
git clone https://github.com/vishaldeyiiest/graphta
cd graphta
poetry install
```


## Environments

Manually Installing required packages under conda environment

```
conda create -n GraphTA python=3.
conda activate GraphTA

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
export TORCH=2.0.0
export CUDA=cu118  

pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

## follow Tensorflow with GPU install instructions:https://www.tensorflow.org/install/pip
- Use pip to install

conda install -y -c rdkit rdkit==2022.09.4
pip install ase
pip install git+https://github.com/bp-kelley/descriptastorus
pip install wandb
```


## Baselines

- We provide the implementation of $\mathtt{FT}$ baseline in `src/molecule_finetune.py`.
- For the baseline $\mathtt{GTOT}$, we run the official implementation at [here](https://github.com/youjibiying/GTOT-Tuning).

## Datasets

For dataset download:
### Chem Dataset

```
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset molecule_datasets
```
Move the `molecule_datasets` to `../datasets/`.


## Pretrained Models
- Create a directory `baseline_weights/classification/`
- Download the pre-trained GNN models `supervised` ($\mathtt{SUP}$) and `supervised_contextpred` ($\mathtt{SUP\text{-}C}$) from [here](https://github.com/snap-stanford/pretrain-gnns).


## Experiments

### For running $\mathtt{MTL}$, $\mathtt{GCS}$, $\mathtt{GNS}$ and $\mathtt{BLO}$

```
poetry run python meta_adapt.py --input_model_file <path to pretrained checkpoints> --split scaffold --dataset $dataset \
    --input_data_dir <path to dataset directory> --aux_2D_mode ${aux_mode} \
    --adapt $adapt --adapt_every ${adapt_every} --epochs 100 --batch_size $batch_size --output_model_dir <path to save>
```
Change the following arguments to run different experiments:
- `$adapt`: mtl, gcs, gns, pcgrad, rcgrad, blo, blo+rcgrad for $\mathtt{MTL}$, $\mathtt{GCS}$, $\mathtt{GNS}$, $\mathtt{PCGrad}$, $\mathtt{RCGrad}$, $\mathtt{BLO}$ and $\mathtt{BLO\text{+}RCGrad}$, respectively.
- `$dataset`: Any one of the following -- sider, clintox, bbbp, bace, tox21, toxcast, hiv, muv.
- `${adapt_every}`: one outer optimization for every `adapt_every` inner optimizations in $\mathtt{BLO}$, which is `r` in the paper (Algo 1).
- `${aux_mode}`: list of auxiliary tasks $\subset \{\text{AM,CP,EP,IG,MP}\}$.
- `${input_data_dir}` to the appropiate directory where the data was extracted.

Follow `scripts/run_meta_adapt.sh` for more details.
For example, to run $\mathtt{BLO}$ experiment on `bace` dataset, run:

```
bash scripts/run_meta_adapt.sh bace model_gin/supervised_contextpred.pth ../tmp/ blo
```