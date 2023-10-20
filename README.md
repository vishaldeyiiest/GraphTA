
# Enhancing Molecular Property Prediction with Auxiliary Learning and Task-Specific Adaptation

## Environments

Installing required packages under conda environment

```
conda create -n GraphTA python=3.9
conda activate GraphTA

conda install pytorch-lightning==1.9.4 -c conda-forge
export TORCH=1.13.1
export CUDA=cu116  # cu102, cu110

pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

## follow Tensorflow with GPU install instructions:https://www.tensorflow.org/install/pip
- Use pip to install

conda install -c conda-forge deepchem==2.7.1
pip install deepchem[torch]
pip install deepchem[tensorflow]

conda install -y -c rdkit rdkit==2022.09.4
pip install ase
pip install git+https://github.com/bp-kelley/descriptastorus
pip install ogb
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
- Download the pre-trained GNN models supervised_contextpred ($\mathtt{SUP\text{-}C}$) from [here](https://github.com/snap-stanford/pretrain-gnns), and $\mathtt{GraphMVP\text{-}C}$ from [here](https://github.com/chao1224/GraphMVP/tree/main).


## Experiments

### For running $\mathtt{MTL}$, $\mathtt{GCS}$, $\mathtt{GNS}$ and $\mathtt{BLO}$

```
python meta_adapt.py --input_model_file <path to pretrained checkpoints> --split scaffold --dataset $dataset --aux_2D_mode ${aux_mode} \
    --adapt $adapt --adapt_every ${adapt_every} --epochs 100 --batch_size $batch_size --output_model_dir <path to save>
```
Change the following arguments to run different experiments:
- `$adapt`: mtl, gcs, gns and blo for $\mathtt{MTL}$, $\mathtt{GCS}$, $\mathtt{GNS}$ and $\mathtt{BLO}$, respectively.
- `$dataset`: Any one of the following -- sider, clintox, bbbp, bace, tox21, toxcast, hiv, muv.
- `${adapt_every}`: one outer optimization for every $adapt_every$ inner optimizations in $\mathtt{BLO}$.
- `${aux_mode}`: list of auxiliary tasks $\subset \{AM,CP,EP,IG,MP\}$.

Follow `scripts/run_meta_adapt.sh` for more details.
For example, to run $\mathtt{BLO}$ experiment on `bace` dataset, run:

```
bash scripts/run_meta_adapt.sh bace model_gin/supervised_contextpred.pth ../tmp/ blo
```