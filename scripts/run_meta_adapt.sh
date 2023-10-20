#!/bin/bash

#SBATCH -A PCON0041
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%j.out

conda init bash
. ~/.bashrc

conda activate GraphPT
cd ../src/

#### GIN fine-tuning
split=scaffold
dataset=$1
model=$2
expt_dir=$3
batch_size=256
adapt=$4
model_group="supc"  ## for logging to WANDB
portion=1
skips=(aux_skip_connection)
#aux_mode="IG MP"
aux_mode="AM EP CP IG MP"

for runseed in 0 1 2; do
for skip in ${skips[@]}; do
    export outdir="${expt_dir}/${runseed}/"
    mkdir -p $outdir/${dataset}/
    #echo Start `date` > $outdir/${dataset}.out

    ### for finetuning
    if [[ $adapt == 'ft' ]]; then
    python molecule_finetune.py --input_model_file ../baseline_weights/classification/${model} \
    --split $split --runseed $runseed --dataset $dataset --model_group ${model_group} --portion $portion \
    --epochs 100 --batch_size $batch_size --output_model_dir $outdir/${dataset}/ > $outdir/${dataset}.out

    ### for adaptation
    else
    python meta_adapt.py --input_model_file ../baseline_weights/classification/${model} \
    --split $split --runseed $runseed --dataset $dataset --aux_2D_mode ${aux_mode} \
    --aux_scale 1 --$skip --adapt $adapt --model_group ${model_group} --portion $portion --adapt_every 10 \
    --epochs 100 --batch_size $batch_size --output_model_dir $outdir/${dataset}/ # >> $outdir/${dataset}.out
    fi
    #echo Done `date` >> $outdir/${dataset}.out 
    
done
done