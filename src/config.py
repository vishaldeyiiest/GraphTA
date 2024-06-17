import argparse

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--runseed", type=int, default=0)
parser.add_argument("--device", type=int, default=0)

# about dataset and dataloader
parser.add_argument(
    "--input_data_dir", type=str, default="../../../GraphPT/datasets/molecule_datasets/"
)
parser.add_argument(
    "--dataset", type=str, default="tox21"
)  # GEOM_3D_nmol50000_nconf5_nupper1000')
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--task", type=str, default="alpha")
parser.add_argument("--portion", type=float, default=1.0)

# about training strategies
parser.add_argument("--split", type=str, default="scaffold")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_scale", type=float, default=1)
parser.add_argument("--decay", type=float, default=0)
parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--lr_decay_step_size", type=int, default=100)
parser.add_argument("--lr_decay_patience", type=int, default=50)

# about molecule GNN
parser.add_argument("--gnn_type", type=str, default="gin")
parser.add_argument("--num_layer", type=int, default=5)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--dropout_ratio", type=float, default=0.5)
parser.add_argument("--graph_pooling", type=str, default="mean")
parser.add_argument("--JK", type=str, default="last")
parser.add_argument("--gnn_lr_scale", type=float, default=1)
parser.add_argument("--model_3d", type=str, default="schnet", choices=["schnet"])

# for AttributeMask
parser.add_argument("--mask_rate", type=float, default=0.15)
parser.add_argument("--mask_edge", type=int, default=0)

# for ContextPred
parser.add_argument("--csize", type=int, default=3)
parser.add_argument("--contextpred_neg_samples", type=int, default=1)

# for SchNet
parser.add_argument("--num_filters", type=int, default=128)
parser.add_argument("--num_interactions", type=int, default=6)
parser.add_argument("--num_gaussians", type=int, default=51)
parser.add_argument("--cutoff", type=float, default=10)
parser.add_argument("--readout", type=str, default="mean", choices=["mean", "add"])
parser.add_argument("--schnet_lr_scale", type=float, default=0.1)

# for 2D-3D Contrastive CL
parser.add_argument("--CL_neg_samples", type=int, default=1)
parser.add_argument(
    "--CL_similarity_metric",
    type=str,
    default="EBM_dot_prod",
    choices=["InfoNCE_dot_prod", "EBM_dot_prod"],
)
parser.add_argument("--T", type=float, default=0.1)
parser.add_argument("--normalize", dest="normalize", action="store_true")
parser.add_argument("--no_normalize", dest="normalize", action="store_false")
parser.add_argument("--SSL_masking_ratio", type=float, default=0.15)
# This is for generative SSL.
parser.add_argument("--AE_model", type=str, default="AE", choices=["AE", "VAE"])
parser.set_defaults(AE_model="VAE")

# for 2D-3D AutoEncoder
parser.add_argument("--AE_loss", type=str, default="l2", choices=["l1", "l2", "cosine"])
parser.add_argument("--detach_target", dest="detach_target", action="store_true")
parser.add_argument("--no_detach_target", dest="detach_target", action="store_false")
parser.set_defaults(detach_target=True)

# for 2D-3D Variational AutoEncoder
parser.add_argument("--beta", type=float, default=1)

# for 2D-3D Contrastive CL and AE/VAE
parser.add_argument("--alpha_1", type=float, default=1)
parser.add_argument("--alpha_2", type=float, default=1)

# for 2D auxiliary and 3D-2D
parser.add_argument("--aux_2D_mode", nargs="+", type=str, default="None")
parser.add_argument("--aux_3D_mode", action="store_true")
parser.add_argument("--alpha_3", type=float, default=0.1)
parser.add_argument("--gamma_joao", type=float, default=0.1)
parser.add_argument("--gamma_joaov2", type=float, default=0.1)

# for adaptation
parser.add_argument(
    "--adapt",
    type=str,
    choices=["mtl", "gcs", "blo", "blo+gns", "gns", "pcgrad", "rcgrad", "blo+rcgrad"],
)  ## baselines
parser.add_argument("--adapt_lr", type=float, default=1e-3)
parser.add_argument("--max_adapt_iter", type=int, default=3)
parser.add_argument("--adapt_every", type=int, default=10)
parser.add_argument("--aux_scale", type=float, default=1)
parser.add_argument(
    "--aux_net", type=str, choices=["identity", "linear", "nonlinear"], default="linear"
)
parser.add_argument(
    "--aux_skip_connection", dest="aux_skip_connection", action="store_true"
)
parser.add_argument(
    "--no_aux_skip_connection", dest="aux_skip_connection", action="store_false"
)


# about if we would print out eval metric for training data
parser.add_argument("--eval_train", dest="eval_train", action="store_true")
parser.add_argument("--no_eval_train", dest="eval_train", action="store_false")
parser.set_defaults(eval_train=True)

# about loading and saving
parser.add_argument("--input_model_file", type=str, default="")
parser.add_argument("--model_group", type=str)
parser.add_argument("--output_model_dir", type=str, default="")
parser.add_argument("--resume_training", type=str, default="")

# verbosity
parser.add_argument("--verbose", dest="verbose", action="store_true")
parser.add_argument("--no_verbose", dest="verbose", action="store_false")
# parser.set_defaults(verbose=True)

# debugging
parser.add_argument("--debug", dest="DEBUG", action="store_true")
parser.set_defaults(DEBUG=False)

args = parser.parse_args()
print("arguments\t", args)
