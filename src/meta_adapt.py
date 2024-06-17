import time
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T

from config import args
from dataloader import DataLoaderHybrid
from models import GNN, GNN_graphpred, AuxiliaryNet
from splitters import random_scaffold_split, random_split, scaffold_split
from util import (
    do_GraphCL,
    do_GraphCLv2,
    map_param_to_block,
    get_num_task,
    seed_everything,
    EarlyStopping,
)

from datasets import MoleculeGraphCLHybridDataset, MoleculeHybridDataset
from util import setup_transform_criteria, setup_2D_models, ScaleGradientNet
from pretrain_hybrid import compute_loss
from molecule_finetune import eval
from auxilearn.optim import MetaOptimizer
from auxilearn.hypernet import (
    Identity,
    MonoHyperNet,
    MonoLinearHyperNet,
    MonoNonlinearHyperNet,
)


weights_transform = nn.Softmax(dim=0)
step = 0


log_metrics = [
    "ROC",
    "Acc",
    "Loss",
    "Weights",
    "Gradient Similarity",
    "Gradient Norm",
    "Aux Scale Balance",
    "Scaled Weights",
]


def save_model(outdir, save_best, epoch=None):
    if save_best:
        output_model_path = outdir + "/model_best.pth"
    elif epoch is None:
        output_model_path = outdir + "/model_final.pth"
    else:
        output_model_path = outdir + "/model_" + str(epoch) + ".pth"

    saved_model_dict = {
        "model": model.state_dict(),
        "support_model": aux_2D_support_model_list.state_dict(),  # type: ignore
        "weights": weights,
    }
    torch.save(saved_model_dict, output_model_path)


def get_grads(model, target_loss, aux_2D_loss_dict):
    target_grad = torch.autograd.grad(
        target_loss, model.molecule_model.parameters(), create_graph=True
    )
    aux_2D_grad_list = []
    for mode in args.aux_2D_mode:
        aux_2D_grad_list.append(
            torch.autograd.grad(
                aux_2D_loss_dict[mode],
                model.molecule_model.parameters(),
                create_graph=True,
            )
        )
    grad_shapes = [i.shape for i in target_grad]
    return target_grad, aux_2D_grad_list, grad_shapes


def flatten_grads(grads):
    return torch.cat([i.flatten() for i in grads])


def unflatten_grads(grads, shapes):
    unflattened_grads = []
    start = 0
    for shape in shapes:
        end = start + np.prod(shape)
        unflattened_grads.append(grads[start:end].reshape(shape))
        start = end
    return unflattened_grads


def project_conflicting_grads(
    target_grad_flat,
    aux_2D_grad_flat,
    grad_sim,
    grad_norm,
    weights=None,
    reduction="sum",
):
    """
    function to project the conflicting gradients and either scale or remove them
    Return: the final auxiliary gradient after surgery
    """
    projected_aux_grads = []
    ## PCGrad operation
    for i, aux_grad in enumerate(aux_2D_grad_flat):
        gsim = grad_sim[i]
        aux_grad_ = aux_grad.clone()
        if gsim < 0:
            aux_grad_ -= gsim * grad_norm[i] * target_grad_flat / grad_norm[-1]

        projected_aux_grads.append(aux_grad_)

    projected_aux_grads = torch.stack(projected_aux_grads)
    # equation 4 in the paper to scale the gradients
    if "rcgrad" in args.adapt:
        norm_w = torch.clamp(grad_norm[-1] / grad_norm[:-1], max=1)
        projected_aux_grads = (norm_w + weights).unsqueeze(-1) / 2 * projected_aux_grads

    if reduction == "sum":
        projected_aux_grads = torch.sum(projected_aux_grads, dim=0)
    elif reduction == "mean":
        projected_aux_grads = torch.mean(projected_aux_grads, dim=0)
    return projected_aux_grads


def train_GS(
    args,
    device,
    model,
    aux_2D_support_model_list,
    criterion,
    train_loader,
    optimizer,
    logger,
    rotation_weights,
    **kwargs,
):
    """
    Adaptation modes: MTL, GCS, GNS, PCGrad, RCGrad
    """
    start_time = time.time()
    for mode, support_model in aux_2D_support_model_list.items():
        if mode != "EP":
            support_model.train()
    model.train()

    aux_2D_loss_accum = 0
    target_loss_accum, total_loss_accum = 0, 0
    aux_2D_loss_accum_dict, aux_2D_acc_accum_dict = dict.fromkeys(
        args.aux_2D_mode, 0
    ), dict.fromkeys(args.aux_2D_mode, 0)

    weights = torch.ones(len(args.aux_2D_mode)).to(device)
    weights_accum = []

    if set(args.aux_2D_mode).intersection(set(["GraphCL", "JOAO", "JOAOv2"])):
        aug_prob_cp = train_loader.dataset.aug_prob
        n_aug = np.random.choice(25, 1, p=aug_prob_cp)[0]
        n_aug1, n_aug2 = n_aug // 5, n_aug % 5

    molecule_readout_func = model.pool

    # =======================================================================================#
    def aggregate_loss(target_loss, aux_2D_loss_dict):
        aux_loss = torch.stack(list(aux_2D_loss_dict.values()))
        if args.adapt == "rcgrad":
            aux_loss = rotation_net(aux_loss)

        aux_2D_loss = torch.sum(aux_loss)
        total_loss = target_loss + aux_2D_loss
        return total_loss, aux_2D_loss

    # =======================================================================================#

    for batch in train_loader:
        if isinstance(batch, list):
            batch1 = batch[1].to(device)
            batch2 = batch[2].to(device)
            batch = batch[0].to(device)
        else:
            batch = batch.to(device)

        target_loss, aux_2D_loss_dict, aux_2D_acc_dict = compute_loss(
            args, batch, model, aux_2D_support_model_list, criterion
        )
        if set(args.aux_2D_mode).intersection(set(["GraphCL", "JOAO"])):
            mode = "GraphCL" if "GraphCL" in args.aux_2D_mode else "JOAO"
            aux_2D_loss_dict[mode] = do_GraphCL(
                batch1=batch1,
                batch2=batch2,
                molecule_model_2D=model.molecule_model,
                projection_head=aux_2D_support_model_list[mode],
                molecule_readout_func=molecule_readout_func,
            )
        elif "JOAOv2" in args.aux_2D_mode:
            aux_2D_loss_dict["JOAOv2"] = do_GraphCLv2(
                batch1=batch1,
                batch2=batch2,
                n_aug1=n_aug1,
                n_aug2=n_aug2,
                molecule_model_2D=model.molecule_model,
                projection_head=aux_2D_support_model_list["JOAOv2"],
                molecule_readout_func=molecule_readout_func,
            )

        target_grad, aux_2D_grad_list, grad_shapes = get_grads(
            model, target_loss, aux_2D_loss_dict
        )
        total_loss, aux_2D_loss = aggregate_loss(target_loss, aux_2D_loss_dict)

        # Backpropagate and update all model parameters, we will modify the gradients of shared paramters
        optimizer.zero_grad()
        total_loss.backward()

        # flatten the gradients
        target_grad_flat = flatten_grads(target_grad)
        aux_2D_grad_flat = torch.stack(
            [flatten_grads(aux_2D_grad) for aux_2D_grad in aux_2D_grad_list]
        )

        # compute cosine similarity of gradients for each parameter and then take the mean
        grad_norm = torch.cat(
            (
                torch.norm(aux_2D_grad_flat, dim=1),
                torch.norm(target_grad_flat.unsqueeze(0), dim=1),
            )
        )
        grad_sim = nn.functional.cosine_similarity(target_grad_flat, aux_2D_grad_flat)

        if args.adapt == "pcgrad" or args.adapt == "rcgrad":
            ## g = g_t + g_i - (g_t.g_i)/||g_t||^2 * g_t
            if args.adapt == "rcgrad":
                ## scale the aux gradients by learned weight and then projecting is equivalent to rotation and projecting
                weights.data = rotation_weights.data  # type: ignore

            projected_aux_grads = project_conflicting_grads(
                target_grad_flat,
                aux_2D_grad_flat,
                weights=rotation_weights,
                grad_sim=grad_sim,
                grad_norm=grad_norm,
            )
            # print(rotation_net.scale_weights)
            unflattened_aux_grads = unflatten_grads(projected_aux_grads, grad_shapes)
            for i, param in enumerate(model.molecule_model.parameters()):
                param.grad = unflattened_aux_grads[i].clone() + target_grad[i].clone()

        else:
            ## g = g_t + w*g_i
            if args.adapt == "gcs":
                ## w = \max(0, cos(g_t, g_i))
                weights = torch.clamp(grad_sim, min=0)

            elif args.adapt == "gns":
                ## w = \min(1, ||g_t||/||g_i||)
                weights = torch.clamp(grad_norm[-1] / grad_norm[:-1], max=1)

            elif args.adapt == "gcs+gns":
                ## w = g_t.g_i * ||g_t||/||g_i||
                weights = torch.clamp(grad_sim * grad_norm[-1] / grad_norm[:-1], min=0)

            aux_grad = torch.stack(
                [
                    aux_2D_grad * weights[i]
                    for i, aux_2D_grad in enumerate(aux_2D_grad_flat)
                ]
            )
            # modify the auxiliary gradients for shared parameters
            aux_grad = torch.sum(aux_grad, dim=0)
            unflattened_aux_grad = unflatten_grads(aux_grad, grad_shapes)
            for i, param in enumerate(model.molecule_model.parameters()):
                param.grad = unflattened_aux_grad[i].clone() + target_grad[i].clone()

        optimizer.step()
        if args.adapt == "rcgrad":
            rotation_weights.data.clamp_(min=0, max=1)  # type: ignore

        for mode in args.aux_2D_mode:
            aux_2D_loss_accum_dict[mode] += aux_2D_loss_dict[mode].detach().cpu().item()
            if mode not in ["GraphCL", "JOAO", "JOAOv2"]:
                aux_2D_acc_accum_dict[mode] += aux_2D_acc_dict[mode]
        aux_2D_loss_accum += aux_2D_loss.detach().cpu().item()
        target_loss_accum += target_loss.detach().cpu().item()
        total_loss_accum += total_loss.detach().cpu().item()
        weights_accum.append(weights.detach().cpu().numpy())

    aux_2D_loss_accum /= len(train_loader)
    target_loss_accum /= len(train_loader)
    total_loss_accum /= len(train_loader)
    weights_accum = np.mean(weights_accum, axis=0)
    weights.data = torch.tensor(weights_accum).to(device)

    for mode in args.aux_2D_mode:
        aux_2D_acc_accum_dict[mode] /= len(train_loader)  # type: ignore
        aux_2D_loss_accum_dict[mode] /= len(train_loader)  # type: ignore

    weights = weights.detach().cpu().numpy()
    grad_sim = grad_sim.detach().cpu().numpy()
    grad_norm = grad_norm.detach().cpu().numpy()
    print_str = (
        f"Weighted Total Loss: {total_loss_accum:.6f}\t"
        + f"Target Loss: {target_loss_accum:.5f}\t"
        + "\t".join(
            [
                f"{mode} Loss: {aux_2D_loss_accum_dict[mode]:.6f}  {mode} Acc: {aux_2D_acc_accum_dict[mode]:.6f}"
                for mode in args.aux_2D_mode
            ]
        )
    )
    logger.info(print_str + f"\tTime: {time.time() - start_time:.6f}")
    logger.info(
        "Weights: "
        + np.array2string(weights, formatter={"float_kind": lambda x: "%.6f" % x})
    )
    logger.info(
        "Gradient Similarity: "
        + np.array2string(grad_sim, formatter={"float_kind": lambda x: "%.6f" % x})
    )
    logger.info(
        "Gradient Norm: "
        + np.array2string(grad_norm, formatter={"float_kind": lambda x: "%.6f" % x})
    )

    dict_to_log = {
        "Weighted Total Loss": total_loss_accum,
        "Weighted SSL Loss": aux_2D_loss_accum,
        "Target Loss": target_loss_accum,
    }
    for i, mode in enumerate(args.aux_2D_mode):
        dict_to_log[f"{mode} Weights"] = weights[i]
        dict_to_log[f"{mode} Gradient Similarity"] = grad_sim[i]
        dict_to_log[f"{mode} Gradient Norm"] = grad_norm[i]
    dict_to_log["Target Weights"] = 1.0
    dict_to_log["Target Norm"] = grad_norm[-1]

    return dict_to_log


def train_BLO(
    args,
    device,
    model,
    aux_2D_support_model_list,
    criterion,
    train_loader,
    optimizer,
    logger,
    valid_loader,
    auxiliary_combine_net,
    meta_optimizer,
    **kwargs,
):
    """
    To train BLO, BLO+RCGrad, BLO+GNS
    """
    start_time = time.time()

    for mode, support_model in aux_2D_support_model_list.items():
        if mode != "EP":
            support_model.train()
    model.train()

    aux_2D_loss_accum = 0
    target_loss_accum, total_loss_accum = 0, 0
    aux_2D_loss_accum_dict, aux_2D_acc_accum_dict = dict.fromkeys(
        args.aux_2D_mode, 0
    ), dict.fromkeys(args.aux_2D_mode, 0)
    aux_scale_balance = torch.ones(len(args.aux_2D_mode) + 1).to(device)
    global step

    if set(args.aux_2D_mode).intersection(set(["GraphCL", "JOAO", "JOAOv2"])):
        aug_prob_cp = train_loader.dataset.aug_prob
        n_aug = np.random.choice(25, 1, p=aug_prob_cp)[0]
        n_aug1, n_aug2 = n_aug // 5, n_aug % 5

    # =======================================================================================#
    def aggregate_loss(target_loss, aux_2D_loss_dict):
        aux_loss = torch.stack(list(aux_2D_loss_dict.values()))  # * args.aux_scale
        stacked_loss = (
            torch.concat((aux_loss, target_loss.reshape(-1))) * aux_scale_balance
        )
        total_loss = auxiliary_combine_net(stacked_loss)
        common_grads = torch.autograd.grad(total_loss, shared_params, create_graph=True)
        # print(aux_loss, target_loss, stacked_loss, total_loss)
        aux_2D_loss = torch.sum(aux_loss)
        return total_loss, aux_2D_loss, common_grads

    # =======================================================================================#

    # =======================================================================================#
    def metastep():
        meta_loss = 0
        for valid_batch in valid_loader:
            if isinstance(valid_batch, list):
                valid_batch = valid_batch[0].to(device)
            else:
                valid_batch = valid_batch.to(device)

            loss, _, _ = compute_loss(
                args, valid_batch, model, aux_2D_support_model_list, criterion
            )
            meta_loss += loss
            # break

        inner_loop_end_train_loss = 0
        for train_batch in train_loader:
            if isinstance(train_batch, list):
                train_batch1 = train_batch[1].to(device)
                train_batch2 = train_batch[2].to(device)
                train_batch = train_batch[0].to(device)
            else:
                train_batch = train_batch.to(device)

            target_loss, aux_2D_loss_dict, _ = compute_loss(
                args, train_batch, model, aux_2D_support_model_list, criterion
            )

            if set(args.aux_2D_mode).intersection(set(["GraphCL", "JOAO"])):
                mode = "GraphCL" if "GraphCL" in args.aux_2D_mode else "JOAO"
                aux_2D_loss_dict[mode] = do_GraphCL(
                    batch1=train_batch1,
                    batch2=train_batch2,
                    molecule_model_2D=model.molecule_model,
                    projection_head=aux_2D_support_model_list[mode],
                    molecule_readout_func=molecule_readout_func,
                )
            elif "JOAOv2" in args.aux_2D_mode:
                aux_2D_loss_dict["JOAOv2"] = do_GraphCLv2(
                    batch1=train_batch1,
                    batch2=train_batch2,
                    n_aug1=n_aug1,
                    n_aug2=n_aug2,
                    molecule_model_2D=model.molecule_model,
                    projection_head=aux_2D_support_model_list["JOAOv2"],
                    molecule_readout_func=molecule_readout_func,
                )

            inner_loop_end_train_loss, _, train_common_grads = aggregate_loss(
                target_loss, aux_2D_loss_dict
            )
            break

        phi = [
            param
            for group in meta_optimizer.meta_optimizer.param_groups
            for param in group["params"]
        ]
        curr_hypergrads = meta_optimizer.step(
            val_loss=meta_loss,
            train_grads=train_common_grads,
            aux_params=phi,
            shared_parameters=shared_params,
            return_grads=True,
        )
        return curr_hypergrads

    # =======================================================================================#

    molecule_readout_func = model.pool

    for batch in train_loader:
        if isinstance(batch, list):
            batch1 = batch[1].to(device)
            batch2 = batch[2].to(device)
            batch = batch[0].to(device)
        else:
            batch = batch.to(device)

        target_loss, aux_2D_loss_dict, aux_2D_acc_dict = compute_loss(
            args, batch, model, aux_2D_support_model_list, criterion
        )

        if set(args.aux_2D_mode).intersection(set(["GraphCL", "JOAO"])):
            mode = "GraphCL" if "GraphCL" in args.aux_2D_mode else "JOAO"
            aux_2D_loss_dict[mode] = do_GraphCL(
                batch1=batch1,
                batch2=batch2,
                molecule_model_2D=model.molecule_model,
                projection_head=aux_2D_support_model_list[mode],
                molecule_readout_func=molecule_readout_func,
            )
        elif "JOAOv2" in args.aux_2D_mode:
            aux_2D_loss_dict["JOAOv2"] = do_GraphCLv2(
                batch1=batch1,
                batch2=batch2,
                n_aug1=n_aug1,
                n_aug2=n_aug2,
                molecule_model_2D=model.molecule_model,
                projection_head=aux_2D_support_model_list["JOAOv2"],
                molecule_readout_func=molecule_readout_func,
            )

        target_grad, aux_2D_grad_list, grad_shapes = get_grads(
            model, target_loss, aux_2D_loss_dict
        )
        # flatten the gradients
        target_grad_flat = flatten_grads(target_grad)
        aux_2D_grad_flat = torch.stack(
            [flatten_grads(aux_2D_grad) for aux_2D_grad in aux_2D_grad_list]
        )
        # compute cosine similarity of gradients for each parameter and then take the mean
        grad_norm = torch.cat(
            (
                torch.norm(aux_2D_grad_flat, dim=1),
                torch.norm(target_grad_flat.unsqueeze(0), dim=1),
            )
        )
        grad_sim = nn.functional.cosine_similarity(target_grad_flat, aux_2D_grad_flat)

        # need to balance gradients across tasks
        if args.adapt == "blo+gns":
            aux_scale_balance = torch.clamp(
                grad_sim * grad_norm[-1] / grad_norm[:-1], max=1, min=0
            )
            # aux_scale_balance = grad_sim

        total_loss, aux_2D_loss, common_grads = aggregate_loss(
            target_loss, aux_2D_loss_dict
        )
        optimizer.zero_grad()
        total_loss.backward()

        ## for rcgrad, need to modify the gradient
        if args.adapt == "blo+rcgrad":
            # scale the gradients by learned weight and then projecting is equivalent to rotation and projection
            w_scale = auxiliary_combine_net.linear.weight[0].type(torch.float32)
            projected_aux_grads = project_conflicting_grads(
                target_grad_flat,
                aux_2D_grad_flat,
                grad_sim=grad_sim,
                grad_norm=grad_norm,
                weights=w_scale[:-1],
            )
            unflattened_aux_grads = unflatten_grads(projected_aux_grads, grad_shapes)
            target_grad = unflatten_grads(target_grad_flat, grad_shapes)
            common_grads = [
                unflattened_aux_grads[i] + target_grad[i]
                for i in range(len(target_grad))
            ]

        # update the shared parameters separately
        for p, g in zip(shared_params, common_grads):
            p.grad = g
        optimizer.step()
        del common_grads
        step += 1

        for mode in args.aux_2D_mode:
            aux_2D_loss_accum_dict[mode] += aux_2D_loss_dict[mode].detach().cpu().item()  # type: ignore
            if mode not in ["GraphCL", "JOAO", "JOAOv2"]:
                aux_2D_acc_accum_dict[mode] += aux_2D_acc_dict[mode]
        aux_2D_loss_accum += aux_2D_loss.detach().cpu().item()
        target_loss_accum += target_loss.detach().cpu().item()
        total_loss_accum += total_loss.detach().cpu().item()

        # for meta loss and gradients
        if step % args.adapt_every == 0 and args.adapt in ["blo", "blo+gns"]:
            curr_hypergrads = metastep()
            if isinstance(auxiliary_combine_net, MonoHyperNet):
                # monotonic network
                auxiliary_combine_net.clamp()

    aux_2D_loss_accum /= len(train_loader)
    total_loss_accum /= len(train_loader)
    target_loss_accum /= len(train_loader)

    for mode in args.aux_2D_mode:
        aux_2D_acc_accum_dict[mode] /= len(train_loader)  # type: ignore
        aux_2D_loss_accum_dict[mode] /= len(train_loader)  # type: ignore

    weights = auxiliary_combine_net.linear.weight[0].detach().cpu().numpy()
    grad_sim = grad_sim.detach().cpu().numpy()
    grad_norm = grad_norm.detach().cpu().numpy()
    aux_scale_balance = aux_scale_balance.detach().cpu().numpy()
    scale_balanced_weights = weights * aux_scale_balance
    print_str = (
        f"Weighted Total Loss: {total_loss_accum:.5f}\t"
        + f"Weighted SSL Loss: {aux_2D_loss_accum:.5f}\t"
        f"Target Loss: {target_loss_accum:.5f}\t"
        + "\t".join(
            [
                f"{mode} Loss: {aux_2D_loss_accum_dict[mode]:.5f}  {mode} Acc: {aux_2D_acc_accum_dict[mode]:.5f}"
                for mode in args.aux_2D_mode
            ]
        )
    )
    logger.info(print_str + f"\tTime: {time.time() - start_time:.5f}")
    logger.info(
        "Weights: "
        + np.array2string(weights, formatter={"float_kind": lambda x: "%.6f" % x})
    )
    logger.info(
        "Gradient Similarity: "
        + np.array2string(grad_sim, formatter={"float_kind": lambda x: "%.6f" % x})
    )
    logger.info(
        "Gradient Norm: "
        + np.array2string(grad_norm, formatter={"float_kind": lambda x: "%.6f" % x})
    )

    # dict to log to WandB
    dict_to_log = {
        "Weighted Total Loss": total_loss_accum,
        "Weighted SSL Loss": aux_2D_loss_accum,
        "Target Loss": target_loss_accum,
    }
    for i, mode in enumerate(args.aux_2D_mode):
        dict_to_log[f"{mode} Weights"] = weights[i]
        dict_to_log[f"{mode} Gradient Similarity"] = grad_sim[i]
        dict_to_log[f"{mode} Gradient Norm"] = grad_norm[i]
        dict_to_log[f"{mode} Norm Scaled Weights"] = scale_balanced_weights[i]
    dict_to_log["Target Weights"] = weights[-1]
    dict_to_log["Target Gradient Norm"] = grad_norm[-1]
    dict_to_log["Target Norm Scaled Weights"] = scale_balanced_weights[-1]

    return dict_to_log


if __name__ == "__main__":
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    seed_everything(args.runseed)

    output_log = "train.log"

    # set up output directory
    outdir = None
    if args.output_model_dir != "":
        (
            os.makedirs(args.output_model_dir)
            if not os.path.exists(args.output_model_dir)
            else None
        )
        # create a new version inside the output_dir with version control incrementing version number
        try:
            version = (
                max(
                    [
                        int(i.split("version_")[-1])
                        for i in os.listdir(args.output_model_dir)
                        if "version_" in i
                    ]
                )
                + 1
            )
        except:
            version = 0
        outdir = os.path.join(args.output_model_dir, "version_" + str(version))
        os.makedirs(outdir)
        output_log = os.path.join(outdir, output_log)

        # save hyperparameters
        json.dump(
            args.__dict__,
            open(os.path.join(outdir, "config.json"), "w"),
            indent=4,
            sort_keys=True,
        )

    logger = logging.getLogger()
    file_handler = logging.FileHandler(output_log)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info("========== Use {} for 2D SSL ==========".format(args.aux_2D_mode))

    transform, criterion = setup_transform_criteria(args)

    data_root = f"{args.input_data_dir}/{args.dataset}"

    compose_transform = T.Compose(transform.values())  # type: ignore

    # Bunch of classification tasks
    num_tasks = get_num_task(args.dataset)
    if set(args.aux_2D_mode).intersection(set(["GraphCL", "JOAO", "JOAOv2"])):
        dataset = MoleculeGraphCLHybridDataset(
            data_root, dataset=args.dataset, transform=compose_transform, mask_ratio=0
        )
        dataset.set_augMode("sample")
        dataset.set_augStrength(0.2)
    else:
        dataset = MoleculeHybridDataset(
            data_root, dataset=args.dataset, transform=compose_transform
        )

    if args.split == "scaffold":
        smiles_list = pd.read_csv(data_root + "/processed/smiles.csv", header=None)[
            0
        ].tolist()  # type: ignore
        train_dataset, valid_dataset, test_dataset = scaffold_split(  # type: ignore
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
        )
        logger.info("split via scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(  # type: ignore
            dataset,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed,
        )
        logger.info("randomly split")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(
            data_root + args.dataset + "/processed/smiles.csv", header=None
        )[
            0
        ].tolist()  # type: ignore
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed,
        )
        logger.info("random scaffold")
    else:
        raise ValueError("Invalid split option.")
    logger.debug(train_dataset[0])  # type: ignore

    l = int(len(train_dataset) * args.portion)
    idx = np.random.choice(len(train_dataset), l, replace=False)

    if args.portion < 1:
        train_dataset = train_dataset[idx]

    train_loader = DataLoaderHybrid(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    if args.adapt in ["blo", "blo+gns", "blo+rcgrad"]:
        l = int(len(train_dataset) * 0.8)
        train_loader = DataLoaderHybrid(
            train_dataset[:l],
            batch_size=args.batch_size,  # type: ignore
            shuffle=True,
            num_workers=args.num_workers,
        )
        aux_loader = DataLoaderHybrid(
            train_dataset[l:],
            batch_size=args.batch_size,  # type: ignore
            shuffle=True,
            num_workers=args.num_workers,
        )

    val_loader = DataLoaderHybrid(
        valid_dataset,
        batch_size=args.batch_size,  # type: ignore
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoaderHybrid(
        test_dataset,
        batch_size=args.batch_size,  # type: ignore
        shuffle=False,
        num_workers=args.num_workers,
    )

    # set up model
    molecule_model_2D = GNN(
        args.num_layer,
        args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
    ).to(device)

    aux_2D_support_model_list = setup_2D_models(args, num_aux_classes=0, device=device)
    aux_2D_support_model_list = nn.ModuleDict(aux_2D_support_model_list)
    rotation_weights = None
    if args.adapt == "rcgrad":
        # rotation_weights = nn.Parameter(torch.ones(len(args.aux_2D_mode), device=device), requires_grad=True)
        # nn.init.uniform_(rotation_weights, 0, 1)
        rotation_net = ScaleGradientNet(len(args.aux_2D_mode), device=device)
        rotation_weights = rotation_net.scale_weights

    # set up target task model
    model = GNN_graphpred(
        args=args, num_tasks=num_tasks, molecule_model=molecule_model_2D
    ).to(device)
    shared_params = [param for name, param in model.molecule_model.named_parameters()]
    shared_params_name = [
        name for name, param in model.molecule_model.named_parameters()
    ]
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file, device)
    model.to(device)
    logger.debug(model)

    # set up optimizer
    # different learning rates for different parts of GNN
    model_param_group = [
        {
            "params": model.molecule_model.parameters(),
            "lr": args.lr * args.gnn_lr_scale,
        },
        {"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale},
    ]
    # params for aux task heads
    for mode in args.aux_2D_mode:
        if mode != "EP":
            params = aux_2D_support_model_list[mode].parameters()
            model_param_group.append({"params": params, "lr": args.lr * args.lr_scale})

    # extra learnable scaling weights
    if args.adapt == "rcgrad":
        model_param_group.append(
            {"params": rotation_weights, "lr": args.adapt_lr * args.lr_scale}
        )

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    N = len(args.aux_2D_mode) + 1 if args.adapt != "gcsMO" else len(args.aux_2D_mode)
    weights = nn.Parameter(torch.ones(N, device=device) / N, requires_grad=True)

    auxnet_mapping = dict(
        identity=Identity,  ## same as MTL with equal weights on all losses
        linear=MonoLinearHyperNet,
        nonlinear=MonoNonlinearHyperNet,
    )

    auxnet_config = dict(input_dim=N, main_task=-1, weight_normalization=False)

    if args.aux_net == "nonlinear":
        auxnet_config["hidden_sizes"] = [int(l) for l in [128, 64]]
        auxnet_config["init_upper"] = 0.2
    else:
        auxnet_config["skip_connection"] = args.aux_skip_connection
        # init value should be uniform for meta-learning based otherwise 1
        auxnet_config["init_value"] = 1 / N if "blo" in args.adapt else 1

    auxiliary_combine_net = auxnet_mapping[args.aux_net](**auxnet_config)
    auxiliary_combine_net = auxiliary_combine_net.to(device)
    param_to_block, num_modules = map_param_to_block(shared_params_name, "module_wise")
    # auxiliary_combine_net = HyperModel(len(args.aux_2D_mode), num_modules, param_to_block)
    meta_params = [{"params": auxiliary_combine_net.parameters(), "lr": args.adapt_lr}]

    meta_optimizer = optim.SGD(
        meta_params, lr=args.adapt_lr, momentum=0.9, weight_decay=args.decay
    )

    if args.adapt in ["blo", "blo+gns", "blo+rcgrad"]:
        meta_optimizer = MetaOptimizer(
            meta_optimizer=meta_optimizer, hpo_lr=1, truncate_iter=3, max_grad_norm=25
        )

    criterion["target"] = nn.BCEWithLogitsLoss(reduction="none")
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_roc, best_val_idx = -1, 0
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    # for GraphCL, JOAO, JOAOv2
    aug_prob = np.ones(25) / 25
    # np.set_printoptions(precision=3, floatmode='fixed') # type: ignore

    if args.adapt in ["blo", "blo+gns", "blo+rcgrad"]:
        train_function = train_BLO
    elif args.adapt in ["gcs", "gns", "gcsMO", "pcgrad", "mtl", "rcgrad"]:
        train_function = train_GS
    else:
        raise ValueError("Invalid adaptation option.")

    eval_metric = roc_auc_score
    aux_loader = (
        val_loader if args.adapt not in ["blo", "blo+gns", "blo+rcgrad"] else aux_loader
    )

    start_epoch = 1
    # start training
    for epoch in range(start_epoch, args.epochs + start_epoch):
        logger.info("epoch: {}".format(epoch))

        if set(args.aux_2D_mode).intersection(set(["JOAO", "JOAOv2"])):
            train_dataset.set_augProb(aug_prob)  # type: ignore
            logger.info("augmentation probability\t", aug_prob)

        dict_to_log = train_function(
            args,
            device,
            model,
            aux_2D_support_model_list,
            criterion,
            train_loader,
            optimizer,
            logger,
            rotation_weights=rotation_weights,  # type: ignore
            valid_loader=aux_loader,
            auxiliary_combine_net=auxiliary_combine_net,
            meta_optimizer=meta_optimizer,
        )

        if args.eval_train:
            train_roc, train_acc, train_target, train_pred, _ = eval(
                model, device, train_loader, eval_metric
            )
        else:
            train_roc = train_acc = 0
        val_roc, val_acc, val_target, val_pred, val_loss = eval(
            model, device, val_loader, eval_metric
        )
        test_roc, test_acc, test_target, test_pred, test_loss = eval(
            model, device, test_loader, eval_metric
        )
        dict_to_log.update({"Val Loss": val_loss, "Test Loss": test_loss})
        dict_to_log.update(
            {"Train ROC": train_roc, "Val ROC": val_roc, "Test ROC": test_roc}
        )
        # wandb.log(dict_to_log)

        train_roc_list.append(train_roc)
        train_acc_list.append(train_acc)
        val_roc_list.append(val_roc)
        val_acc_list.append(val_acc)
        test_roc_list.append(test_roc)
        test_acc_list.append(test_acc)
        logger.info(
            "train: {:.6f}\tval: {:.6f}\ttest: {:.6f}\n".format(
                train_roc, val_roc, test_roc
            )
        )

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_val_idx = epoch - 1
            if outdir is not None:
                save_model(outdir, save_best=True)

        # early stopping
        early_stopping(val_roc)
        if args.dataset in ["hiv", "muv"] and early_stopping.early_stop:
            logger.info("Early stopping")
            break

    if set(args.aux_2D_mode).intersection(set(["JOAO", "JOAOv2"])):
        logger.info("augmentation probability\t", aug_prob)

    logger.info(
        "best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
            train_roc_list[best_val_idx],
            val_roc_list[best_val_idx],
            test_roc_list[best_val_idx],
        )
    )

    if outdir is not None:
        save_model(outdir, save_best=False)
