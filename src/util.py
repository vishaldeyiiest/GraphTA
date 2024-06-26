import os
import logging
import random
from math import sqrt

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem
from scipy import stats
from sklearn.metrics import roc_auc_score
from datasets import (
    graph_data_obj_to_nx_simple,
    nx_to_graph_data_obj_simple,
    RDKIT_PROPS,
)
from torch_geometric.nn import global_mean_pool
from models import Discriminator, GNN

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


def seed_everything(runseed):
    random.seed(runseed)
    np.random.seed(runseed)
    torch.manual_seed(runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runseed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(runseed)


def save_model(args, molecule_model_2D, molecule_model_3D, save_best, epoch=None):
    mode = "_".join(args.aux_2D_mode)
    mode += "3D" if args.aux_3D_mode else ""
    if not args.output_model_dir == "":
        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            torch.save(
                molecule_model_2D.state_dict(),
                args.output_model_dir + f"{mode}_model.pth",
            )
            saver_dict = {
                "model": molecule_model_2D.state_dict(),
                "model_3D": (
                    molecule_model_3D.state_dict() if args.aux_3D_mode else None
                ),
            }
            torch.save(saver_dict, args.output_model_dir + f"{mode}_model_complete.pth")

        elif epoch is None:
            torch.save(
                molecule_model_2D.state_dict(),
                args.output_model_dir + f"{mode}_model_final.pth",
            )
            saver_dict = {
                "model": molecule_model_2D.state_dict(),
                "model_3D": (
                    molecule_model_3D.state_dict() if args.aux_3D_mode else None
                ),
            }
            torch.save(
                saver_dict, args.output_model_dir + f"{mode}_model_complete_final.pth"
            )

        else:
            torch.save(
                molecule_model_2D.state_dict(),
                args.output_model_dir + f"{mode}_model_{epoch}.pth",
            )
            saver_dict = {
                "model": molecule_model_2D.state_dict(),
                "model_3D": (
                    molecule_model_3D.state_dict() if args.aux_3D_mode else None
                ),
            }
            torch.save(
                saver_dict, args.output_model_dir + f"{mode}_model_complete_{epoch}.pth"
            )

    return


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if args.CL_similarity_metric == "InfoNCE_dot_prod":
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1.0 / B

    elif args.CL_similarity_metric == "EBM_dot_prod":
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat(
            [Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0
        )
        neg_X = X.repeat((args.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + args.CL_neg_samples * loss_neg

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / (
            len(pred_pos) + len(pred_neg)
        )
        CL_acc = CL_acc.detach().cpu().item()

    else:
        raise Exception

    return CL_loss, CL_acc


def dual_CL(X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2


def do_GraphCL(
    batch1, batch2, molecule_model_2D, projection_head, molecule_readout_func
):
    x1 = molecule_model_2D(batch1.x, batch1.edge_index, batch1.edge_attr)
    x1 = molecule_readout_func(x1, batch1.batch)
    x1 = projection_head(x1)

    x2 = molecule_model_2D(batch2.x, batch2.edge_index, batch2.edge_attr)
    x2 = molecule_readout_func(x2, batch2.batch)
    x2 = projection_head(x2)

    T = 0.1
    batch, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / torch.einsum(
        "i,j->ij", x1_abs, x2_abs
    )
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch), range(batch)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = -torch.log(loss).mean()
    return loss


def do_GraphCLv2(
    batch1,
    batch2,
    n_aug1,
    n_aug2,
    molecule_model_2D,
    projection_head,
    molecule_readout_func,
):
    x1 = molecule_model_2D(batch1.x, batch1.edge_index, batch1.edge_attr)
    x1 = molecule_readout_func(x1, batch1.batch)
    x1 = projection_head[n_aug1](x1)

    x2 = molecule_model_2D(batch2.x, batch2.edge_index, batch2.edge_attr)
    x2 = molecule_readout_func(x2, batch2.batch)
    x2 = projection_head[n_aug2](x2)

    T = 0.1
    batch, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / torch.einsum(
        "i,j->ij", x1_abs, x2_abs
    )
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch), range(batch)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = -torch.log(loss).mean()
    return loss


def do_MotifPred(batch, molecule_repr, criterion, motif_pred_model):
    pred = motif_pred_model(molecule_repr)
    y = batch.motif_label.view(pred.shape).double()
    loss = criterion(pred, y)

    y_true = y.cpu().detach().numpy()
    y_scores = pred.cpu().detach().numpy()
    roc_list = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i] == 1) > 0:
            roc_list.append(roc_auc_score(y_true[i], y_scores[i]))
    roc_auc = sum(roc_list) / len(roc_list)
    return loss, roc_auc


def update_augmentation_probability_JOAO(
    loader,
    molecule_model_2D,
    projection_head,
    molecule_readout_func,
    gamma_joao,
    device,
):
    # joao
    aug_prob = loader.dataset.aug_prob
    loss_aug = np.zeros(25)
    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        loader.dataset.set_augProb(_aug_prob)

        count, count_stop = 0, len(loader.dataset) // (loader.batch_size * 10) + 1
        # for efficiency, we only use around 10% of data to estimate the loss
        with torch.no_grad():
            for _, batch1, batch2 in loader:
                batch1 = batch1.to(device)
                batch2 = batch2.to(device)

                loss = do_GraphCL(
                    batch1=batch1,
                    batch2=batch2,
                    molecule_model_2D=molecule_model_2D,
                    projection_head=projection_head,
                    molecule_readout_func=molecule_readout_func,
                )

                loss_aug[n] += loss.item()
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= count

    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1 / 25))
    mu_min, mu_max = b.min() - 1 / 25, b.max() - 1 / 25
    mu = (mu_min + mu_max) / 2

    while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b - mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b - mu, 0)
    aug_prob /= aug_prob.sum()
    return aug_prob


def update_augmentation_probability_JOAOv2(
    loader,
    molecule_model_2D,
    projection_head,
    molecule_readout_func,
    gamma_joao,
    device,
):
    # joaov2
    aug_prob = loader.dataset.aug_prob
    loss_aug = np.zeros(25)
    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        loader.dataset.set_augProb(_aug_prob)

        count, count_stop = 0, len(loader.dataset) // (loader.batch_size * 10) + 1
        # for efficiency, we only use around 10% of data to estimate the loss
        n_aug1, n_aug2 = n // 5, n % 5
        with torch.no_grad():
            for _, batch1, batch2 in loader:
                batch1 = batch1.to(device)
                batch2 = batch2.to(device)

                loss = do_GraphCLv2(
                    batch1=batch1,
                    batch2=batch2,
                    n_aug1=n_aug1,
                    n_aug2=n_aug2,
                    molecule_model_2D=molecule_model_2D,
                    projection_head=projection_head,
                    molecule_readout_func=molecule_readout_func,
                )

                loss_aug[n] += loss.item()
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= count

    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1 / 25))
    mu_min, mu_max = b.min() - 1 / 25, b.max() - 1 / 25
    mu = (mu_min + mu_max) / 2

    while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b - mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b - mu, 0)
    aug_prob /= aug_prob.sum()
    return aug_prob


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def check_same_molecules(s1, s2):
    mol1 = AllChem.MolFromSmiles(s1)  # type: ignore
    mol2 = AllChem.MolFromSmiles(s2)  # type: ignore
    return AllChem.MolToInchi(mol1) == AllChem.MolToInchi(mol2)


class NegativeEdge:

    def __init__(self):
        """Randomly sample negative edges"""
        pass

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        edge_set = set(
            [
                str(data.edge_index[0, i].cpu().item())
                + ","
                + str(data.edge_index[1, i].cpu().item())
                for i in range(data.edge_index.shape[1])
            ]
        )

        redundant_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5 * num_edges):
            node1 = redundant_sample[0, i].cpu().item()
            node2 = redundant_sample[1, i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if (
                edge_str not in sampled_edge_set
                and edge_str not in edge_set
                and not node1 == node2
            ):
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges / 2:
                break

        data.negative_edge_index = redundant_sample[:, sampled_ind]

        return data


class ExtractSubstructureContextPair:

    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the root node."""
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        """
        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx"""
        num_atoms = data.x.size()[0]
        if root_idx is None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(
            G, root_idx, self.k
        ).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor(
                [substruct_node_map[root_idx]]
            )  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(
            G, root_idx, self.l1
        ).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(
            G, root_idx, self.l2
        ).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(
            set(context_node_idxes).intersection(set(substruct_node_idxes))
        )
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [
                context_node_map[old_idx] for old_idx in context_substruct_overlap_idxes
            ]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = torch.tensor(
                context_substruct_overlap_idxes_reorder
            )

        return data

    def __repr__(self):
        return "{}(k={},l1={}, l2={})".format(
            self.__class__.__name__, self.k, self.l1, self.l2
        )


def reset_idxes(G):
    """Resets node indices such that they are numbered from 0 to num_nodes - 1
    :return: copy of G with relabelled node indices, mapping"""
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms"""
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, data, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label"""

        if masked_atom_indices is None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        data.masked_x = data.x.clone()
        for atom_idx in masked_atom_indices:
            data.masked_x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in {u, v} and bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor([self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(connected_edge_indices).to(
                    torch.int64
                )

        return data

    def __repr__(self):
        return (
            "{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})".format(
                self.__class__.__name__,
                self.num_atom_type,
                self.num_edge_type,
                self.mask_rate,
                self.mask_edge,
            )
        )


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    # ci = S / z
    return S / z


def get_num_task(dataset):
    """used in molecule_finetune.py"""
    if dataset == "tox21":
        return 12
    elif dataset in ["hiv", "bace", "bbbp", "donor"]:
        return 1
    elif dataset == "pcba":
        return 92
    elif dataset == "muv":
        return 17
    elif dataset == "toxcast":
        return 617
    elif dataset == "sider":
        return 27
    elif dataset == "clintox":
        return 2
    raise ValueError("Invalid dataset name.")


def map_param_to_block(shared_params_name, level):
    param_to_block = {}
    if level == "param_wise":
        for i, name in enumerate(shared_params_name):
            param_to_block[i] = i
        module_num = len(param_to_block)
        return param_to_block, module_num

    elif level == "module_wise":
        for i, name in enumerate(shared_params_name):
            if "x_embedding" in name:
                param_to_block[i] = 0
            elif "gnns" in name or "batch_norms" in name:
                param_to_block[i] = int(name.split(".")[1]) + 1
            else:
                raise ValueError("Invalid parameter name.")
        module_num = len(set(param_to_block.values()))
        return param_to_block, module_num


def setup_transform_criteria(args):
    transform, criterion = {}, {}
    for mode in args.aux_2D_mode:
        if mode == "EP":
            transform[mode] = NegativeEdge()
            criterion[mode] = nn.BCEWithLogitsLoss()
        if mode == "AM":
            transform[mode] = MaskAtom(
                num_atom_type=119,
                num_edge_type=5,
                mask_rate=args.mask_rate,
                mask_edge=args.mask_edge,
            )
            criterion[mode] = nn.CrossEntropyLoss()
        if mode == "CP":
            l1 = args.num_layer - 1
            l2 = l1 + args.csize
            transform[mode] = ExtractSubstructureContextPair(args.num_layer, l1, l2)
            criterion[mode] = nn.BCEWithLogitsLoss()
        if mode in ["IG", "MP"]:
            criterion[mode] = nn.BCEWithLogitsLoss()
    return transform, criterion


def setup_2D_models(args, num_aux_classes=0, device=torch.device("cpu")):
    # set up 2D SSL model
    aux_2D_support_model_list = dict.fromkeys(args.aux_2D_mode)
    # set up dummy 2D SSL model for EP
    if "EP" in args.aux_2D_mode:
        aux_2D_support_model_list["EP"] = None

    # set up 2D SSL model for IG
    infograph_discriminator_SSL_model = None
    if "IG" in args.aux_2D_mode:
        infograph_discriminator_SSL_model = Discriminator(args.emb_dim).to(device)
        aux_2D_support_model_list["IG"] = infograph_discriminator_SSL_model

    # set up 2D SSL model for AM
    molecule_atom_masking_model = None
    if "AM" in args.aux_2D_mode:
        molecule_atom_masking_model = torch.nn.Linear(args.emb_dim, 119).to(device)
        aux_2D_support_model_list["AM"] = molecule_atom_masking_model

    # set up 2D SSL model for CP
    molecule_context_model = None
    if "CP" in args.aux_2D_mode:
        l1 = args.num_layer - 1
        l2 = l1 + args.csize
        molecule_context_model = GNN(
            int(l2 - l1),
            args.emb_dim,
            JK=args.JK,  # type: ignore
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
        ).to(device)
        aux_2D_support_model_list["CP"] = molecule_context_model

    # set up 2D SSL model for Motif
    if "MP" in args.aux_2D_mode:
        motif_pred_model = nn.Linear(args.emb_dim, len(RDKIT_PROPS)).to(device)
        aux_2D_support_model_list["MP"] = motif_pred_model

    # set up 2D SSL model for GraphCL, JOAO, JOAOv2
    projection_head = None
    if set(args.aux_2D_mode).intersection(set(["GraphCL", "JOAO"])):
        mode = "GraphCL" if "GraphCL" in args.aux_2D_mode else "JOAO"
        projection_head = nn.Sequential(
            nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300)
        ).to(device)
        aux_2D_support_model_list[mode] = projection_head

    if "JOAOv2" in args.aux_2D_mode:
        projection_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300)
                )
                for _ in range(5)
            ]
        ).to(device)
        aux_2D_support_model_list["JOAOv2"] = projection_head

    # if args.gen_aux_tasks:
    #    aux_2D_support_model_list['aux'] = nn.Sequential(
    #        nn.Linear(args.emb_dim, 128),
    #        nn.ReLU(),
    #        nn.Linear(128, num_aux_classes),
    #       nn.Softmax(dim=1)).to(device)

    return aux_2D_support_model_list


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_roc = 0

    def __call__(self, val_roc):
        if self.best_roc - val_roc < self.min_delta:
            self.best_roc = val_roc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class GradientManipulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale_weights):
        # In forward pass, we just pass the input through and save scale_weights
        ctx.save_for_backward(input, scale_weights)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve input and scale_weights from the context
        input, scale_weights = ctx.saved_tensors

        # Scale the gradient by the weight vector
        grad_input = grad_output * scale_weights

        # The gradient with respect to the weights is the element-wise product of
        # the incoming gradient and the input
        grad_weights = grad_output * input

        return grad_input, grad_weights


class ScaleGradientNet(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(input_dim, device=device))
        # self.scale_weights.data.uniform_(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim))

    def forward(self, losses):
        output = GradientManipulation.apply(losses, self.scale_weights)
        return output
