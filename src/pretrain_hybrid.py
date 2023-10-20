import time
import os
import json
import numpy as np
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
import torch_geometric.transforms as T


from config import args
from dataloader import DataLoaderHybrid
from batch import BatchHybrid
from models import (GNN, AutoEncoder, Discriminator, SchNet,
                    VariationalAutoEncoder)
from pretrain_AM import do_AttrMasking
from pretrain_CP import do_ContextPred
from pretrain_EP import do_EdgePred
from pretrain_IG import do_InfoGraph
from util import (ExtractSubstructureContextPair, MaskAtom, NegativeEdge,
                  do_GraphCL, do_GraphCLv2, do_MotifPred, dual_CL,
                  update_augmentation_probability_JOAO,
                  update_augmentation_probability_JOAOv2,
                  setup_2D_models, setup_transform_criteria, seed_everything)
from datasets import Molecule3DHybridDataset, MoleculeGraphCLHybridDataset

def save_model(outdir, save_best, epoch=None):
    mode = '_'.join(args.aux_2D_mode)
    mode += '_3D' if args.aux_3D_mode else ''
    
    if save_best:
        global optimal_loss
        print('save model with loss: {:.5f}'.format(optimal_loss))
        torch.save(molecule_model_2D.state_dict(), outdir + f'{mode}_model.pth')
        saver_dict = {
            'model':    molecule_model_2D.state_dict(),
            'model_3D': molecule_model_3D.state_dict() if args.aux_3D_mode else None,
        }
        torch.save(saver_dict, outdir + f'{mode}_model_complete.pth')

    elif epoch is None:
        torch.save(molecule_model_2D.state_dict(), outdir + f'{mode}_model_final.pth')
        saver_dict = {
            'model':    molecule_model_2D.state_dict(),
            'model_3D': molecule_model_3D.state_dict() if args.aux_3D_mode else None,
        }
        torch.save(saver_dict, outdir + f'{mode}_model_complete_final.pth')

    else:
        torch.save(molecule_model_2D.state_dict(), outdir + f'{mode}_model_{epoch}.pth')
        saver_dict = {
            'model':    molecule_model_2D.state_dict(),
            'model_3D': molecule_model_3D.state_dict() if args.aux_3D_mode else None,
        }
        torch.save(saver_dict, outdir + f'{mode}_model_complete_{epoch}.pth')

    return


def compute_loss(args, batch, model, 
                 aux_2D_support_model_list, criterion, aux_model=None):
    if args.adapt == 'uncert':
        node_repr, logsigma = model.molecule_model(batch.x, batch.edge_index, batch.edge_attr)
    else:
        node_repr = model.molecule_model(batch.x, batch.edge_index, batch.edge_attr)
    
    logsigma = None
    molecule_repr = model.pool(node_repr, batch.batch)
    target_pred = model.graph_pred_linear(molecule_repr)
    y = batch.y.view(target_pred.shape).to(torch.float64)

    # Whether y is non-null or not.
    is_valid = y ** 2 > 0
    # Loss matrix
    loss_mat = criterion['target'](target_pred.double(), (y + 1) / 2)
    # loss matrix after removing null target
    loss_mat = torch.where(
        is_valid, loss_mat,
        torch.zeros(loss_mat.shape, device=target_pred.device, dtype=loss_mat.dtype))

    target_loss = torch.sum(loss_mat) / torch.sum(is_valid) ## target prediction loss

    aux_2D_loss_dict, aux_2D_acc_dict = get_aux_2D_loss(args, batch, node_repr, model.molecule_model,
                                                     molecule_repr, aux_2D_support_model_list, criterion, aux_model)     

    if args.adapt == 'uncert':
        return target_loss, aux_2D_loss_dict, aux_2D_acc_dict, logsigma
    return target_loss, aux_2D_loss_dict, aux_2D_acc_dict


def get_aux_2D_loss(args, batch, node_repr, molecule_model_2D,
                 molecule_2D_repr,
                 aux_2D_support_model_list, criterion, aux_model=None):
    aux_2D_loss, aux_2D_acc = dict.fromkeys(args.aux_2D_mode, 0), dict.fromkeys(args.aux_2D_mode, 0)

    ##### To obtain 2D SSL loss and acc
    if 'EP' in args.aux_2D_mode:
        aux_2D_loss['EP'], aux_2D_acc['EP'] = do_EdgePred(
            node_repr=node_repr, batch=batch, criterion=criterion['EP'])

    if 'IG' in args.aux_2D_mode:
        aux_2D_loss['IG'], aux_2D_acc['IG'] = do_InfoGraph(
            node_repr=node_repr, batch=batch,
            molecule_repr=molecule_2D_repr, criterion=criterion['IG'],
            infograph_discriminator_SSL_model=aux_2D_support_model_list['IG'])

    if 'AM' in args.aux_2D_mode:
        masked_node_repr = molecule_model_2D(batch.masked_x, batch.edge_index, batch.edge_attr)
        aux_2D_loss['AM'], aux_2D_acc['AM'] = do_AttrMasking(
            batch=batch, criterion=criterion['AM'], node_repr=masked_node_repr,
            molecule_atom_masking_model=aux_2D_support_model_list['AM'])

    if 'CP' in args.aux_2D_mode:
        aux_2D_loss['CP'], aux_2D_acc['CP'] = do_ContextPred(
            batch=batch, criterion=criterion['CP'], args=args,
            molecule_substruct_model=molecule_model_2D,
            molecule_context_model=aux_2D_support_model_list['CP'],
            molecule_readout_func=global_mean_pool)

    if 'MP' in args.aux_2D_mode:
        aux_2D_loss['MP'], aux_2D_acc['MP'] = do_MotifPred(
            batch=batch,
            molecule_repr=molecule_2D_repr, criterion=criterion['MP'],
            motif_pred_model=aux_2D_support_model_list['MP'])

    if aux_model is not None:
        aux_target = aux_model(batch)
        aux_pred = aux_2D_support_model_list['aux'](molecule_2D_repr)
        aux_2D_loss['aux'] = criterion['aux'](aux_pred, aux_target)
        aux_2D_acc['aux'] = 0 
        #float(torch.sum(torch.max(aux_pred.detach(), dim=1)[1] == aux_target).cpu().item())/len(aux_pred)

    return aux_2D_loss, aux_2D_acc


def train_no_aug(args, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    if args.aux_3D_mode:
        molecule_model_3D.train() # type: ignore
    for mode, support_model in aux_2D_support_model_list.items():
        if mode != 'EP':
            support_model.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0
    aux_2D_loss_accum = 0 
    aux_2D_loss_accum_dict, aux_2D_acc_accum_dict = dict.fromkeys(args.aux_2D_mode, 0), dict.fromkeys(args.aux_2D_mode, 0)

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for batch in l:
        batch = batch.to(device)

        node_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)
        loss = 0

        ##### To obtain 3D-2D SSL loss and acc
        if args.aux_3D_mode:
            molecule_3D_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch) #type:ignore
            CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)

            AE_loss_1 = AE_2D_3D_model(molecule_2D_repr, molecule_3D_repr)
            AE_loss_2 = AE_3D_2D_model(molecule_3D_repr, molecule_2D_repr)
            AE_acc_1 = AE_acc_2 = 0
            AE_loss = (AE_loss_1 + AE_loss_2) / 2
            CL_loss_accum += CL_loss.detach().cpu().item()
            CL_acc_accum += CL_acc
            AE_loss_accum += AE_loss.detach().cpu().item()
            AE_acc_accum += (AE_acc_1 + AE_acc_2) / 2

            if args.alpha_1 > 0:
                loss += CL_loss * args.alpha_1
            if args.alpha_2 > 0:
                loss += AE_loss * args.alpha_2

        if args.aux_2D_mode:
            aux_2D_loss_dict, aux_2D_acc_dict = get_aux_2D_loss(args, batch, node_repr, molecule_model_2D,
                                                                molecule_2D_repr, aux_2D_support_model_list, criterion)
            aux_2D_loss = torch.mean(torch.stack(list(aux_2D_loss_dict.values()))) ##TODO: add weighted sum and more efficient operation
            for mode in args.aux_2D_mode:
                aux_2D_acc_accum_dict[mode] += aux_2D_acc_dict[mode]
                aux_2D_loss_accum_dict[mode] += aux_2D_loss_dict[mode].detach().cpu().item()
        else:
            raise Exception

        aux_2D_loss_accum += aux_2D_loss.detach().cpu().item()
        #aux_2D_acc_accum += aux_2D_acc

        loss += aux_2D_loss * args.alpha_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)
    aux_2D_loss_accum /= len(loader)
    #aux_2D_acc_accum /= len(loader)
    temp_loss = args.alpha_1 * CL_loss_accum + args.alpha_2 * AE_loss_accum + args.alpha_3 * aux_2D_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(outdir, save_best=True)
    for mode in args.aux_2D_mode:
        aux_2D_acc_accum_dict[mode] /= len(loader) # type: ignore
        aux_2D_loss_accum_dict[mode] /= len(loader) # type: ignore

    print_str = f"CL Loss: {CL_loss_accum:.5f}  CL Acc: {CL_acc_accum:.5f}\t" + \
        f"AE Loss: {AE_loss_accum:.5f}  AE Acc: {AE_acc_accum:.5f}\t" + \
        "\t".join([f"{mode} Loss: {aux_2D_loss_accum_dict[mode]:.5f}  {mode} Acc: {aux_2D_acc_accum_dict[mode]:.5f}" for mode in args.aux_2D_mode]) 
    logger.info(print_str + f"\tTime: {time.time() - start_time:.5f}")

    return


def train_with_aug(args, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    if args.aux_3D_mode:
        molecule_model_3D.train() # type: ignore
    for mode, support_model in aux_2D_support_model_list.items():
        if mode != 'EP':
            support_model.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0
    aux_2D_loss_accum, aux_2D_acc_accum = 0, 0
    aux_2D_loss_accum_dict, aux_2D_acc_accum_dict = dict.fromkeys(args.aux_2D_mode, 0), dict.fromkeys(args.aux_2D_mode, 0)

    aug_prob_cp = loader.dataset.aug_prob
    n_aug = np.random.choice(25, 1, p=aug_prob_cp)[0]
    n_aug1, n_aug2 = n_aug // 5, n_aug % 5

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader

    for batch, batch1, batch2 in l:
        batch = batch.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        #batch = BatchHybrid.from_data_list(batch.to_data_list())

        node_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)
        loss = 0
        if args.aux_3D_mode:
            molecule_3D_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch) # type: ignore

            ##### To obtain 3D-2D SSL loss and acc
            CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)

            AE_loss_1 = AE_2D_3D_model(molecule_2D_repr, molecule_3D_repr)
            AE_loss_2 = AE_3D_2D_model(molecule_3D_repr, molecule_2D_repr)
            AE_acc_1 = AE_acc_2 = 0
            AE_loss = (AE_loss_1 + AE_loss_2) / 2
            CL_loss_accum += CL_loss.detach().cpu().item()
            CL_acc_accum += CL_acc
            AE_loss_accum += AE_loss.detach().cpu().item()
            AE_acc_accum += (AE_acc_1 + AE_acc_2) / 2
            if args.alpha_1 > 0:
                loss += CL_loss * args.alpha_1
            if args.alpha_2 > 0:
                loss += AE_loss * args.alpha_2

        ##### To obtain 2D SSL loss and acc
        if args.aux_2D_mode:
            aux_2D_loss_dict, aux_2D_acc_dict = get_aux_2D_loss(args, batch, node_repr, molecule_model_2D,
                                                                molecule_2D_repr, aux_2D_support_model_list, criterion)

            if set(args.aux_2D_mode).intersection(set(['GraphCL', 'JOAO'])):
                mode = 'GraphCL' if 'GraphCL' in args.aux_2D_mode else 'JOAO'
                aux_2D_loss_dict[mode] = do_GraphCL(
                    batch1=batch1, batch2=batch2,
                    molecule_model_2D=molecule_model_2D, projection_head=aux_2D_support_model_list[mode],
                    molecule_readout_func=molecule_readout_func)
            elif 'JOAOv2' in args.aux_2D_mode:
                aux_2D_loss_dict['JOAOv2'] = do_GraphCLv2(
                    batch1=batch1, batch2=batch2, n_aug1=n_aug1, n_aug2=n_aug2,
                    molecule_model_2D=molecule_model_2D, projection_head=aux_2D_support_model_list['JOAOv2'],
                    molecule_readout_func=molecule_readout_func)
            
            for mode in args.aux_2D_mode:
                if mode not in ['GraphCL', 'JOAO', 'JOAOv2']:
                    aux_2D_acc_accum_dict[mode] += aux_2D_acc_dict[mode]
                aux_2D_loss_accum_dict[mode] += aux_2D_loss_dict[mode].detach().cpu().item()
        
        aux_2D_loss = torch.mean(torch.stack(list(aux_2D_loss_dict.values()))) # type: ignore
        ##TODO: add weighted sum and more efficient operation
        aux_2D_loss_accum += aux_2D_loss.detach().cpu().item()
        #aux_2D_acc_accum += aux_2D_acc
        loss += aux_2D_loss * args.alpha_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global aug_prob
    if 'JOAO' in args.aux_2D_mode:
        aug_prob = update_augmentation_probability_JOAO(
            loader=loader, molecule_model_2D=molecule_model_2D, projection_head=aux_2D_support_model_list['JOAO'],
            molecule_readout_func=molecule_readout_func,
            gamma_joao=args.gamma_joao, device=device)
    elif 'JOAOv2' in args.aux_2D_mode:
        aug_prob = update_augmentation_probability_JOAOv2(
            loader=loader, molecule_model_2D=molecule_model_2D, projection_head=aux_2D_support_model_list['JOAOv2'],
            molecule_readout_func=molecule_readout_func,
            gamma_joao=args.gamma_joaov2, device=device)

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)
    aux_2D_loss_accum /= len(loader)
    aux_2D_acc_accum /= len(loader)
    temp_loss = args.alpha_1 * CL_loss_accum + args.alpha_2 * AE_loss_accum + args.alpha_3 * aux_2D_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(outdir, save_best=True)
    
    for mode in args.aux_2D_mode:
        aux_2D_acc_accum_dict[mode] /= len(loader) # type: ignore
        aux_2D_loss_accum_dict[mode] /= len(loader) # type: ignore

    print_str = f"CL Loss: {CL_loss_accum:.5f}  CL Acc: {CL_acc_accum:.5f}\t" + \
        f"AE Loss: {AE_loss_accum:.5f}  AE Acc: {AE_acc_accum:.5f}\t" + \
        "\t".join([f"{mode} Loss: {aux_2D_loss_accum_dict[mode]:.5f}  {mode} Acc: {aux_2D_acc_accum_dict[mode]:.5f}" for mode in args.aux_2D_mode]) 
    logger.info(print_str + f"\tTime: {time.time() - start_time:.5f}")

    return


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(args.runseed)
    output_log = 'train.log' if not args.test_time_training else 'test.log'

    # set up output directory
    outdir = None
    if args.output_model_dir != '':
        os.makedirs(args.output_model_dir) if not os.path.exists(args.output_model_dir) else None
        # create a new version inside the output_dir with version control incrementing version number
        try:
            version = max([int(i.split('version_')[-1]) for i in os.listdir(args.output_model_dir) if 'version_' in i]) + 1
        except:
            version = 0
        outdir = os.path.join(args.output_model_dir, 'version_'+str(version)+'/')
        os.makedirs(outdir)
        output_log = os.path.join(outdir, output_log)

        # save hyperparameters
        json.dump(args.__dict__, open(os.path.join(outdir, 'config.json'), 'w'),
                  indent=4, sort_keys=True)

    logger = logging.getLogger()
    file_handler = logging.FileHandler(output_log)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info('========== Use {} for 2D SSL =========='.format(args.aux_2D_mode))
    
    transform, criterion = setup_transform_criteria(args)

    data_root = '../../datasets/{}/'.format('GEOM_3D_nmol50000_nconf5_nupper1000') \
        if args.input_data_dir == '' \
        else '{}/{}/'.format(args.input_data_dir, args.dataset)
    
    compose_transform = T.Compose(transform.values()) # type: ignore
    if set(args.aux_2D_mode).intersection(set(['GraphCL', 'JOAO', 'JOAOv2'])):
        dataset = MoleculeGraphCLHybridDataset(data_root, dataset=args.dataset,
                                                transform=compose_transform, mask_ratio=args.SSL_masking_ratio)
        dataset.set_augMode('sample')
        dataset.set_augStrength(0.2)
    else:
        dataset = Molecule3DHybridDataset(data_root, dataset=args.dataset,
                                           transform=compose_transform, mask_ratio=args.SSL_masking_ratio)

    if args.DEBUG:
        dataset = dataset[:500]
    loader = DataLoaderHybrid(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    # set up 2D base model
    molecule_model_2D = GNN(args.num_layer, args.emb_dim, JK=args.JK,
                            drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    molecule_readout_func = global_mean_pool
    molecule_model_3D = None
    aux_2D_support_model_list = setup_2D_models(args, num_aux_classes=0, device=device)
    aux_2D_support_model_list = nn.ModuleDict(aux_2D_support_model_list)

    # set up parameters
    model_param_group = [{'params': molecule_model_2D.parameters(),
                          'lr':     args.lr * args.gnn_lr_scale}]

    # set up 3D base model
    if args.aux_3D_mode:
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)

        # set up VAE model
        if args.AE_model == 'AE':
            AE_2D_3D_model = AutoEncoder(
                emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
            AE_3D_2D_model = AutoEncoder(
                emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
        elif args.AE_model == 'VAE':
            AE_2D_3D_model = VariationalAutoEncoder(
                emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target,
                beta=args.beta).to(device)
            AE_3D_2D_model = VariationalAutoEncoder(
                emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target,
                beta=args.beta).to(device)
        else:
            raise Exception

        model_param_group += [{'params': molecule_model_3D.parameters(),
                          'lr':     args.lr * args.schnet_lr_scale},
                         {'params': AE_2D_3D_model.parameters(),
                          'lr': args.lr * args.gnn_lr_scale},
                         {'params': AE_3D_2D_model.parameters(),
                          'lr': args.lr * args.schnet_lr_scale}] 
        
    for mode, aux_2D_support_model in aux_2D_support_model_list.items():
        if mode != 'EP':
            model_param_group.append({'params': aux_2D_support_model.parameters(),
                                    'lr': args.lr*args.lr_scale})
    # set up optimizers
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    # for GraphCL, JOAO, JOAOv2
    aug_prob = np.ones(25) / 25
    np.set_printoptions(precision=3, floatmode='fixed')

    if set(args.aux_2D_mode).intersection(set(['GraphCL', 'JOAO', 'JOAOv2'])):
        train_function = train_with_aug
    else:
        train_function = train_no_aug

    # start training
    for epoch in range(1, args.epochs + 1):
        logger.info('epoch: {}'.format(epoch))

        if set(args.aux_2D_mode).intersection(set(['JOAO', 'JOAOv2'])):
            dataset.set_augProb(aug_prob) # type: ignore
            logger.info('augmentation probability\t', aug_prob)
        train_function(args, device, loader, optimizer)

        if epoch == 50:
            save_model(outdir, save_best=False, epoch=50)

    if set(args.aux_2D_mode).intersection(set(['JOAO', 'JOAOv2'])):
        logger.info('augmentation probability\t', aug_prob)

    # save final model weight
    save_model(outdir, save_best=False)
