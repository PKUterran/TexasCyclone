import argparse
from cmath import isnan
import torch
import torch.nn.functional as F
import json
from typing import List, Dict, Any, Tuple
from functools import reduce
from time import time
from tqdm import tqdm

from data.graph import Netlist, expand_netlist
from data.pretrain import DIS_ANGLE_TYPE, load_pretrain_data
from data.utils import set_seed, mean_dict
from train.model import NaiveGNN, PlaceGNN
import dgl
import random


def pretrain_ours(
        args: argparse.Namespace,
        train_datasets: List[str],
        valid_datasets: List[str],
        test_datasets: List[str],
        log_dir: str = None,
        fig_dir: str = None,
        model_dir: str = None,
):
    # Configure environment
    # torch.autograd.set_detect_anomaly(True)
    logs: List[Dict[str, Any]] = []
    use_cuda = args.device != 'cpu'
    use_tqdm = args.use_tqdm
    device = torch.device(args.device)

    set_seed(args.seed, use_cuda=use_cuda)

    # Load data
    print(f'Loading data...')

    def unpack_netlist_dis_angle(list_netlist_dict_nid_dis_angle: List[Tuple[Netlist, Dict[int, DIS_ANGLE_TYPE]]]
                                 ) -> List[Tuple[Netlist, DIS_ANGLE_TYPE]]:
        list_netlist_dis_angle = []
        for netlist, dict_nid_dis_angle in list_netlist_dict_nid_dis_angle:
            dict_netlist = expand_netlist(netlist)
            for nid, sub_nl in dict_netlist.items():
                list_netlist_dis_angle.append((sub_nl, dict_nid_dis_angle[nid]))
        return list_netlist_dis_angle

    train_list_netlist_dis_angle = unpack_netlist_dis_angle(
        [load_pretrain_data(dataset) for dataset in train_datasets])
    valid_list_netlist_dis_angle = unpack_netlist_dis_angle(
        [load_pretrain_data(dataset) for dataset in valid_datasets])
    test_list_netlist_dis_angle = unpack_netlist_dis_angle(
        [load_pretrain_data(dataset) for dataset in test_datasets])
    print(f'\t# of samples: '
          f'{len(train_list_netlist_dis_angle)} train, '
          f'{len(valid_list_netlist_dis_angle)} valid, '
          f'{len(test_list_netlist_dis_angle)} test.')

    # Configure model
    print(f'Building model...')
    sample_netlist = train_list_netlist_dis_angle[0][0] \
        if train_list_netlist_dis_angle else test_list_netlist_dis_angle[0][0]
    raw_cell_feats = sample_netlist.cell_prop_dict['feat'].shape[1]
    raw_net_feats = sample_netlist.net_prop_dict['feat'].shape[1]
    raw_pin_feats = sample_netlist.pin_prop_dict['feat'].shape[1]
    config = {
        'DEVICE': device,
        'CELL_FEATS': args.cell_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'PASS_TYPE': args.pass_type,
    }

    if args.gnn == 'naive':
        model = NaiveGNN(raw_cell_feats, raw_net_feats, raw_pin_feats, config)
    elif args.gnn == 'place':
        config.update({
            'NUM_LAYERS': 1,
            'NUM_HEADS': 2,
        })
        model = PlaceGNN(raw_cell_feats, raw_net_feats, raw_pin_feats, config)
    else:
        assert False, f'Undefined GNN {args.gnn}'

    if args.model:
        model_dicts = torch.load(f'model/{args.model}.pkl', map_location=device)
        model.load_state_dict(model_dicts)
        model.eval()
    n_param = 0
    for name, param in model.named_parameters():
        print(f'\t{name}: {param.shape}')
        n_param += reduce(lambda x, y: x * y, param.shape)
    print(f'# of parameters: {n_param}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=(1 - args.lr_decay))

    # Train model
    best_metric = 1e8  # lower is better

    for epoch in range(0, args.epochs + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})

        def train(list_netlist_dis_angle: List[Tuple[Netlist, DIS_ANGLE_TYPE]]):
            model.train()
            t1 = time()
            losses = []
            random.shuffle(list_netlist_dis_angle)
            n_netlist = len(list_netlist_dis_angle)
            iter_i_netlist_dis_angle = tqdm(enumerate(list_netlist_dis_angle), total=n_netlist) \
                if use_tqdm else enumerate(list_netlist_dis_angle)

            batch_netlist = []
            total_batch_nodes_num = 0
            total_batch_edge_idx = 0
            batch_cell_feature = []
            batch_net_feature = []
            batch_pin_feature = []
            sub_netlist_feature_idrange = []
            batch_angle = []
            batch_dis = []
            batch_cell_size = []
            cnt_graph = 0

            for j, (netlist, dis_angle) in iter_i_netlist_dis_angle:
                cnt_graph+=1
                batch_netlist.append(netlist)
                father, _ = netlist.graph.edges(etype='points-to')
                edge_idx_num = father.size(0)
                sub_netlist_feature_idrange.append([total_batch_edge_idx, total_batch_edge_idx + edge_idx_num])
                total_batch_edge_idx += edge_idx_num
                total_batch_nodes_num += netlist.graph.num_nodes('cell')
                batch_cell_feature.append(netlist.cell_prop_dict['feat'])
                batch_net_feature.append(netlist.net_prop_dict['feat'])
                batch_pin_feature.append(netlist.pin_prop_dict['feat'])
                batch_dis.append(dis_angle[0])
                batch_angle.append(dis_angle[1])
                batch_cell_size.append(netlist.cell_prop_dict['size'])
                if cnt_graph > 128 or j == n_netlist - 1:
                    batch_cell_feature = torch.vstack(batch_cell_feature)
                    batch_net_feature = torch.vstack(batch_net_feature)
                    batch_pin_feature = torch.vstack(batch_pin_feature)
                    batch_cell_size = torch.vstack(batch_cell_size)
                    batch_graph = dgl.batch([sub_netlist.graph for sub_netlist in batch_netlist])
                    batch_edge_dis, batch_edge_angle = model.forward(
                        batch_graph, (batch_cell_feature, batch_net_feature, batch_pin_feature),batch_cell_size)
                    # batch_edge_dis,batch_edge_angle = batch_edge_dis.cpu(),batch_edge_angle.cpu()
                    for nid in range(len(batch_dis)):
                        begin_idx, end_idx = sub_netlist_feature_idrange[nid]
                        edge_dis, edge_angle = batch_edge_dis[begin_idx:end_idx], batch_edge_angle[begin_idx:end_idx]
                        dis_angle = [batch_dis[nid], batch_angle[nid]]
                        edge_dis_loss = F.mse_loss(torch.log(edge_dis + 1),
                                                   torch.log(dis_angle[0].to(device) + 1))
                        edge_angle_loss = F.mse_loss(edge_angle, dis_angle[1].to(device))
                        loss = sum((
                            edge_dis_loss * 1.0,
                            edge_angle_loss * 1.0,
                        ))
                        losses.append(loss)
                    (sum(losses) / len(losses)).backward()
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=20, norm_type=2)
                    optimizer.step()
                    losses.clear()
                    batch_netlist = []
                    sub_netlist_feature_idrange = []
                    total_batch_nodes_num = 0
                    total_batch_edge_idx = 0
                    batch_cell_feature = []
                    batch_net_feature = []
                    batch_pin_feature = []
                    batch_angle = []
                    batch_dis = []
                    batch_cell_size = []
                    cnt_graph = 0
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()
            print(f"\tTraining time per epoch: {time() - t1}")

        def evaluate(list_netlist_dis_angle: List[Tuple[Netlist, DIS_ANGLE_TYPE]],
                     dataset_name: str, netlist_names: List[str], verbose=True) -> float:
            model.eval()
            ds = []
            dni = {}
            print(f'\tEvaluate {dataset_name}:')
            n_netlist = len(list_netlist_dis_angle)
            iter_i_netlist_dis_angle = tqdm(enumerate(list_netlist_dis_angle), total=n_netlist) \
                if use_tqdm else enumerate(list_netlist_dis_angle)

            batch_netlist = []
            total_batch_nodes_num = 0
            total_batch_edge_idx = 0
            batch_cell_feature = []
            batch_net_feature = []
            batch_pin_feature = []
            sub_netlist_feature_idrange = []
            batch_angle = []
            batch_dis = []
            batch_cell_size = []

            for j, (netlist, dis_angle) in iter_i_netlist_dis_angle:
                dni[j] = {}
                batch_netlist.append(j)
                father, _ = netlist.graph.edges(etype='points-to')
                edge_idx_num = father.size(0)
                sub_netlist_feature_idrange.append([total_batch_edge_idx, total_batch_edge_idx + edge_idx_num])
                total_batch_edge_idx += edge_idx_num
                total_batch_nodes_num += netlist.graph.num_nodes('cell')
                batch_cell_feature.append(netlist.cell_prop_dict['feat'])
                batch_net_feature.append(netlist.net_prop_dict['feat'])
                batch_pin_feature.append(netlist.pin_prop_dict['feat'])
                batch_dis.append(dis_angle[0])
                batch_angle.append(dis_angle[1])
                batch_cell_size.append(netlist.cell_prop_dict['size'])
                if total_batch_nodes_num > 10000 or j == n_netlist - 1:
                    batch_cell_feature = torch.vstack(batch_cell_feature)
                    batch_net_feature = torch.vstack(batch_net_feature)
                    batch_pin_feature = torch.vstack(batch_pin_feature)
                    batch_cell_size = torch.vstack(batch_cell_size)
                    batch_graph = []
                    for j_ in batch_netlist:
                        sub_netlist, _ = list_netlist_dis_angle[j_]
                        batch_graph.append(sub_netlist.graph)
                    batch_graph = dgl.batch(batch_graph)
                    batch_edge_dis, batch_edge_angle = model.forward(
                        batch_graph, (batch_cell_feature, batch_net_feature, batch_pin_feature),batch_cell_size)
                    # batch_edge_dis,batch_edge_angle = batch_edge_dis.cpu(),batch_edge_angle.cpu()
                    for nid in range(len(batch_dis)):
                        begin_idx, end_idx = sub_netlist_feature_idrange[nid]
                        edge_dis, edge_angle = batch_edge_dis[begin_idx:end_idx], batch_edge_angle[begin_idx:end_idx]
                        dis_angle = [batch_dis[nid], batch_angle[nid]]
                        edge_dis_loss = F.mse_loss(torch.log(edge_dis + 1),
                                                   torch.log(dis_angle[0].to(device) + 1))
                        edge_angle_loss = F.mse_loss(edge_angle, dis_angle[1].to(device))
                        loss = sum((
                            edge_dis_loss * 1.0,
                            edge_angle_loss * 0.1,
                        ))
                        # print("dis angle",dis_angle[0])
                        nid_ = batch_netlist[nid]
                        dni[nid_]['net_dis_loss'] = float(edge_dis_loss.data)
                        dni[nid_]['net_angle_loss'] = float(edge_angle_loss.data)
                        dni[nid_]['loss'] = float(loss.data)
                        del loss
                    batch_netlist = []
                    sub_netlist_feature_idrange = []
                    total_batch_nodes_num = 0
                    total_batch_edge_idx = 0
                    batch_cell_feature = []
                    batch_net_feature = []
                    batch_pin_feature = []
                    batch_angle = []
                    batch_dis = []
                    batch_cell_size = []
                    torch.cuda.empty_cache()

            net_dis_loss = sum(v['net_dis_loss'] for v in dni.values()) / len(dni)
            net_angle_loss = sum(v['net_angle_loss'] for v in dni.values()) / len(dni)
            total_loss = sum(v['loss'] for v in dni.values()) / len(dni)
            print(f'\t\tEdge Distance Loss: {net_dis_loss}')
            print(f'\t\tEdge Angle Loss: {net_angle_loss}')
            print(f'\t\tTotal Loss: {total_loss}')
            d = {
                f'{dataset_name}_net_dis_loss': float(net_dis_loss),
                f'{dataset_name}_net_angle_loss': float(net_angle_loss),
                f'{dataset_name}_loss': float(total_loss),
            }
            ds.append(d)
            logs[-1].update(mean_dict(ds))
            return logs[-1][f'{dataset_name}_loss']

        t0 = time()
        if epoch:
            for _ in range(args.train_epoch):
                train(train_list_netlist_dis_angle)
                scheduler.step()
        logs[-1].update({'train_time': time() - t0})
        t2 = time()
        valid_metric = None
        evaluate(train_list_netlist_dis_angle, 'train', train_datasets, verbose=False)
        if len(valid_list_netlist_dis_angle):
            valid_metric = evaluate(valid_list_netlist_dis_angle, 'valid', valid_datasets)
        if len(test_list_netlist_dis_angle):
            evaluate(test_list_netlist_dis_angle, 'test', test_datasets)

        if valid_metric is not None and valid_metric < best_metric:
            best_metric = valid_metric
            if model_dir is not None:
                print(f'\tSaving model to {model_dir}/{args.name}.pkl ...:')
                torch.save(model.state_dict(), f'{model_dir}/{args.name}.pkl')

        print("\tinference time", time() - t2)
        logs[-1].update({'eval_time': time() - t2})
        if log_dir is not None:
            with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
                json.dump(logs, fp)
