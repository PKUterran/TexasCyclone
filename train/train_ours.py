import argparse
from copy import copy
import numpy as np
import torch
import json
from typing import List, Dict, Any
from functools import reduce
from time import time
from tqdm import tqdm

from data.graph import Netlist, Layout, expand_netlist, sequentialize_netlist, assemble_layout_with_netlist_info
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_dis_angle
from data.utils import set_seed, mean_dict
from train.model import NaiveGNN, PlaceGNN
from train.functions import AreaLoss, HPWLLoss, SampleOverlapLoss, MacroOverlapLoss, SampleNetOverlapLoss
import dgl


def train_ours(
        args: argparse.Namespace,
        train_datasets: List[str],
        valid_datasets: List[str],
        test_datasets: List[str],
        log_dir: str = None,
        fig_dir: str = None,
        model_dir: str = None,
):
    # Configure environment
    logs: List[Dict[str, Any]] = []
    use_cuda = args.device != 'cpu'
    use_tqdm = args.use_tqdm
    device = torch.device(args.device)

    set_seed(args.seed, use_cuda=use_cuda)

    # Load data
    # torch.autograd.set_detect_anomaly(True)
    print(f'Loading data...')
    train_netlists = [netlist_from_numpy_directory(dataset) for dataset in train_datasets]
    valid_netlists = [netlist_from_numpy_directory(dataset) for dataset in valid_datasets]
    test_netlists = [netlist_from_numpy_directory(dataset) for dataset in test_datasets]
    print(f'\t# of samples: '
          f'{len(train_netlists)} train, '
          f'{len(valid_netlists)} valid, '
          f'{len(test_netlists)} test.')

    # Configure model
    print(f'Building model...')
    sample_netlist = train_netlists[0] if train_netlists else test_netlists[0]
    raw_cell_feats = sample_netlist.cell_prop_dict['feat'].shape[1]
    raw_net_feats = sample_netlist.net_prop_dict['feat'].shape[1]
    raw_pin_feats = sample_netlist.pin_prop_dict['feat'].shape[1]
    config = {
        'DEVICE': device,
        'CELL_FEATS': args.cell_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'NUM_LAYERS': 3,
        'NUM_HEADS': 8,
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
        print(f'\tUsing model model/{args.model}.pkl')
        model_dicts = torch.load(f'model/{args.model}.pkl', map_location=device)
        model.load_state_dict(model_dicts)
        model.eval()
    n_param = 0
    for name, param in model.named_parameters():
        print(f'\t{name}: {param.shape}')
        n_param += reduce(lambda x, y: x * y, param.shape)
    print(f'# of parameters: {n_param}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))

    # Train model
    best_metric = 1e8  # lower is better
    evaluate_cell_pos_corner_dict = {}
    sample_overlap_loss_op = SampleOverlapLoss(span=4)
    macro_overlap_loss_op = MacroOverlapLoss(max_cap=50)
    area_loss_op = AreaLoss(device)
    hpwl_loss_op = HPWLLoss(device)
    cong_loss_op = SampleNetOverlapLoss(device, span=4)

    for epoch in range(0, args.epochs + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})

        def train(netlists: List[Netlist]):
            model.train()
            t1 = time()
            losses = []
            seq_netlists = reduce(lambda x, y: x + y, [sequentialize_netlist(nl) for nl in netlists])
            n_netlist = len(seq_netlists)
            iter_i_netlist = tqdm(enumerate(seq_netlists), total=n_netlist) \
                if use_tqdm else enumerate(seq_netlists)

            batch_netlist = []
            total_batch_nodes_num = 0
            total_batch_edge_idx = 0
            batch_cell_feature = []
            batch_net_feature = []
            batch_pin_feature = []
            sub_netlist_feature_idrange = []

            for j, netlist in iter_i_netlist:
                batch_netlist.append(netlist)
                father, _ = netlist.graph.edges(etype='points-to')
                edge_idx_num = father.size(0)
                sub_netlist_feature_idrange.append([total_batch_edge_idx, total_batch_edge_idx + edge_idx_num])
                total_batch_edge_idx += edge_idx_num
                total_batch_nodes_num += netlist.graph.num_nodes('cell')
                batch_cell_feature.append(netlist.cell_prop_dict['feat'])
                batch_net_feature.append(netlist.net_prop_dict['feat'])
                batch_pin_feature.append(netlist.pin_prop_dict['feat'])
                if total_batch_nodes_num > 10000 or j == n_netlist - 1:
                    batch_cell_feature = torch.vstack(batch_cell_feature)
                    batch_net_feature = torch.vstack(batch_net_feature)
                    batch_pin_feature = torch.vstack(batch_pin_feature)
                    batch_graph = dgl.batch([sub_netlist.graph for sub_netlist in batch_netlist])
                    batch_edge_dis, batch_edge_angle = model.forward(
                        batch_graph, (batch_cell_feature, batch_net_feature, batch_pin_feature))
                    # batch_edge_dis,batch_edge_angle = batch_edge_dis.cpu(),batch_edge_angle.cpu()
                    for nid, sub_netlist in enumerate(batch_netlist):
                        begin_idx, end_idx = sub_netlist_feature_idrange[nid]
                        edge_dis, edge_angle = batch_edge_dis[begin_idx:end_idx], batch_edge_angle[begin_idx:end_idx]
                        layout, dis_loss = layout_from_netlist_dis_angle(sub_netlist, edge_dis, edge_angle)
                        sample_overlap_loss = sample_overlap_loss_op.forward(layout)
                        macro_overlap_loss = macro_overlap_loss_op.forward(layout)
                        overlap_loss = sample_overlap_loss + macro_overlap_loss * 10
                        area_loss = area_loss_op.forward(layout, limit=[0, 0, *layout.netlist.layout_size])
                        hpwl_loss = hpwl_loss_op.forward(layout)
                        assert not torch.isnan(dis_loss), f"{dis_loss}"
                        assert not torch.isnan(hpwl_loss)
                        assert not torch.isnan(area_loss)
                        assert not torch.isnan(macro_overlap_loss)
                        assert not torch.isnan(sample_overlap_loss)
                        assert not torch.isinf(dis_loss), f"{dis_loss}"
                        assert not torch.isinf(hpwl_loss)
                        assert not torch.isinf(area_loss)
                        assert not torch.isinf(macro_overlap_loss)
                        assert not torch.isinf(sample_overlap_loss)
                        loss = sum((
                            args.dis_lambda * dis_loss,
                            args.overlap_lambda * overlap_loss,
                            args.area_lambda * area_loss,
                            args.hpwl_lambda * hpwl_loss,
                            # args.cong_lambda * cong_loss,
                        ))
                        losses.append(loss)
                        # if len(losses) >= args.batch:
                    (sum(losses) / len(losses)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                    optimizer.step()
                    losses.clear()
                    batch_netlist = []
                    sub_netlist_feature_idrange = []
                    total_batch_nodes_num = 0
                    total_batch_edge_idx = 0
                    batch_cell_feature = []
                    batch_net_feature = []
                    batch_pin_feature = []
                    torch.cuda.empty_cache()
            print(f"\tTraining time per epoch: {time() - t1}")

        def evaluate(netlists: List[Netlist], dataset_name: str, netlist_names: List[str], verbose=True) -> float:
            model.eval()
            ds = []
            print(f'\tEvaluate {dataset_name}:')
            n_netlist = len(netlists)
            iter_name_netlist = tqdm(zip(netlist_names, netlists), total=n_netlist) \
                if use_tqdm else zip(netlist_names, netlists)
            for netlist_name, netlist in iter_name_netlist:
                print(f'\tFor {netlist_name}:')
                dict_netlist = expand_netlist(netlist)
                iter_i_sub_netlist = tqdm(dict_netlist.items(), total=len(dict_netlist.items()), leave=False) \
                    if use_tqdm else dict_netlist.items()
                total_len = len(dict_netlist.items())
                dni: Dict[int, Dict[str, Any]] = {}  # dict_netlist_info

                batch_netlist_id = []
                total_batch_nodes_num = 0
                total_batch_edge_idx = 0
                batch_cell_feature = []
                batch_net_feature = []
                batch_pin_feature = []
                sub_netlist_feature_idrange = []
                cnt = 0

                for nid, sub_netlist in iter_i_sub_netlist:
                    dni[nid] = {}
                    batch_netlist_id.append(nid)
                    father, _ = sub_netlist.graph.edges(etype='points-to')
                    edge_idx_num = father.size(0)
                    sub_netlist_feature_idrange.append([total_batch_edge_idx, total_batch_edge_idx + edge_idx_num])
                    total_batch_edge_idx += edge_idx_num
                    total_batch_nodes_num += sub_netlist.graph.num_nodes('cell')
                    batch_cell_feature.append(sub_netlist.cell_prop_dict['feat'])
                    batch_net_feature.append(sub_netlist.net_prop_dict['feat'])
                    batch_pin_feature.append(sub_netlist.pin_prop_dict['feat'])
                    if total_batch_nodes_num > 10000 or cnt == total_len - 1:
                        batch_cell_feature = torch.vstack(batch_cell_feature)
                        batch_net_feature = torch.vstack(batch_net_feature)
                        batch_pin_feature = torch.vstack(batch_pin_feature)
                        batch_graph = []
                        for nid_ in batch_netlist_id:
                            netlist = dict_netlist[nid_]
                            batch_graph.append(netlist.graph)
                        # batch_graph = dgl.batch([sub_netlist.graph for _,sub_netlist in batch_netlist_id])
                        batch_graph = dgl.batch(batch_graph)
                        batch_edge_dis, batch_edge_angle = model.forward(
                            batch_graph, (batch_cell_feature, batch_net_feature, batch_pin_feature))
                        # batch_edge_dis,batch_edge_angle = batch_edge_dis.cpu(),batch_edge_angle.cpu()
                        for j, nid_ in enumerate(batch_netlist_id):
                            sub_netlist_ = dict_netlist[nid_]
                            begin_idx, end_idx = sub_netlist_feature_idrange[j]
                            edge_dis, edge_angle = \
                                batch_edge_dis[begin_idx:end_idx], batch_edge_angle[begin_idx:end_idx]
                            layout, dis_loss = layout_from_netlist_dis_angle(sub_netlist_, edge_dis, edge_angle)
#                             print(nid_, layout.netlist.layout_size)
                            dni[nid_]['dis_loss'] = float(dis_loss.cpu().clone().detach().data)
                            sample_overlap_loss = sample_overlap_loss_op.forward(layout).cpu().clone().detach()
                            dni[nid_]['sample_overlap_loss'] = sample_overlap_loss.data
                            macro_overlap_loss = macro_overlap_loss_op.forward(layout).cpu().clone().detach()
                            dni[nid_]['macro_overlap_loss'] = float(macro_overlap_loss.data)
                            dni[nid_]['overlap_loss'] = dni[nid_]['sample_overlap_loss'] + dni[nid_][
                                'macro_overlap_loss'] * 10
                            area_loss = area_loss_op.forward(
                                layout, limit=[0, 0, *layout.netlist.layout_size]).cpu().clone().detach()
                            dni[nid_]['area_loss'] = float(area_loss.data)
                            hpwl_loss = hpwl_loss_op.forward(layout).cpu().clone().detach()
                            dni[nid_]['hpwl_loss'] = hpwl_loss.data
                            # cong_loss = cong_loss_op.forward(layout).cpu().clone().detach()
                            # dni[nid_]['cong_loss'] = float(cong_loss.data)
                            dni[nid_]['cell_pos'] = copy(layout.cell_pos)
                            assert not torch.isnan(dis_loss)
                            # assert not torch.isnan(cong_loss)
                            assert not torch.isnan(hpwl_loss)
                            assert not torch.isnan(area_loss)
                            assert not torch.isnan(macro_overlap_loss)
                            assert not torch.isnan(sample_overlap_loss)
                            assert not torch.isinf(dis_loss)
                            # assert not torch.isinf(cong_loss)
                            assert not torch.isinf(hpwl_loss)
                            assert not torch.isinf(area_loss)
                            assert not torch.isinf(macro_overlap_loss)
                            assert not torch.isinf(sample_overlap_loss)
                        batch_netlist_id = []
                        sub_netlist_feature_idrange = []
                        total_batch_nodes_num = 0
                        total_batch_edge_idx = 0
                        batch_cell_feature = []
                        batch_net_feature = []
                        batch_pin_feature = []
                    cnt += 1
                    torch.cuda.empty_cache()
                layout = assemble_layout_with_netlist_info(dni, dict_netlist, device=device)
                # layout = assemble_layout({nid: nif['layout'] for nid, nif in dni.items()}, device=torch.device("cpu"))
                dis_loss = sum(v['dis_loss'] for v in dni.values()) / len(dni)
                sample_overlap_loss = sum(v['sample_overlap_loss'] for v in dni.values()) / len(dni)
                macro_overlap_loss = sum(v['macro_overlap_loss'] for v in dni.values()) / len(dni)
                overlap_loss = sum(v['overlap_loss'] for v in dni.values()) / len(dni)
                area_loss = sum(v['area_loss'] for v in dni.values()) / len(dni)
                hpwl_loss = sum(v['hpwl_loss'] for v in dni.values()) / len(dni)
                # cong_loss = sum(v['cong_loss'] for v in dni.values()) / len(dni)
                loss = sum((
                    args.dis_lambda * dis_loss,
                    args.overlap_lambda * overlap_loss,
                    args.area_lambda * area_loss,
                    args.hpwl_lambda * hpwl_loss,
                    # args.cong_lambda * cong_loss,
                ))
                print(f'\t\tDiscrepancy Loss: {dis_loss}')
                print(f'\t\tSample Overlap Loss: {sample_overlap_loss}')
                print(f'\t\tMacro Overlap Loss: {macro_overlap_loss}')
                print(f'\t\tTotal Overlap Loss: {overlap_loss}')
                print(f'\t\tArea Loss: {area_loss}')
                print(f'\t\tHPWL Loss: {hpwl_loss}')
                # print(f'\t\tCongestion Loss: {cong_loss}')
                print(f'\t\tTotal Loss: {loss}')
                d = {
                    f'{dataset_name}_dis_loss': float(dis_loss),
                    f'{dataset_name}_sample_overlap_loss': float(sample_overlap_loss),
                    f'{dataset_name}_macro_overlap_loss': float(macro_overlap_loss),
                    f'{dataset_name}_overlap_loss': float(overlap_loss),
                    f'{dataset_name}_area_loss': float(area_loss),
                    f'{dataset_name}_hpwl_loss': float(hpwl_loss),
                    # f'{dataset_name}_cong_loss': float(cong_loss),
                    f'{dataset_name}_loss': float(loss),
                }
                ds.append(d)
                evaluate_cell_pos_corner_dict[netlist_name] = layout.cell_pos.cpu().detach().numpy() - layout.cell_size.cpu().detach().numpy()
                del loss
                torch.cuda.empty_cache()

            logs[-1].update(mean_dict(ds))
            return logs[-1][f'{dataset_name}_loss']

        t0 = time()
        if epoch:
            for _ in range(args.train_epoch):
                train(train_netlists)
                scheduler.step()
        logs[-1].update({'train_time': time() - t0})
        t2 = time()
        valid_metric = None
        evaluate(train_netlists, 'train', train_datasets, verbose=False)
        if len(valid_netlists):
            valid_metric = evaluate(valid_netlists, 'valid', valid_datasets)
        if len(test_netlists):
            evaluate(test_netlists, 'test', test_datasets)

        if valid_metric is not None and valid_metric < best_metric:
            best_metric = valid_metric
            if model_dir is not None:
                print(f'\tSaving model to {model_dir}/{args.name}.pkl ...:')
                torch.save(model.state_dict(), f'{model_dir}/{args.name}.pkl')
        for dataset, cell_pos_corner in evaluate_cell_pos_corner_dict.items():
            print(f'\tSaving cell positions to {dataset}/output-{args.name}.npy ...:')
            np.save(f'{dataset}/output-{args.name}.npy', cell_pos_corner)
        evaluate_cell_pos_corner_dict.clear()

        print("\tinference time", time() - t2)
        logs[-1].update({'eval_time': time() - t2})
        if log_dir is not None:
            with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
                json.dump(logs, fp)
