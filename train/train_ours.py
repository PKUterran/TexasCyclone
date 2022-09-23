import argparse
import numpy as np
import torch
import json
from typing import List, Dict, Any, Tuple
from functools import reduce
from time import time
from tqdm import tqdm

from data.Netlist import Netlist, netlist_from_numpy_directory, netlist_from_numpy_directory_old
from data.Layout import Layout, layout_from_netlist_dis_angle, layout_from_directory
from data.utils import set_seed, mean_dict
from train.model import NaiveGNN
from train.functions import AreaLoss, HPWLLoss, SampleOverlapLoss, SampleNetOverlapLoss


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
    print(f'Loading data...')
    train_netlists = [netlist_from_numpy_directory(dataset) for dataset in train_datasets]
    valid_netlists = [netlist_from_numpy_directory(dataset) for dataset in valid_datasets]
    test_netlists = [netlist_from_numpy_directory(dataset) for dataset in test_datasets]

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
    }

    if args.gnn == 'naive':
        model = NaiveGNN(raw_cell_feats, raw_net_feats, raw_pin_feats, config)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))

    # Train model
    best_metric = 1e8  # lower is better
    evaluate_cell_pos_dict = {}
    overlap_loss_op = SampleOverlapLoss(span=4)
    # area_loss_op = AreaLoss()
    hpwl_loss_op = HPWLLoss()
    cong_loss_op = SampleNetOverlapLoss(device, span=4)

    for epoch in range(0, args.epochs + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})

        def forward(netlist: Netlist) -> Tuple[Layout, torch.Tensor]:
            net_dis, net_angle, pin_dis, pin_angle = model.forward(netlist)
            layout, dis_loss = layout_from_netlist_dis_angle(netlist, net_dis, net_angle, pin_dis, pin_angle)
            return layout, dis_loss

        def train(netlists: List[Netlist]):
            model.train()
            t1 = time()
            losses = []
            n_netlist = len(netlists)
            iter_i_netlist = tqdm(enumerate(netlists), total=n_netlist) \
                if use_tqdm else enumerate(netlists)
            for j, netlist in iter_i_netlist:
                layout, dis_loss = forward(netlist)
                overlap_loss = overlap_loss_op.forward(layout)
                # area_loss = area_loss_op.forward(layout, limit=)
                hpwl_loss = hpwl_loss_op.forward(layout)
                cong_loss = cong_loss_op.forward(layout)
                loss = sum((
                    args.dis_lambda * dis_loss,
                    args.overlap_lambda * overlap_loss,
                    # args.area_lambda * area_loss,
                    args.hpwl_lambda * hpwl_loss,
                    args.cong_lambda * cong_loss,
                ))
                losses.append(loss)
                if len(losses) >= args.batch or j == n_netlist - 1:
                    sum(losses).backward()
                    optimizer.step()
                    losses.clear()
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
                layout, dis_loss = forward(netlist)
                overlap_loss = overlap_loss_op.forward(layout)
                # area_loss = area_loss_op.forward(layout, limit=)
                hpwl_loss = hpwl_loss_op.forward(layout)
                cong_loss = cong_loss_op.forward(layout)
                loss = sum((
                    args.dis_lambda * dis_loss,
                    args.overlap_lambda * overlap_loss,
                    # args.area_lambda * area_loss,
                    args.hpwl_lambda * hpwl_loss,
                    args.cong_lambda * cong_loss,
                ))
                print(f'\t\tDiscrepancy Loss: {dis_loss.data}')
                print(f'\t\tOverlap Loss: {overlap_loss.data}')
                # print(f'\t\tArea Loss: {area_loss.data}')
                print(f'\t\tHPWL Loss: {hpwl_loss.data}')
                print(f'\t\tCongestion Loss: {cong_loss.data}')
                print(f'\t\tTotal Loss: {loss.data}')
                d = {
                    f'{dataset_name}_dis_loss': float(dis_loss.data),
                    f'{dataset_name}_overlap_loss': float(overlap_loss.data),
                    # f'{dataset_name}_area_loss': float(area_loss.data),
                    f'{dataset_name}_hpwl_loss': float(hpwl_loss.data),
                    f'{dataset_name}_cong_loss': float(cong_loss.data),
                    f'{dataset_name}_loss': float(loss.data),
                }
                ds.append(d)
                evaluate_cell_pos_dict[netlist_name] = layout.cell_pos.cpu().detach().numpy()

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
            for dataset, cell_pos in evaluate_cell_pos_dict.items():
                print(f'\tSaving cell positions to {dataset}/output-{args.name}.npy ...:')
                np.save(f'{dataset}/output-{args.name}.npy', cell_pos)
            evaluate_cell_pos_dict.clear()
            if model_dir is not None:
                print(f'\tSaving model to {model_dir}/{args.name}.pkl ...:')
                torch.save(model.state_dict(), f'{model_dir}/{args.name}.pkl')

        print("\tinference time", time() - t2)
        logs[-1].update({'eval_time': time() - t2})
        if log_dir is not None:
            with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
                json.dump(logs, fp)
