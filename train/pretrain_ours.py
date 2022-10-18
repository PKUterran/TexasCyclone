import argparse
import torch
import torch.nn.functional as F
import json
from time import time
from typing import List, Dict, Any, Tuple
from functools import reduce
from time import time
from tqdm import tqdm

from data.graph import Netlist, expand_netlist
from data.pretrain import DIS_ANGLE_TYPE, load_pretrain_data
from data.utils import set_seed, mean_dict
from train.model import NaiveGNN


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

    def compact_train_netlist_dis_angle(list_netlist_dis_angle: List[Tuple[Netlist, DIS_ANGLE_TYPE]]
                                       ) -> list_netlist_dis_angle: List[Tuple[Netlist, DIS_ANGLE_TYPE]]:
            pass
    
    train_list_netlist_dis_angle = unpack_netlist_dis_angle(
        [load_pretrain_data(dataset) for dataset in train_datasets])
    valid_list_netlist_dis_angle = unpack_netlist_dis_angle(
        [load_pretrain_data(dataset) for dataset in valid_datasets])
    test_list_netlist_dis_angle = unpack_netlist_dis_angle(
        [load_pretrain_data(dataset) for dataset in test_datasets])
    compacted_train_list_netlist_dis_angle = compact_train_netlist_dis_angle(train_list_netlist_dis_angle)
    print(f'\t# of samples: '
          f'{len(compacted_train_list_netlist_dis_angle)} compacted train, '
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

    for epoch in range(0, args.epochs + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})

        def forward(netlist: Netlist) -> DIS_ANGLE_TYPE:
            edge_dis, edge_angle = model.forward(netlist)
            return edge_dis, edge_angle

        def train(list_netlist_dis_angle: List[Tuple[Netlist, DIS_ANGLE_TYPE]]):
            model.train()
            t1 = time()
            losses = []
            n_netlist = len(list_netlist_dis_angle)
            iter_i_netlist_dis_angle = tqdm(enumerate(list_netlist_dis_angle), total=n_netlist) \
                if use_tqdm else enumerate(list_netlist_dis_angle)
            for j, (netlist, dis_angle) in iter_i_netlist_dis_angle:
                edge_dis, edge_angle = forward(netlist)
                edge_dis_loss = F.mse_loss(edge_dis, dis_angle[0]) ** 0.5
                edge_angle_loss = F.mse_loss(edge_angle, dis_angle[1]) ** 0.5
                loss = sum((
                    edge_dis_loss * 0.001,
                    edge_angle_loss * 0.1,
                ))
                losses.append(loss)
                if len(losses) >= args.batch or j == n_netlist - 1:
                    sum(losses).backward()
                    optimizer.step()
                    losses.clear()
            print(f"\tTraining time per epoch: {time() - t1}")

        def evaluate(list_netlist_dis_angle: List[Tuple[Netlist, DIS_ANGLE_TYPE]],
                     dataset_name: str, netlist_names: List[str], verbose=True) -> float:
            model.eval()
            ds = []
            print(f'\tEvaluate {dataset_name}:')
            for netlist_name in netlist_names:
                print(f'\t\t{netlist_name}:')
            n_netlist = len(list_netlist_dis_angle)
            iter_i_netlist_dis_angle = tqdm(list_netlist_dis_angle, total=n_netlist) \
                if use_tqdm else list_netlist_dis_angle
            flag = False
            for netlist, dis_angle in iter_i_netlist_dis_angle:
                t0 = time()
                edge_dis, edge_angle = forward(netlist)
                print(time() - t0)
                edge_dis_loss = F.mse_loss(edge_dis, dis_angle[0]) ** 0.5
                edge_angle_loss = F.mse_loss(edge_angle, dis_angle[1]) ** 0.5
                loss = sum((
                    edge_dis_loss * 0.001,
                    edge_angle_loss * 0.1,
                ))
                d = {
                    f'{dataset_name}_net_dis_loss': float(edge_dis_loss.data),
                    f'{dataset_name}_net_angle_loss': float(edge_angle_loss.data),
                    f'{dataset_name}_loss': float(loss.data),
                }
                print(time() - t0)
                ds.append(d)
                if not flag:
                    flag = True
                else:
                    exit(1234)

            d_t = mean_dict(ds)
            logs[-1].update(d_t)
            print(f'\t\tEdge Distance Loss: {d_t[f"{dataset_name}_net_dis_loss"]}')
            print(f'\t\tEdge Angle Loss: {d_t[f"{dataset_name}_net_angle_loss"]}')
            print(f'\t\tTotal Loss: {d_t[f"{dataset_name}_loss"]}')
            return logs[-1][f'{dataset_name}_loss']

        t0 = time()
        if epoch:
            for _ in range(args.train_epoch):
                train(compacted_train_list_netlist_dis_angle)
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
