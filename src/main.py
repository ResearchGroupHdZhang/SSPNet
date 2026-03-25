import os
import pickle
import random
from collections import defaultdict

import torch
import numpy as np
import dill
import argparse
from torch.optim import Adam
from modules import SSPNetModel

from training import Test, Train


def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)


def get_model_name(args):
    model_name = [
        f'dim_{args.dim}',  f'lr_{args.lr}', f'coef_{args.coef}',
        f'dp_{args.dp}', f'ddi_{args.target_ddi}'
    ]
    if args.embedding:
        model_name.append('embedding')
    return '-'.join(model_name)


def parse_args():
    parser = argparse.ArgumentParser('Experiment For DrugRec')

    parser.add_argument('--Test', action='store_true', help="evaluating mode")
    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    # parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument('--dataset', default='MIMIC_3', type=str, help='choose dataset')
    parser.add_argument(
        '--model_name', type=str,
        help="the model name for training, if it's left blank,"
        " the code will generate a new model name on it own"
    )
    parser.add_argument(
        '--resume_path', type=str,
        help='path of well trained model, only for evaluating the model'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='gpu id to run on, negative for cpu'
    )
    parser.add_argument(
        '--target_ddi', type=float, default=0.06,
        help='expected ddi for training'
    )
    parser.add_argument(
        '--coef', default=2.5, type=float,
        help='coefficient for DDI Loss Weight Annealing'
    )
    parser.add_argument(
        '--embedding', action='store_true',
        help='use embedding table for substructures' +
        'if it\'s not chosen, the substructure will be encoded by GNN'
    )
    parser.add_argument(
        '--epochs', default=200, type=int,
        help='the epochs for training'
    )

    args = parser.parse_args()
    if args.Test and args.resume_path is None:
        raise FileNotFoundError('Can\'t Load Model Weight From Empty Dir')
    if args.model_name is None:
        args.model_name = get_model_name(args)

    return args


def dataset_ddi(data, ddi_A):
    data_iterator = iter(data)
    all_cnt = 0
    dd_cnt = 0
    for patient in data_iterator:
        for adm in patient:
            med_code_set = adm[2]
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    else:
       return dd_cnt / all_cnt

if __name__ == '__main__':
    set_seed()
    args = parse_args()
    print(args)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    data_path = '../data/' + args.dataset + '/records_final.pkl'
    voc_path = '../data/' + args.dataset + '/voc_final.pkl'
    ddi_adj_path = '../data/' + args.dataset + '/ddi_A_final.pkl'
    ddi_mask_path = '../data/' + args.dataset + '/ddi_mask_H.pkl'
    ehr_adj_path = '../data/' + args.dataset + '/ehr_adj_final.pkl'
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)

    diag_voc, pro_voc, med_voc = \
        voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    model = SSPNetModel(
        emb_dim=args.dim,
        ehr_adj=ehr_adj, ddi_adj=ddi_adj,
        voc_size=voc_size,
        use_embedding=args.embedding, device=device, dropout=args.dp
    ).to(device)

    med = [i for i in range(voc_size[2])]
    sorted_meds = torch.tensor(med).to(device)

    if args.Test:
        Test(model, args.resume_path, device, data_test, sorted_meds, voc_size, ddi_adj, dataset_path=args.dataset)
    else:
        if not os.path.exists(os.path.join('../saved', args.model_name)):
            os.makedirs(os.path.join('../saved', args.model_name))
        log_dir = os.path.join('../saved', args.model_name)
        optimizer = Adam(model.parameters(), lr=args.lr)
        Train(
            model, device, data_train, data_eval, data_test, sorted_meds, voc_size, ddi_adj,
            optimizer, log_dir, args.coef, args.target_ddi, EPOCH=args.epochs, dataset_path=args.dataset
        )




