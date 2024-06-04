import os
from collections import Counter
import argparse
import random
import sys
import itertools
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from datasets import datasets
import hparams_registry
from lib import misc
from CSI.csi_domainset import get_domains
from pathlib import Path
from lib.utils import write_result_to_txt
from lib.logger import Logger
from trainer import train

def get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str,default="E:\wifi\WiSR")#"/mnt/ssd1/LiuSJ/")#E:\wifi\WiSR"
    parser.add_argument('--dataset', type=str, default='CSI')
    parser.add_argument('--csidataset', type=str, default='Widar3')#'Widar3'#'CSIDA',#'ARIL'
    parser.add_argument('--algorithm', type=str, default="INDG")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--checkpoint_freq', type=int, default=1 )
    parser.add_argument('--output_dir', type=str, default="./train_output/")
    parser.add_argument('--results_file', type=str, default="test_results.txt")
    parser.add_argument("--evalmode",default="fast",help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--model_save", default=300, type=int, help="Model save start step")

    parser.add_argument('--data_type', type=str, default="amp+pha")
    parser.add_argument('--source_domains', type=str, default=None)
    parser.add_argument('--target_domains', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--backbone', type=str, default="ChannelAttentionNet")
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--ica', action='store_true')
    parser.add_argument('--exp_name', type=str, default="")

    args = parser.parse_known_args()

    return args

def main(args,left_argv):

    if sys.platform == "linux":
        username = os.getlogin()
        if username =="ahnu":
            args.data_dir="/media/ahnu/ssk_data/wifi/WiSR"
            args.batch_size=128
            args.steps=200

        args.output_dir=f"train_output_ssh_{username}"

    os.makedirs(args.output_dir, exist_ok=True)

    # miro path setup
    args.out_dir = Path(args.output_dir )
    args.out_dir.mkdir(exist_ok=True, parents=True)
    os.makedirs(args.out_dir / "logs" ,exist_ok=True)
    logger = Logger.get(args.out_dir / "logs" / f"log_{args.domain_type}.txt")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))



    # setup hparams
    #miro
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    hparams['max_epoch']=args.steps
    hparams['batch_size'] = args.batch_size

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.output_dir}"

    assert torch.cuda.is_available(), "CUDA is not available"
    
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset=vars(datasets)[args.dataset](args)

    # print dataset information
    logger.nofmt("Dataset:")
    logger.nofmt(f"\t[{args.dataset}] #envs={args.source_domains}, #classes={dataset.num_classes}")
    logger.info(f"Target test envs = {args.target_domains}")
    logger.info(f"Output_dir = {args.output_dir}")

    n_steps = args.steps
    checkpoint_freq = args.checkpoint_freq
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")
    
    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    results=train(
        args.target_domains,
        args=args,
        hparams=hparams,
        n_steps=n_steps,
        checkpoint_freq=checkpoint_freq,
        logger=logger,
    )
    write_result_to_txt(args,results)



if __name__ == "__main__":
    domain_type='user'
    '''
    {
        'Widar3':['loc',
                'user',
                'room',
                'ori',
                ]
    '''
    ibegin=0
    imax=500
    args,left_argv=get_args()

    if domain_type=="user":
        rx_pairs = list(itertools.combinations([1], 1))  # [1,2,3,4,5,6]
        loc_ids = ['1', '2', '3', '4', '5']
        for rx_pair in rx_pairs:
            for loc in loc_ids:
                ori_ids=['1-3']
                for ori in ori_ids:
                    if len(rx_pair)==2:
                        args.rxs=f"{rx_pair[0]}+{rx_pair[1]}"
                    else:
                        args.rxs=str(rx_pair[0])
                    oris=[ori]
                    locs=[loc]
                    dataset_domain_list=get_domains(args.csidataset,domain_type,ibegin,imax,rxs=[args.rxs],oris=oris,loc_ids=locs)
                    for i in range(len(dataset_domain_list[args.csidataset])):
                        args.source_domains=dataset_domain_list[args.csidataset][i]['source_domains']
                        args.target_domains=dataset_domain_list[args.csidataset][i]['target_domains']
                        ###
                        args.domain_type=domain_type
                        args.oris="-".join(oris)
                        args.locs="-".join(locs)
                        main(args,left_argv)


    if domain_type == "ori":
        loc_ids=['1']
        rx_pairs = list(itertools.combinations([1], 1))
        for rx_pair in rx_pairs:
            for loc in loc_ids:
                    if len(rx_pair) == 2:
                        args.rxs = f"{rx_pair[0]}+{rx_pair[1]}"
                    else:
                        args.rxs = str(rx_pair[0])
                    # oris = [ori]
                    locs = [loc]
                    dataset_domain_list = get_domains(args.csidataset, domain_type, ibegin, imax, rxs=[args.rxs],
                                                       loc_ids=locs)
                    for i in range(len(dataset_domain_list[args.csidataset])):
                        args.source_domains = dataset_domain_list[args.csidataset][i]['source_domains']
                        args.target_domains = dataset_domain_list[args.csidataset][i]['target_domains']
                        ###
                        args.domain_type = domain_type
                        # args.oris = "-".join(oris)
                        args.locs = "-".join(locs)
                        main(args, left_argv)

    if domain_type == "loc":
        rx_pairs = list(itertools.combinations([1], 1))
        ori_ids=['1']
        for rx_pair in rx_pairs:
                for ori in ori_ids:
                    if len(rx_pair) == 2:
                        args.rxs = f"{rx_pair[0]}+{rx_pair[1]}"
                    else:
                        args.rxs = str(rx_pair[0])
                    oris = [ori]
                    dataset_domain_list = get_domains(args.csidataset, domain_type, ibegin, imax, rxs=[args.rxs],
                                                      oris=oris)
                    for i in range(len(dataset_domain_list[args.csidataset])):
                        args.source_domains = dataset_domain_list[args.csidataset][i]['source_domains']
                        args.target_domains = dataset_domain_list[args.csidataset][i]['target_domains']
                        ###
                        args.domain_type = domain_type
                        args.oris = "-".join(oris)
                        main(args, left_argv)
