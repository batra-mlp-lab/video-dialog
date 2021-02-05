import argparse
import datetime
import gc
import json
import logging
import math
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from args import get_args
from dataloader import VisDialDataset
from decoders import Decoder
from encoders import Encoder, LateFusionEncoder
from models import AVSD
from utils import get_gt_ranks, process_ranks, scores_to_ranks, visualize
import pprint


parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)
LateFusionEncoder.add_cmdline_args(parser)

args = get_args(parser)

viz = visualize.VisdomVisualize(
    env_name=args.visEnvName,
    server=args.server,
    port=args.serverPort)

# seed for reproducibility
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
model_args = args
'''
log_path = os.path.join(args.load_path, 'eval_results.log')
logging.basicConfig(filename='eval_results.log')
'''

cur = os.getcwd()
os.chdir(args.load_path)
checkpoints = sorted(
    filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
checkpoints = [file for file in checkpoints if file.endswith(".pth")]
logging.info("Evaluate the following checkpoints: %s", args.load_path)
os.chdir(cur)

# set device and default tensor type
device = "cpu"
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    args.num_gpu = torch.cuda.device_count()
    device = "cuda"

# set this because only late fusion encoder is supported yet
args.concat_history = True

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------
dataset = VisDialDataset(args, [args.split])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=dataset.collate_fn)

# iterations per epoch
setattr(args, 'iter_per_epoch', math.ceil(
    dataset.num_data_points[args.split] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------
'''
model = AVSD(model_args)
model._load_state_dict_(components)
print("Loaded model from {}".format(args.load_path))

if args.gpuid >= 0:
    model = torch.nn.DataParallel(model, output_device=0, dim=0)
    model = model.to(device)
'''
# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

def convert_list_to_tensor(batch):
    new_batch = {}
    for k, v in batch.items():
        # tensor of list of strings isn't possible, hence removing the image fnames from the batch sent into the training module.
        if isinstance(v, list) and not (k == "img_fnames"):
            new_batch[k] = torch.Tensor(v)
        elif isinstance(v, torch.Tensor):
            new_batch[k] = v
    return new_batch


def repeat_tensors(batch, num_repeat):
    """In the last iterations, when the number of samples are not multiple of the num_gpu, this function will repeat the last few samples"""
    new_batch = batch.copy()
    for i in range(num_repeat):
        for k, v in batch.items():
            if isinstance(v, list):
                new_batch[k].append(v[-1])
            elif isinstance(v, torch.Tensor):
                new_batch[k] = torch.cat((new_batch[k], v[-1].unsqueeze(0)), 0)
    return new_batch

'''
if args.use_gt:
    # ------------------------------------------------------------------------
    # calculate automatic metrics and finish
    # ------------------------------------------------------------------------
    all_ranks = []
    for i, batch in tqdm(enumerate(tqdm(dataloader))):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = Variable(batch[key], volatile=True)
                if args.gpuid >= 0:
                    batch[key] = batch[key].cuda()

        if not batch["vid_feat"].shape[0] % args.num_gpu == 0:
            num_repeat = args.num_gpu - \
                batch["vid_feat"].shape[0] % args.num_gpu
            batch = repeat_tensors(batch, num_repeat)
        new_batch = convert_list_to_tensor(batch)
        dec_out, _ = model(new_batch)
        ranks = scores_to_ranks(dec_out.data)
        gt_ranks = get_gt_ranks(ranks, batch['ans_ind'].data)
        all_ranks.append(gt_ranks)
    all_ranks = torch.cat(all_ranks, 0)
    process_ranks(all_ranks)
    gc.collect()
else:
    # ------------------------------------------------------------------------
    # prepare json for submission
    # ------------------------------------------------------------------------
    ranks_json = []
    for i, batch in tqdm(enumerate(tqdm(dataloader))):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = Variable(batch[key], volatile=True)
                if args.gpuid >= 0:
                    batch[key] = batch[key].cuda()

        if not batch["vid_feat"].shape[0] % args.num_gpu == 0:
            num_repeat = args.num_gpu - \
                batch["vid_feat"].shape[0] % args.num_gpu
            batch = repeat_tensors(batch, num_repeat)
        new_batch = convert_list_to_tensor(batch)
        dec_out, _ = model(new_batch)
        ranks = scores_to_ranks(dec_out.data)
        ranks = ranks.view(-1, 10, 100)

        for i in range(len(batch['img_fnames'])):
            # cast into types explicitly to ensure no errors in schema
            if args.split == 'test':
                ranks_json.append({
                    'image_id': int(batch['img_fnames'][i][-16:-4]),
                    'round_id': int(batch['num_rounds'][i]),
                    'ranks': list(ranks[i][batch['num_rounds'][i] - 1])
                })
            else:
                for j in range(batch['num_rounds'][i]):

                    # read saved model and args
                    # ----------------------------------------------------------------------------
'''

print("Evaluation start time: {}".format(
    datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')))

i=0

for checkpoint in checkpoints:

    print('checkpoint:', checkpoint)
    model_path = os.path.join(args.load_path, checkpoint)
    components = torch.load(model_path)
    model_args = components['model_args']
    if i == 0:
        viz.showText(pprint.pformat(args, indent=4))
        i +=1
    model = AVSD(model_args)
    model._load_state_dict_(components)
    model_args.gpuid = args.gpuid
    model_args.batch_size = args.batch_size


    for arg in vars(args):
        print('{:<20}: {}'.format(arg, getattr(args, arg)))

    # ----------------------------------------------------------------------------
    # setup the model
    # ----------------------------------------------------------------------------


    print("Loaded model from {}".format(args.load_path))

    if args.gpuid >= 0:
        model = torch.nn.DataParallel(model, output_device=0, dim=0)
        model = model.to(device)

    # ----------------------------------------------------------------------------
    # evaluation
    # ----------------------------------------------------------------------------

    print("Evaluation start time: {}".format(
        datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')))
    model.eval()

    if args.use_gt:
        viz.save()
        # ------------------------------------------------------------------------
        # calculate automatic metrics and finish
        # ------------------------------------------------------------------------
        all_ranks = []
        for i, batch in tqdm(enumerate(tqdm(dataloader))):
            for key in batch:
                if not isinstance(batch[key], list):
                    batch[key] = Variable(batch[key], volatile=True)
                    if args.gpuid >= 0:
                        batch[key] = batch[key].cuda()

            new_batch = convert_list_to_tensor(batch)
            dec_out, _ = model(new_batch)
            ranks = scores_to_ranks(dec_out.data)
            gt_ranks = get_gt_ranks(ranks, batch['ans_ind'].data)
            all_ranks.append(gt_ranks)
        all_ranks = torch.cat(all_ranks, 0)

        all_metrics = process_ranks(all_ranks, args.load_path, checkpoint[6:-4])
        iter_id = checkpoint[6:-4]
        for metric_name, metric_value in all_metrics.items():
            print(f"{metric_name}: {metric_value}")
            if 'round' in metric_name:
                viz.plotLine(iter_id, metric_value, 'Retrieval Round Val Metrics Round -' + metric_name.split('_')[-1],
                             metric_name)
            else:
                viz.plotLine(iter_id.split('_')[1], metric_value, 'Retrieval Val Metrics', metric_name)
        gc.collect()
    else:
        # ------------------------------------------------------------------------
        # prepare json for submission
        # ------------------------------------------------------------------------
        ranks_json = []
        for i, batch in tqdm(enumerate(tqdm(dataloader))):
            for key in batch:
                if not isinstance(batch[key], list):
                    batch[key] = Variable(batch[key], volatile=True)
                    if args.gpuid >= 0:
                        batch[key] = batch[key].cuda()


            new_batch = convert_list_to_tensor(batch)
            dec_out, _ = model(new_batch)
            ranks = scores_to_ranks(dec_out.data)
            ranks = ranks.view(-1, 10, 100)

            for i in range(len(batch['img_fnames'])):
                # cast into types explicitly to ensure no errors in schema
                if args.split == 'test':
                    ranks_json.append({
                        'image_id': int(batch['img_fnames'][i][-16:-4]),
                        'round_id': int(batch['num_rounds'][i]),
                        'ranks': list(ranks[i][batch['num_rounds'][i] - 1])
                    })
                else:
                    for j in range(batch['num_rounds'][i]):
                        ranks_json.append({
                            'image_id': int(batch['img_fnames'][i][-16:-4]),
                            'round_id': int(j + 1),
                            'ranks': list(ranks[i][j])
                        })
            gc.collect()

    if args.save_ranks:
        print("Writing ranks to {}".format(args.save_path))
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        json.dump(ranks_json, open(args.save_path, 'w'))


