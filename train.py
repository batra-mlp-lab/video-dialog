import argparse
import datetime
import gc
import math
import os
import random

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from args import get_args
from dataloader import VisDialDataset
from decoders import Decoder
from encoders import Encoder, LateFusionEncoder
from models import AVSD
from utils import get_gt_ranks, process_ranks, scores_to_ranks, visualize

parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)
LateFusionEncoder.add_cmdline_args(parser)

args = get_args(parser)
args.numpy_path += "/num_frames_{}".format(args.num_frames)
start_time = datetime.datetime.strftime(
    datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')

if args.save_path == 'checkpoints/':
    # args.save_path += start_time
    args.save_path += 'input_type_{0}_s3d_mixed_5c_fps_{1}_num_frames_{2}_text_encoder_{3}_lr_{4}_unfreeze_layer_{5}_finetune_{6}_use_npy_{7}_batch_size_{8}_finetuneBert_{9}'.format(
        args.input_type, args.fps, args.num_frames, args.text_encoder, args.lr, args.unfreeze_layers, args.finetune, args.use_npy, args.batch_size, args.finetune_textEncoder)
# -------------------------------------------------------------------------------------
# setting visdom args
# -------------------------------------------------------------------------------------
viz = visualize.VisdomLinePlot(
    env_name=args.visEnvName,
    server=args.server,
    port=args.serverPort)

# seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

if args.set_cuda_device is not '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.set_cuda_device

# set device and default tensor type
device = "cpu"
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    args.num_gpu = torch.cuda.device_count()
    device = "cuda"

# transfer all options to model
model_args = args

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------

if args.load_path != '':
    components = torch.load(args.load_path)
    model_args = components['model_args']
    model_args.gpuid = args.gpuid
    model_args.batch_size = args.batch_size

    # this is required by dataloader
    args.img_norm = model_args.img_norm

for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

viz.writeText(args)
# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(args, ['train'])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        drop_last=True,
                        collate_fn=dataset.collate_fn)

dataset_val = VisDialDataset(args, ['val'])
dataloader_val = DataLoader(dataset_val,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=dataset.collate_fn)

dataset_test = VisDialDataset(args, ['test'])
dataloader_test = DataLoader(dataset_test,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=dataset.collate_fn)
# ----------------------------------------------------------------------------
# setting model args
# ----------------------------------------------------------------------------

# transfer some useful args from dataloader to model
for key in {'num_data_points', 'vocab_size', 'max_ques_count',
            'max_ques_len', 'max_ans_len'}:
    setattr(model_args, key, getattr(dataset, key))

# iterations per epoch
setattr(args, 'iter_per_epoch', math.ceil(
    dataset.num_data_points['train'] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

model = AVSD(model_args)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of model params {0}".format(total_params))
if args.finetune:
    total_params = sum(p.numel()
                       for p in model.encoder.video_embed.parameters() if p.requires_grad)
    print("Total number of s3dg params {0}".format(total_params))
optimizer = optim.Adam(list(model.parameters()),
                       lr=args.lr, weight_decay=args.weight_decay)

scheduler = lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=args.lr_decay_rate)

if args.load_path != '':
    model._load_state_dict_(components)
    print("Loaded model from {}".format(args.load_path))
print("Encoder: {}".format(args.encoder))
print("Decoder: {}".format(args.decoder))

if args.gpuid >= 0:
    model = torch.nn.DataParallel(model, output_device=0, dim=0)
    model = model.to(device)

# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

model.train()
os.makedirs(args.save_path, exist_ok=True)
with open(os.path.join(args.save_path, "args_{0}.txt".format(start_time)), "w") as f:
    f.write(str(args))
f.close()

running_loss = 0.0
train_begin = datetime.datetime.utcnow()
print("Training start time: {}".format(
    datetime.datetime.strftime(train_begin, '%d-%b-%Y-%H:%M:%S')))


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


log_loss = []
for epoch in range(1, model_args.num_epochs + 1):
    for i, batch in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        model.train()
        model.zero_grad()
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = Variable(batch[key])
                if args.gpuid >= 0:
                    batch[key] = batch[key].cuda()

        # --------------------------------------------------------------------
        # forward-backward pass and optimizer step
        # --------------------------------------------------------------------
        # if not batch["vid_feat"].shape[0] % args.num_gpu == 0:
        #     num_repeat = args.num_gpu - batch["vid_feat"].shape[0] % args.num_gpu
        #     batch = repeat_tensors(batch, num_repeat)
        new_batch = convert_list_to_tensor(batch)
        _, cur_loss = model(new_batch)
        cur_loss = cur_loss.mean()
        cur_loss.backward()

        optimizer.step()
        gc.collect()

        # --------------------------------------------------------------------
        # update running loss and decay learning rates
        # --------------------------------------------------------------------
        train_loss = cur_loss.item()
        #import pdb
        # pdb.set_trace()

        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * cur_loss.item()
        else:
            running_loss = cur_loss.item()

        if optimizer.param_groups[0]['lr'] > args.min_lr:
            scheduler.step()

        # --------------------------------------------------------------------
        # print after every few iterations

        if (i + 1) % args.eval_step == 0:
            print("Running validation")
            validation_losses = []
            model.eval()
            model.zero_grad()
            for _, val_batch in enumerate(dataloader_val):
                for key in val_batch:
                    if not isinstance(val_batch[key], list):
                        val_batch[key] = Variable(val_batch[key])
                        if args.gpuid >= 0:
                            val_batch[key] = val_batch[key].cuda()

                # if not val_batch["vid_feat"].shape[0] % args.num_gpu == 0:
                #     num_repeat = args.num_gpu - val_batch["vid_feat"].shape[0] % args.num_gpu
                #     val_batch = repeat_tensors(val_batch, num_repeat)
                # print(val_batch["img_fnames"])
                new_batch_v = convert_list_to_tensor(val_batch)
                _, cur_loss = model(new_batch_v)
                cur_loss = cur_loss.mean()
                validation_losses.append(cur_loss.item())

            validation_loss = np.mean(validation_losses)
            iteration = (epoch - 1) * args.iter_per_epoch + i

            log_loss.append((epoch,
                             iteration,
                             running_loss,
                             train_loss,
                             validation_loss,
                             optimizer.param_groups[0]['lr']))

            # print current time, running average, learning rate, iteration, epoch
            print("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][val loss: {:6f}][lr: {:7f}]".format(
                datetime.datetime.utcnow() - train_begin, epoch,
                iteration, running_loss, validation_loss,
                optimizer.param_groups[0]['lr']))

            viz.plotLine('Loss', 'Train', 'LOSS', iteration, train_loss)
            viz.plotLine('Loss', 'Val', 'LOSS', iteration, validation_loss)
    # ------------------------------------------------------------------------
    # save checkpoints and final model
    # ------------------------------------------------------------------------
    if epoch % args.save_step == 0:
        torch.save({
            'encoder': model.module.encoder.state_dict(),
            'decoder': model.module.decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model.module.args
        }, os.path.join(args.save_path, 'model_epoch_{}.pth'.format(epoch)))
        print('Running evaluation for checkpoint:', epoch)
        model.eval()
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
        process_ranks(all_ranks, args.save_path, epoch)

        f.close()
        gc.collect()
        model.train()

torch.save({
    'encoder': model.module.encoder.state_dict(),
    'decoder': model.module.decoder.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model.module.args
}, os.path.join(args.save_path, 'model_final.pth'))

np.save(os.path.join(args.save_path, 'log_loss'), log_loss)
