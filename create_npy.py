import argparse
import os
import random

import cv2
import ffmpeg
import h5py
import numpy as np
import pandas as pd
import torch
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import io, transforms
from tqdm import tqdm

random.seed(42)
np.random.seed(42)


class Transform(object):

    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def __call__(self, add_jitter=False, crop_size=224):
        transform = transforms.Compose([
            self.random_crop(crop_size),
        ])
        return transform

    def to_tensor(self):
        return transforms.ToTensor()

    def random_crop(self, size):
        return transforms.RandomCrop(size, pad_if_needed=True)

    def colorJitter(self):
        return transforms.ColorJitter(0.4, 0.2, 0.2, 0.1)


class CustomDataset(Dataset):

    def __init__(self, args, path):
        """Initialize the dataset with splits given by 'subsets', where
        subsets is taken from ['train', 'val', 'test']
        """
        super(CustomDataset, self).__init__()
        self.args = args
        self.path = path
        self.fl_list = self.get_filenames(
            os.path.join(args.video_root, path))
        self.transform = Transform()

    def __len__(self):
        return len(self.fl_list)

    def _get_opencv_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = cap.read()
        frames = [frame]
        while ret:
            ret, frame = cap.read()
            if frame is not None:
                frames.append(frame)
        cap.release()
        frames_array = np.concatenate(np.expand_dims(frames, 0))
        return frames_array

    def get_filenames(self, path):
        results = []
        results += [each for each in os.listdir(path) if each.endswith('.mp4')]
        return results

    def _get_video_torch(self, video_path):
        vframes, _, vmeta = io.read_video(video_path)
        vframes = vframes.permute(0, 3, 1, 2)
        vframes = self.transform(self.args.video_size)(vframes)
        if vframes.shape[0] < self.args.num_frames:
            zeros = th.zeros(
                (3, self.args.num_frames - video.shape[0], self.args.video_size, self.args.video_size), dtype=th.uint8)
            vframes = th.cat((vframes, zeros), axis=0)
        # Gets n_frames from tne entire video, linearly spaced
        vid_indices = np.linspace(
            0, vframes.shape[0] - 1, self.args.num_frames, dtype=int)
        vid = vframes[vid_indices, :].permute(1, 0, 2, 3)
        for i in range(3):
            for j in range(vid.shape[1]):
                if vid[i, j, :, :].sum() == 0:
                    print(i, j)
        return vid

    def _get_video(self, video_path, start=0, end=0):
        '''
        :param video_path: Path of the video file
        start: Start time for the video
        end: End time.
        :return: video: video_frames.
        '''
        # start_seek = random.randint(start, int(max(start, end - self.num_sec)))
        start_seek = 0
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=self.args.fps)
        )
        if self.args.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.args.crop_only:
            '''
            Changes from the original code, because we have few videos that have <224 resolution and needs to be scaled up after cropping, and cropping needs to take care of the size of the image which it did not before.
            cmd = (cmd.crop('(iw - {})*{}'.format(self.args.video_size, aw),
                         '(ih - {})*{}'.format(self.args.video_size, ah),
                         str(self.args.video_size), str(self.args.video_size))
            )'''
            cmd = (
                cmd.crop('max(0, (iw-{}))*{}'.format(self.args.video_size, aw),
                         'max(0, (ih-{}))*{}'.format(self.args.video_size, ah),
                         'min(iw, {})'.format(self.args.video_size),
                         'min(ih, {})'.format(self.args.video_size))
                .filter('scale', self.args.video_size, self.args.video_size)
            )
        else:
            cmd = (
                cmd.crop('(iw - max(0, min(iw,ih)))*{}'.format(aw),
                         '(ih - max(0, min(iw,ih)))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.args.video_size, self.args.video_size)
            )
        if self.args.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )

        video = np.frombuffer(out, np.uint8).reshape(
            [-1, self.args.video_size, self.args.video_size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.args.num_frames:
            zeros = th.zeros(
                (3, self.args.num_frames - video.shape[1], self.args.video_size, self.args.video_size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        # Gets n_frames from tne entire video, linearly spaced
        vid_indices = np.linspace(
            0, video.shape[1]-1, self.args.num_frames, dtype=int)
        return video[:, vid_indices]

    def __getitem__(self, idx):
        video_file = self.fl_list[idx]
        write_file = os.path.join(
            self.args.write_path, video_file.replace(".mp4", ".npy"))
        video_path = os.path.join(
            self.args.video_root, self.path, video_file)
        vid = self._get_video_torch(video_path)
        np.save(write_file, vid)
        return video_file


def main(args):
    dataloader = torch.utils.data.DataLoader(
        CustomDataset(args, args.train_val_path),
        batch_size=1,
        shuffle=False, drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(
        CustomDataset(args, args.test_path),
        batch_size=1,
        shuffle=False, drop_last=True)

    if args.train:
        for i, batch in tqdm(enumerate(dataloader)):
            print("train ", batch)
    if args.test:
        for i, batch in tqdm(enumerate(dataloader_val)):
            print("val ", batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', type=int, default=40,
                        help='num_frame')
    parser.add_argument('--video_root', default='./data/charades/videos')

    parser.add_argument('--write_path', default="./data/charades")
    parser.add_argument('--video_size', type=int, default=224,
                        help='random seed')
    parser.add_argument('--fps', type=int, default=16, help='')
    parser.add_argument('--crop_only', type=int, default=1,
                        help='random seed')
    parser.add_argument('--center_crop', type=int, default=0,
                        help='random seed')
    parser.add_argument('--random_flip', type=int, default=0,
                        help='random seed')
    parser.add_argument('--train', default=1)
    parser.add_argument('--test', default=1)
    args = parser.parse_args()
    args.train_val_path = "train_val"
    args.test_path = "test"
    args.write_path += "/num_frames_{}".format(args.num_frames)
    os.makedirs(args.write_path, exist_ok=True)
    main(args)
