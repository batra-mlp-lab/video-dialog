import json
import os
import pdb
import random
from random import shuffle
from transformers import BertTokenizer

import ffmpeg
import h5py
import hdfdict
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from six import iteritems
from torch.utils.data import Dataset
from torchvision import io, transforms
from tqdm import tqdm


class Transform(object):

    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def __call__(self, add_jitter=False, crop_size=224):
        transform = transforms.Compose([
            self.random_crop(crop_size)
        ])
        return transform

    def to_tensor(self):
        return transforms.ToTensor()

    def random_crop(self, size):
        return transforms.RandomCrop(size, pad_if_needed=True)

    def colorJitter(self):
        return transforms.ColorJitter(0.4, 0.2, 0.2, 0.1)


class VisDialDataset(Dataset):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Dataloader specific arguments')
        parser.add_argument(
            '-input_img', default='data/data_img.h5', help='HDF5 file with image features')
        # parser.add_argument(
        #     '-input_vid', default='data/data_video.h5', help='HDF5 file with video features')
        parser.add_argument(
            '-input_audio', default='data/data_audio.h5', help='HDF5 file with audio features')
        parser.add_argument('-input_ques', default='data/dialogs.h5',
                            help='HDF5 file with preprocessed questions')
        parser.add_argument('-input_json', default='data/params.json',
                            help='JSON file with image paths and vocab')
        parser.add_argument(
            '-img_norm', default=1, choices=[1, 0], help='normalize the image feature. 1=yes, 0=no')
        return parser

    def __init__(self, args, subsets):
        """Initialize the dataset with splits given by 'subsets', where
        subsets is taken from ['train', 'val', 'test']
        """
        super(VisDialDataset, self).__init__()
        self.args = args
        self.subsets = tuple(subsets)
        self.transform = Transform()

        print("Dataloader loading json file: {}".format(args.input_json))
        with open(args.input_json, 'r') as info_file:
            info = json.load(info_file)
            # possible keys: {'ind2word', 'word2ind', 'unique_img_(split)'}
            for key, value in iteritems(info):
                setattr(self, key, value)

        # add <START> and <END> to vocabulary
        word_count = len(self.word2ind)
        self.word2ind['<START>'] = word_count + 1
        self.word2ind['<END>'] = word_count + 2
        self.start_token = self.word2ind['<START>']
        self.end_token = self.word2ind['<END>']

        # padding + <START> + <END> token
        self.vocab_size = word_count + 3
        print("Vocab size with <START>, <END>: {}".format(self.vocab_size))

        # construct reverse of word2ind after adding tokens
        self.ind2word = {
            int(ind): word
            for word, ind in iteritems(self.word2ind)
        }

        print("Dataloader loading h5 file: {}".format(args.input_ques))
        ques_file = h5py.File(args.input_ques, 'r')

        if 'I' in args.input_type:
            print("Dataloader loading h5 file: {}".format(args.input_img))
            img_file = h5py.File(args.input_img, 'r')

        if 'V' in args.input_type:
            print("Dataloader loading h5 file: {}".format(args.input_vid))
            vid_file = args.input_vid

        if 'A' in args.input_type:
            print("Dataloader loading h5 file: {}".format(args.input_audio))
            audio_file = h5py.File(args.input_audio, 'r')

        # load all data mats from ques_file into this
        self.data = {}

        # map from load to save labels
        io_map = {
            'ques_{}': '{}_ques',
            'ques_length_{}': '{}_ques_len',
            'ans_{}': '{}_ans',
            'ans_length_{}': '{}_ans_len',
            'img_pos_{}': '{}_img_pos',
            'cap_{}': '{}_cap',
            'cap_length_{}': '{}_cap_len',
            'opt_{}': '{}_opt',
            'opt_length_{}': '{}_opt_len',
            'opt_list_{}': '{}_opt_list',
            'num_rounds_{}': '{}_num_rounds',
            'ans_index_{}': '{}_ans_ind'
        }

        # processing every split in subsets
        for dtype in subsets:  # dtype is in ['train', 'val', 'test']
            print("\nProcessing split [{}]...".format(dtype))
            # read the question, answer, option related information
            for load_label, save_label in iteritems(io_map):
                if load_label.format(dtype) not in ques_file:
                    continue
                self.data[save_label.format(dtype)] = torch.from_numpy(
                    np.array(ques_file[load_label.format(dtype)], dtype='int64'))

            if 'V' in args.input_type:
                print("Reading video features...")

                # Charades dataset features are all saved in one h5 file as a key, feat dictionary
                # vid_feats = hdfdict.load(
                #     args.input_vid + "_{0}.h5".format(dtype))
                # If this throws an error because it cannot find the video filename,uncomment below
                vid_feats = hdfdict.load(
                    args.input_vid + "_{0}.h5".format("train"))
                vid_feats.update(hdfdict.load(
                    args.input_vid + "_{0}.h5".format("test")))

                img_fnames = getattr(self, 'unique_img_' + dtype)
                self.data[dtype + '_img_fnames'] = img_fnames
                self.data[dtype + '_vid_fv'] = vid_feats

            if 'I' in args.input_type:
                print("Reading image features...")
                img_feats = torch.from_numpy(
                    np.array(img_file['images_' + dtype]))

                if args.img_norm:
                    print("Normalizing image features...")
                    img_feats = F.normalize(img_feats, dim=1, p=2)

                img_fnames = getattr(self, 'unique_img_' + dtype)
                self.data[dtype + '_img_fnames'] = img_fnames
                self.data[dtype + '_img_fv'] = img_feats

            if 'A' in args.input_type:
                print("Reading audio features...")
                audio_feats = torch.from_numpy(
                    np.array(audio_file['images_' + dtype]))
                audio_feats = F.normalize(audio_feats, dim=1, p=2)
                self.data[dtype + '_audio_fv'] = audio_feats

            # record some stats, will be transferred to encoder/decoder later
            # assume similar stats across multiple data subsets
            # maximum number of questions per image, ideally 10
            self.max_ques_count = self.data[dtype + '_ques'].size(1)
            # maximum length of question
            self.max_ques_len = self.data[dtype + '_ques'].size(2)
            # maximum length of answer
            self.max_ans_len = self.data[dtype + '_ans'].size(2)

        # reduce amount of data for preprocessing in fast mode
        # TODO
        self.num_data_points = {}
        for dtype in subsets:
            self.num_data_points[dtype] = len(self.data[dtype + '_ques'])
            print("[{0}] no. of threads: {1}".format(
                dtype, self.num_data_points[dtype]))
        print("\tMax no. of rounds: {}".format(self.max_ques_count))
        print("\tMax ques len: {}".format(self.max_ques_len))
        print("\tMax ans len: {}".format(self.max_ans_len))

        # prepare history
        if 'DH' in args.input_type or 'C' in args.input_type:
            for dtype in subsets:
                self._process_history(dtype)

        for dtype in subsets:
            # 1 indexed to 0 indexed
            self.data[dtype + '_opt'] -= 1
            if dtype + '_ans_ind' in self.data:
                self.data[dtype + '_ans_ind'] -= 1

        # default pytorch loader dtype is set to train
        if 'train' in subsets:
            self._split = 'train'
        else:
            self._split = subsets[0]

        if args.overfit:
            self.num_data_points['train'] = 5
            self.num_data_points['val'] = 5

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets  # ['train', 'val', 'test']
        self._split = split

    # ------------------------------------------------------------------------
    # methods to override - __len__ and __getitem__ methods
    # ------------------------------------------------------------------------

    def __len__(self):
        return self.num_data_points[self._split]

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
                cmd.crop('max(0, (iw - {}))*{}'.format(self.args.video_size, aw),
                         'max(0, (ih - {}))*{}'.format(self.args.video_size, ah),
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

        dtype = self._split
        item = {'index': idx}
        item['num_rounds'] = self.data[dtype + '_num_rounds'][idx]

        # get video features
        if 'V' in self.args.input_type:
            item['img_fnames'] = self.data[dtype + '_img_fnames'][idx]
            # item['img_fnames'] is as train_val/vid_id.jpg hence the splits
            vid_id = item['img_fnames'].split("/")[-1].split(".")[0]
            if ".mp4" not in vid_id:
                vid_id = vid_id + ".mp4"

            if self.args.finetune:
                f_dtype = "train_val"
                if dtype == "test":
                    f_dtype = "test"
                if self.args.use_npy:
                    video_path = os.path.join(self.args.numpy_path, vid_id)
                    item['vid_feat'] = torch.from_numpy(np.load(
                        video_path.replace(".mp4", ".npy")))
                else:
                    video_path = os.path.join(
                        self.args.video_root, f_dtype, vid_id)
                    item['vid_feat'] = self._get_video()(video_path)
            else:
                item['vid_feat'] = torch.from_numpy(
                    self.data[dtype + '_vid_fv'][vid_id]).reshape(-1)

        # get image features
        if 'I' in self.args.input_type:
            item['img_feat'] = self.data[dtype + '_img_fv'][idx]
            item['img_fnames'] = [self.data[dtype + '_img_fnames'][idx]]

        # get audio features
        if 'A' in self.args.input_type:
            item['audio_feat'] = self.data[dtype + '_audio_fv'][idx]

        # get history tokens
        if 'DH' in self.args.input_type or 'caption' in self.args.input_type:
            item['hist_len'] = self.data[dtype + '_hist_len'][idx]
            item['hist_len'][item['hist_len'] == 0] += 1
            item['hist'] = self.data[dtype + '_hist'][idx]

        # get question tokens
        item['ques'] = self.data[dtype + '_ques'][idx]
        item['ques_len'] = self.data[dtype + '_ques_len'][idx]

        # get options tokens
        opt_inds = self.data[dtype + '_opt'][idx]
        opt_size = list(opt_inds.size())
        new_size = torch.Size(opt_size + [-1])
        ind_vector = opt_inds.view(-1)

        option_in = self.data[dtype + '_opt_list'].index_select(0, ind_vector)
        opt_len = self.data[dtype + '_opt_len'].index_select(0, ind_vector)
        option_in = option_in.view(new_size)
        opt_len = opt_len.view(opt_size)

        item['opt'] = option_in
        item['opt_len'] = opt_len
        # if dtype != 'test':
        ans_ind = self.data[dtype + '_ans_ind'][idx]
        item['ans_ind'] = ans_ind.view(-1)

        # convert zero length sequences to one length
        # this is for handling empty rounds of v1.0 test, they will be dropped anyway
        # if dtype == 'test':
        item['ques_len'][item['ques_len'] == 0] += 1
        item['opt_len'][item['opt_len'] == 0] += 1
        return item

    # -------------------------------------------------------------------------
    # collate function utilized by dataloader for batching
    # -------------------------------------------------------------------------

    def collate_fn(self, batch):
        dtype = self._split
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        out = {}
        for key in merged_batch:
            if key in {'index', 'num_rounds', 'img_fnames'}:
                out[key] = merged_batch[key]
            elif key in {'cap_len'}:
                out[key] = torch.Tensor(merged_batch[key]).long()
            else:
                out[key] = torch.stack(merged_batch[key], 0)
        # Dynamic shaping of padded batch
        if 'hist' in out:
            out['hist'] = out['hist'][:, :, :torch.max(
                out['hist_len'])].contiguous()
        out['ques'] = out['ques'][:, :, :torch.max(
            out['ques_len'])].contiguous()
        out['opt'] = out['opt'][:, :, :, :torch.max(
            out['opt_len'])].contiguous()

        return out

    # -------------------------------------------------------------------------
    # preprocessing functions
    # -------------------------------------------------------------------------

    def _process_history(self, dtype):
        """Process caption as well as history. Optionally, concatenate history
        for lf-encoder."""
        captions = self.data[dtype + '_cap']
        questions = self.data[dtype + '_ques']
        ques_len = self.data[dtype + '_ques_len']
        cap_len = self.data[dtype + '_cap_len']
        max_ques_len = questions.size(2)

        answers = self.data[dtype + '_ans']
        ans_len = self.data[dtype + '_ans_len']
        num_convs, num_rounds, max_ans_len = answers.size()

        if self.args.concat_history:
            self.max_hist_len = min(
                num_rounds * (max_ques_len + max_ans_len), 400)
            history = torch.zeros(num_convs, num_rounds,
                                  self.max_hist_len).long()
        else:
            history = torch.zeros(num_convs, num_rounds,
                                  max_ques_len + max_ans_len).long()
        hist_len = torch.zeros(num_convs, num_rounds).long()

        if 'DH' in self.args.input_type:
            # go over each question and append it with answer
            for th_id in range(num_convs):
                clen = cap_len[th_id]
                hlen = min(clen, max_ques_len + max_ans_len)
                for round_id in range(num_rounds):
                    if round_id == 0:
                        # first round has caption as history
                        history[th_id][round_id][:max_ques_len + max_ans_len] \
                            = captions[th_id][:max_ques_len + max_ans_len]
                    else:
                        qlen = ques_len[th_id][round_id - 1]
                        alen = ans_len[th_id][round_id - 1]
                        # if concat_history, string together all previous question-answer pairs
                        if self.args.concat_history:
                            history[th_id][round_id][:hlen] = history[th_id][round_id - 1][:hlen]
                            history[th_id][round_id][hlen] = self.word2ind['<END>']
                            if qlen > 0:
                                history[th_id][round_id][hlen + 1:hlen + qlen + 1] \
                                    = questions[th_id][round_id - 1][:qlen]
                            if alen > 0:
                                # print(round_id, history[th_id][round_id][:10], answers[th_id][round_id][:10])
                                history[th_id][round_id][hlen + qlen + 1:hlen + qlen + alen + 1] \
                                    = answers[th_id][round_id - 1][:alen]
                            hlen = hlen + qlen + alen + 1
                        # else, history is just previous round question-answer pair
                        else:
                            if qlen > 0:
                                history[th_id][round_id][:qlen] = questions[th_id][round_id - 1][:qlen]
                            if alen > 0:
                                history[th_id][round_id][qlen:qlen + alen] \
                                    = answers[th_id][round_id - 1][:alen]
                            hlen = alen + qlen
                    # save the history length
                    hist_len[th_id][round_id] = hlen
        else:  # -- caption only
            # go over each question and append it with answer
            for th_id in range(num_convs):
                clen = cap_len[th_id]
                hlen = min(clen, max_ques_len + max_ans_len)
                for round_id in range(num_rounds):
                    history[th_id][round_id][:max_ques_len +
                                             max_ans_len] = captions[th_id][:max_ques_len + max_ans_len]
                    hist_len[th_id][round_id] = hlen

        self.data[dtype + '_hist'] = history
        self.data[dtype + '_hist_len'] = hist_len
