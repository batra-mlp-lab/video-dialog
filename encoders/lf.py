import torch
from torch import nn
from torch.nn import functional as F
from utils import DynamicRNN
from utils import DynamicRNN
from encoders.s3dg_video import S3D
from transformers import BertTokenizer, BertModel

class LateFusionEncoder(nn.Module):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Encoder specific arguments')
        parser.add_argument('-img_feature_size', default=1024,
                            help='Channel size of image feature')
        parser.add_argument('-vid_feature_size', default=1024,
                            help='Channel size of video feature')
        parser.add_argument('-audio_feature_size', default=1024,
                            help='Channel size of audio feature')
        parser.add_argument('-embed_size', default=300,
                            help='Size of the input word embedding')
        parser.add_argument('-rnn_hidden_size', default=512,
                            help='Size of the multimodal embedding')
        parser.add_argument('-num_layers', default=2,
                            help='Number of layers in LSTM')
        parser.add_argument('-max_history_len', default=60,
                            help='Size of the multimodal embedding')
        parser.add_argument('-dropout', default=0.5, help='Dropout')
        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.text_encoder == 'lstm':
            args.embed_size = 300
            self.word_embed = nn.Embedding(
                args.vocab_size, args.embed_size, padding_idx=0)
        else:
            freeze_embeddings = True
            args.embed_size = 768
            self.word_embed = BertModel.from_pretrained('bert-base-uncased')
            # Freeze all the layers and use bert to encode the text for now
            if not self.args.finetune_textEncoder:
                print('Freezing all bert layers')
                for param in self.word_embed.parameters():
                    param.requires_grad = False
            else:
                print('Finetuning text encoder layers')
                for param in list(self.word_embed.embeddings.parameters()):
                    param.requires_grad = False
                print("Froze Embedding Layer")
                freeze_layers = "0,1,2,3,4,5,6,7,8,9"
                layer_indexes = [int(x) for x in freeze_layers.split(",")]
                for layer_idx in layer_indexes:
                    for param in list(self.word_embed.encoder.layer[layer_idx].parameters()):
                        param.requires_grad = False
                    print("Froze Layer: ", layer_idx)
            for name, param in self.word_embed.named_parameters():
                    print(name, param.requires_grad)
        if self.args.finetune:
            self.video_embed = S3D(
                dict_path='data/s3d_dict.npy', space_to_depth=True)
            self.video_embed.load_state_dict(
                torch.load('data/s3d_howto100m.pth'), strict=False)

            self.video_embed.train()
            if self.args.unfreeze_layers:
                self.__freeze_s3dg_layers()

        if 'DH' in args.input_type or 'C' in args.input_type:
            self.hist_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size,
                                    args.num_layers, batch_first=True, dropout=args.dropout)
            self.hist_rnn = DynamicRNN(self.hist_rnn)

        self.ques_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size,
                                args.num_layers, batch_first=True,
                                dropout=args.dropout)
        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        self.dropout = nn.Dropout(p=args.dropout)
        # fusion layer
        if args.input_type == 'Q_only':
            fusion_size = args.rnn_hidden_size
        if args.input_type == 'Q_DH':
            fusion_size = args.rnn_hidden_size * 2
        if args.input_type == 'Q_A':
            fusion_size = args.rnn_hidden_size + args.audio_feature_size
        if args.input_type == 'Q_I' or args.input_type == 'Q_V':
            fusion_size = args.img_feature_size + args.rnn_hidden_size
        if args.input_type == 'Q_C_I' or args.input_type == 'Q_DH_V' or args.input_type == 'Q_DH_I':
            fusion_size = args.img_feature_size + args.rnn_hidden_size * 2
        if args.input_type == 'Q_V_A':
            fusion_size = args.img_feature_size + \
                args.rnn_hidden_size + args.audio_feature_size
        if args.input_type == 'Q_DH_V_A':
            fusion_size = args.img_feature_size + \
                args.rnn_hidden_size * 2 + args.audio_feature_size

        self.fusion = nn.Linear(fusion_size, args.rnn_hidden_size)

        if args.weight_init == 'xavier':
            nn.init.xavier_uniform_(self.fusion.weight.data)
        elif args.weight_init == 'kaiming':
            nn.init.kaiming_uniform_(self.fusion.weight.data)
        nn.init.constant_(self.fusion.bias.data, 0)

    def __freeze_s3dg_layers(self):
        # Only train _4 and _5 layers
        layers = ["mixed_5c"]
        if self.args.unfreeze_layers == 2:
            layers = ["mixed_5b", "mixed_5c"]
        for name, param in self.video_embed.named_parameters():
            param.requires_grad = False
            if any(l in name for l in layers):
                param.requires_grad = True

    def forward(self, batch):
        if 'I' in self.args.input_type:
            img = batch['img_feat']
            # repeat image feature vectors to be provided for every round
            img = img.view(-1, 1, self.args.img_feature_size)
            img = img.repeat(1, self.args.max_ques_count, 1)
            img = img.view(-1, self.args.img_feature_size)

        if 'A' in self.args.input_type:
            audio = batch['audio_feat']
            # repeat audio feature vectors to be provided for every round
            audio = audio.view(-1, 1, self.args.audio_feature_size)
            audio = audio.repeat(1, self.args.max_ques_count, 1)
            audio = audio.view(-1, self.args.audio_feature_size)

        if 'V' in self.args.input_type:
            if self.args.finetune:
                # In this case, vid_feat has video frames.Multiplication by 255 because s3d video frames are normalised
                vid = self.video_embed(batch['vid_feat'].float())[
                    "mixed_5c"] * 255.0
            else:
                vid = batch['vid_feat'] * 255.0
            # repeat image feature vectors to be provided for every round
            vid = vid.view(-1, 1, self.args.vid_feature_size)
            vid = vid.repeat(1, self.args.max_ques_count, 1)
            vid = vid.view(-1, self.args.vid_feature_size)

        if 'DH' in self.args.input_type or 'C' in self.args.input_type:
            hist = batch['hist']
            # embed history
            hist = hist.view(-1, hist.size(2))
            hist_embed = self.word_embed(hist)

            if self.args.text_encoder == 'BERT':
                hist_embed = hist_embed['last_hidden_state']

            hist_embed = self.hist_rnn(hist_embed, batch['hist_len'])

        ques = batch['ques']

        # embed questions
        ques = ques.view(-1, ques.size(2))
        ques_embed = self.word_embed(ques)
        if self.args.text_encoder == 'BERT':
            ques_embed = ques_embed['last_hidden_state']
        ques_embed = self.ques_rnn(ques_embed, batch['ques_len'])

        if self.args.input_type == 'Q_only':
            fused_vector = ques_embed
        if self.args.input_type == 'Q_DH':
            fused_vector = torch.cat((ques_embed, hist_embed), 1)
        if self.args.input_type == 'Q_A':
            fused_vector = torch.cat((audio, ques_embed), 1)
        if self.args.input_type == 'Q_I':
            fused_vector = torch.cat((img, ques_embed), 1)
        if self.args.input_type == 'Q_V':
            fused_vector = torch.cat((vid, ques_embed), 1)
        if self.args.input_type == 'Q_DH_I':
            fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        if self.args.input_type == 'Q_DH_V':
            fused_vector = torch.cat((vid, ques_embed, hist_embed), 1)
        if self.args.input_type == 'Q_C_I':
            fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        if self.args.input_type == 'Q_V_A':
            fused_vector = torch.cat((vid, audio, ques_embed), 1)
        if self.args.input_type == 'Q_DH_V_A':
            fused_vector = torch.cat((vid, audio, ques_embed, hist_embed), 1)

        fused_vector = self.dropout(fused_vector)

        fused_embedding = torch.tanh(self.fusion(fused_vector))
        return fused_embedding
