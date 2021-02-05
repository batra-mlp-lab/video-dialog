import argparse


def get_args(parser, description='MILNCE'):
    if parser is None:
        parser = argparse.ArgumentParser(description=description)

    parser.add_argument_group('Input modalites arguments')

    parser.add_argument('-input_type', default='Q_DH_V',
                        choices=['Q_only', 'Q_DH', 'Q_A', 'Q_I', 'Q_V', 'Q_C_I', 'Q_DH_V', 'Q_DH_I', 'Q_V_A', 'Q_DH_V_A'], help='Specify the inputs')

    parser.add_argument_group('Encoder Decoder choice arguments')
    parser.add_argument('-encoder', default='lf-ques-im-hist',
                        choices=['lf-ques-im-hist'], help='Encoder to use for training')
    parser.add_argument('-concat_history', default=True,
                        help='True for lf encoding')
    parser.add_argument('-decoder', default='disc',
                        choices=['disc'], help='Decoder to use for training')
    parser.add_argument('-finetune_textEncoder', default=0, type=int,
                         help= 'Finetune the text encoder')
    parser.add_argument_group('Optimization related arguments')
    parser.add_argument('-num_epochs', default=45, type=int, help='Epochs')
    parser.add_argument('-batch_size', default=12, type=int, help='Batch size')
    parser.add_argument('-lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('-lr_decay_rate', default=0.9,
                        type=float, help='Decay  for lr')
    parser.add_argument('-min_lr', default=5e-5, type=float,
                        help='Minimum learning rate')
    parser.add_argument('-weight_init', default='xavier',
                        choices=['xavier', 'kaiming'], help='Weight initialization strategy')
    parser.add_argument('-weight_decay', default=5e-4,
                        help='Weight decay for l2 regularization')
    parser.add_argument('-overfit', action='store_true',
                        help='Overfit on 5 examples, meant for debugging')
    parser.add_argument('-gpuid', default=0, type=int, help='GPU id to use')

    parser.add_argument_group('Checkpointing related arguments')
    parser.add_argument('-load_path', default='',
                        help='Checkpoint to load path from')
    parser.add_argument('-save_path', default='checkpoints/',
                        help='Path to save checkpoints')
    parser.add_argument('-save_step', default=4, type=int,
                        help='Save checkpoint after every save_step epochs')
    parser.add_argument('-eval_step', default=100, type=int,
                        help='Run validation after every eval_step iterations')
    parser.add_argument('-input_vid', default="data/charades_s3d_mixed_5c_fps_16_num_frames_40_original_scaled",
                        help=".h5 file path for the charades s3d features.")
    parser.add_argument('-finetune', default=0, type=int,
                        help="When set true, the model finetunes the s3dg model for video")

    # S3DG parameters and dataloader
    parser.add_argument('-num_frames', type=int, default=40,
                        help='num_frame')
    parser.add_argument('-video_size', type=int, default=224,
                        help='random seed')
    parser.add_argument('-fps', type=int, default=16, help='')
    parser.add_argument('-crop_only', type=int, default=1,
                        help='random seed')
    parser.add_argument('-center_crop', type=int, default=0,
                        help='random seed')
    parser.add_argument('-random_flip', type=int, default=0,
                        help='random seed')
    parser.add_argument('-video_root', default='data/videos')
    parser.add_argument('-unfreeze_layers', default=1, type=int,
                        help="if 1, unfreezes _5 layers, if 2 unfreezes _4 and _5 layers, if 0, unfreezes all layers")
    parser.add_argument("-text_encoder", default="lstm",
                        help="lstm or transformer", type=str)
    parser.add_argument("-use_npy", default=1, type=int,
                        help="Uses npy instead of reading from videos")
    parser.add_argument("-numpy_path", default="data/charades")
    parser.add_argument("-num_workers", default=8, type=int)

    parser.add_argument_group('Visualzing related arguments')
    parser.add_argument('-enableVis', type=int, default=1)
    parser.add_argument('-visEnvName', type=str, default='s3d_Nofinetune')
    parser.add_argument('-server', type=str, default='127.0.0.1')
    parser.add_argument('-serverPort', type=int, default=8855)
    parser.add_argument('-set_cuda_device', type=str, default='')
    parser.add_argument("-seed", type=int, default=1,
                        help="random seed for initialization")

    parser.add_argument('-save_ranks', action='store_true', help='Whether to save retrieved ranks')
    parser.add_argument('-use_gt', action='store_true', help='Whether to use ground truth for retriveing ranks')
    parser.add_argument('--split', default='test', choices=['val', 'test', 'train'], help='Split to evaluate on')
    # ----------------------------------------------------------------------------
    # input arguments and options
    # ----------------------------------------------------------------------------

    args = parser.parse_args()
    return args
