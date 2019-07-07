import argparse


parser = argparse.ArgumentParser()

###########################################
# dataset
# parser.add_argument('--dataset', required=False, help='soccer| balls | soccer_seq', default='soccer')
parser.add_argument('--dataset', required=False, help='provided | multi | new_sweaty | new_seq', default='provided')
# parser.add_argument('--data_root', required=False, help='root if the new data', default='testDataset')
parser.add_argument('--data_root', required=False, help='root if the new data', default='../SoccerData1')
# parser.add_argument('--xml', required=False, help='test xml path', default=None)

###########################################
# hyperparameters
parser.add_argument('--lr', default=1e-6, type=float)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--weight_decay', default=1e-3, help='regularization constant for l_2 regularizer of W')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--nm_epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--drop_p', type=float, default=0.5, help='Dropout Probability')

###########################################
# sweaty net
parser.add_argument('--net', required=False, help='net1| net2| net3', default='net1')

################### sequential part ########################

###########################################
# tcn hyperparameters
parser.add_argument('--hist', default=20)
parser.add_argument('--nhid', default=50, type=int)
parser.add_argument('--output_size', default=2, type=int)
parser.add_argument('--levels', default=2, type=int)
parser.add_argument('--ksize', default=5, type=int)
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (default: 0.05)')

###########################################
# balls
parser.add_argument('--map_size_x', default=120, type=int)
parser.add_argument('--map_size_y', default=160, type=int)
parser.add_argument('--window_size', default=20, type=int)
parser.add_argument('--n_balls', default=1, type=int)
parser.add_argument('--min_sigma', default=2, type=int)
parser.add_argument('--max_sigma', default=5, type=int)
parser.add_argument('--max_shift', default=300, type=int)
parser.add_argument('--max_move_steps', default=60, type=int)
parser.add_argument('--min_move_steps', default=30, type=int)

parser.add_argument('--balls_folder', default='toy.seq', help='toy.seq | test.toy.seq')
# parser.add_argument('--seq_dataset_root', default='')

# parser.add_argument('--data_root_seq', default='SoccerDataMulti') # seq_real_balls
parser.add_argument('--data_root_seq', default='SoccerDataSeq') # seq_real_balls
parser.add_argument('--real_balls', default=True, type=bool)

###########################################
# resume
parser.add_argument('--sweaty_resume_str',
                    default='model/sweaty/Model_lr_0.001_opt_adam_epoch_100_net_net1_drop_0.5',
                    help='always load part for the sweatynet')

parser.add_argument('--seq_resume', default=True, type=bool,
                    help='load model for embeddings, if positive then it is number of '
                         'epoch which should be loaded')
parser.add_argument('--seq_resume_str',
                    # default='model/tcn_ed/tcn_ed2_1.tcn_ed.ep60.lr1.0e-03_20.pth.tar')
                    default='model/gru/lstm.two.gru.ep60.lr1.0e-03_20.pth.tar')

parser.add_argument('--seq_both_resume', default=True,
                    help='finetuned sweaty net with sequential part simultaneously, load for testing')
parser.add_argument('--seq_both_resume_str',
                    default='model/both/tcn.big.ft._lr_1e-05_opt_adam_epoch_18')

###########################################
# save
parser.add_argument('--result_root', default='results',
                    help='folder to output plots')
parser.add_argument('--seq_save_model', default='gru.ft.',
                    help='model name to save with (prefix)')
parser.add_argument('--save_out', default='seq_output',
                    help='folder name where save some output resuls')

###########################################
# additional
parser.add_argument('--device', default='cuda')
parser.add_argument('--suffix', default='tcn.save')
parser.add_argument('--seq_predict', default=1, type=int)
parser.add_argument('--seq_model', default='tcn', help='lstm | tcn | gru')
parser.add_argument('--print_every', type=int, default=1, help='print checkpoints')
parser.add_argument('--save_every', type=int, default=1, help='model checkpoints')
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
parser.add_argument('--model_root', default='model/', help='folder to output model checkpoints')
parser.add_argument('--reproduce', default='time', help='best | all | time results to reproduce')



opt = parser.parse_args()
opt.input_size = (640, 480)
opt.rot_degree = 45
opt.seq_predict = 1 if opt.seq_model == 'tcn' else 2
# opt.model = opt.seq_model
# if opt.datset == 'soccer':
#     opt.data_root = 'SoccerData1'
# if opt.dataset == 'balls':
#     opt.data_root = 'toy.seq/npy'

