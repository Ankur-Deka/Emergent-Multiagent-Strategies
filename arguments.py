import argparse
import os
import sys
import torch
import shutil


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    # environment
    parser.add_argument('--env-name', default='fortattack-v1', help='one from {simple_spread, simple_formation, simple_line})')
    parser.add_argument('--num-agents', type=int, default=3)
    parser.add_argument('--masking', action='store_true', help='restrict communication to within some threshold')
    parser.add_argument('--mask-dist', type=float, default=1.0, help='distance to restrict comms')
    parser.add_argument('--dropout-masking', action='store_true', help='dropout masking enabled')
    parser.add_argument('--entity-mp', action='store_true', help='enable entity message passing', default = False)
    parser.add_argument('--identity-size', default=0, type=int, help='size of identity vector')

    # training 
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    parser.add_argument('--num-processes', type=int, default=1, help='how many training CPU processes to use (default: 32)')
    parser.add_argument('--num-steps', type=int, default=1000, help='number of forward steps in PPO (default: 128)')   # data collection steps in PPO
    parser.add_argument('--num-env-steps', type=int, default=100, help='number of forward steps in PPO (default: 128)')   # after how many steps environment resets
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--num-frames', type=int, default=int(50e6), help='number of frames to train (default: 5e3)')
    parser.add_argument('--arena-size', type=int, default=1, help='size of arena')

    # evaluation
    parser.add_argument('--num-eval-episodes', type=int, default=1, help='number of episodes to evaluate with')
    parser.add_argument('--dist-threshold', type=float, default=0.1, help='distance within landmark is considered covered (for simple_spread)')
    parser.add_argument('--render', action='store_true', default = False)
    parser.add_argument('--record-video', action='store_true', default=False, help='record evaluation video')
    
    # PPO
    parser.add_argument('--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.05)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    
    # logging
    parser.add_argument('--save-dir', default='tmp_16', help='directory to save models (default: tmp)')
    parser.add_argument('--log-dir', default='logs', help='directory to save logs')
    parser.add_argument('--save-interval', type=int, default=10, help='save interval, one save per n updates (default: 200)')
    parser.add_argument('--log-interval', type=int, default=1, help='log interval, one log per n updates (default: 1)')
    
    # Miscellaneous
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load-dir', default='tmp', help='filename to load all policies from')     
    parser.add_argument('--ckpt', default=20, help='model ckpt')
    parser.add_argument('--train-guards-only', default=False, action='store_true', help='train only the guards against pretrained attackers')
    parser.add_argument('--pretrained-guard', default=False, action='store_true', help='use a pretrained guard to initialize training')
    parser.add_argument('--guard-load-dir', default='tmp')
    parser.add_argument('--guard-ckpt', default=2520)
    parser.add_argument('--attacker-load-dir', default='tmp')
    parser.add_argument('-l', '--attacker-ckpts', nargs = '+', type = int, default=[220,650,1240,1600,2520])     # training guards against trained attackers
    parser.add_argument('--eval-interval', default=100, type=int)
    parser.add_argument('--continue-training', default = False, action='store_true')
    parser.add_argument('--out-file', type = str, default = None, help='output file for storing observations')

    # we always set these to TRUE, so automating this
    parser.add_argument('--no-clipped-value-loss', action='store_true')
    
    args = parser.parse_args()

    args.clipped_value_loss = not args.no_clipped_value_loss

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.continue_training:
        args.save_dir = args.load_dir
        
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.save_dir = './marlsave/'+args.save_dir
    args.log_dir = args.save_dir + '/' + args.log_dir

    args.load_dir = './marlsave/'+args.load_dir+'/ep'+str(args.ckpt)+'.pt'

    if args.train_guards_only:
        args.attacker_load_dir = './marlsave/'+args.attacker_load_dir
        if args.pretrained_guard:
            print('loading pretrained guards')
            args.guard_load_dir = './marlsave/'+args.guard_load_dir+'/ep'+str(args.guard_ckpt)+'.pt'

    if args.continue_training:
        assert args.load_dir is not None and os.path.exists(args.load_dir), \
        "Please specify valid model file to load if you want to continue training"

    if args.identity_size > 0:
        assert args.identity_size >= args.num_agents, 'identity size should either be 0 or >= number of agents!'

    if not args.masking:
        args.mask_dist = None
    elif args.masking and args.dropout_masking:
        args.mask_dist = -10
        
    # raise warning if save directory already exists
    if not (args.test or args.continue_training):
        if os.path.exists(args.save_dir):
            print('\nSave directory exists already! Enter')
            ch = input('c (rename the existing directory with _old and continue)\ns (stop)!\ndel (delete existing dir): ')
            if ch == 's':
                sys.exit(0)
            elif ch == 'c':
                os.rename(args.save_dir, args.save_dir+'_old')
            elif ch == 'del':
                shutil.rmtree(args.save_dir)
            else:
                raise NotImplementedError('Unknown input')
        os.makedirs(args.save_dir)
    
    return args
