import argparse
import os

def al_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein_name', type=str, default='P51449')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='features embedding dimensions ')
    parser.add_argument('--smi_dim', type=int, default=256,
                        help='smiles extractor hidden dimensions ')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=1000,help='predictor train epoch')
    parser.add_argument('--stop_epoch', type=int, default=300)
    parser.add_argument('--gen_epoch', type=int, default=10,help='generator finetune epoch')
    parser.add_argument('--al_iter', type=int, default=50)
    parser.add_argument('--num_sample', type=int, default=120)
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--runseed', type=int, default=5656,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--now', type=int, default=0)
    parser.add_argument('--max_smi_len', type=int, default=100, help='Length of input sequences.')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory for log data.')

    args, _ = parser.parse_known_args()
    return args

def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, f"log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)