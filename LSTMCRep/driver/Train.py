import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from model.BiLSTMModel import *
from driver.CDHelper import *
from data.Dataloader import *
import pickle


def train(data, clone_detection, vocab, config, outfile):
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))

    start_time = time.time()
    print('total batch num: ' + str(batch_num))
    batch_iter = 0
    with open(outfile, 'w') as file:
        for onebatch in data_iter(data, config.train_batch_size, False):
            tinst = batch_data_variable(onebatch, vocab)
            if clone_detection.use_cuda:
                tinst.to_cuda(clone_detection.device)
            represents = clone_detection.forward(tinst.inputs)
            cur_batch_size = len(onebatch)
            for idx in range(cur_batch_size):
                cur_inst = onebatch[idx]
                cur_represent = represents[idx].detach().cpu().numpy()
                index = cur_inst.id
                reps = [str(val) for val in cur_represent]
                outline = str(index) + '\t' + ' '.join(reps)
                file.write(outline + '\n')

            during_time = float(time.time() - start_time)
            print("batch:%d, time:%.2f" % (batch_iter, during_time))
            batch_iter += 1


if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)

    # gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='code.cfg')
    argparser.add_argument('--infile', default='clone4/bcb.txt')
    argparser.add_argument('--outfile', default='clone4/bcb.represent')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file)
    torch.set_num_threads(args.thread)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    token_vec = vocab.create_token_embs(config.pretrained_token_file)
    word_vec = vocab.create_word_embs(config.pretrained_word_file)

    config.use_cuda = False
    if gpu and args.gpu != -1:
        config.use_cuda = True
        torch.cuda.set_device(args.gpu)
        print('GPU ID:' + str(args.gpu))
    print("\nGPU using status: ", config.use_cuda)
    # print(config.use_cuda)

    model = BiLSTMModel(vocab, config, token_vec, word_vec)
    model.load_state_dict(torch.load(config.load_model_path, map_location=lambda storage, loc: storage))

    if config.use_cuda:
        # torch.backends.cudnn.enabled = True
        model = model.cuda(args.gpu)

    classifier = CloneDetection(model)
    data = read_corpus(args.infile)
    train(data, classifier, vocab, config, args.outfile)