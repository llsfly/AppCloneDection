import sys
sys.path.extend(["../","./"])
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

def train(pos_data, clone_detection, vocab, config):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, clone_detection.model.parameters()), config)

    global_step = 0
    for iter in range(config.train_iters):
        start_time = time.time()
        neg_data = creat_negative_corpus(pos_data, vocab, config.neg_count)

        data = pos_data + neg_data
        total_num = len(data)
        batch_num = int(np.ceil(total_num / float(config.train_batch_size)))
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))

        batch_iter = 0
        total_loss = 0
        for onebatch in data_iter(data, config.train_batch_size, True):
            tinst = batch_data_variable(onebatch, vocab)
            clone_detection.model.train()
            if clone_detection.use_cuda:
                tinst.to_cuda(clone_detection.device)
            clone_detection.forward(tinst.inputs)
            loss = clone_detection.compute_loss(tinst.outputs)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            total_loss += loss_value
            loss.backward()

            during_time = float(time.time() - start_time)

            print("Step:%d, Iter:%d, batch:%d, time:%.2f, loss:%.2f, avg loss=%.2f" \
                %(global_step, iter, batch_iter, during_time, loss_value, total_loss/(batch_iter+1)))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, clone_detection.model.parameters()), \
                                        max_norm=config.clip)
                optimizer.step()
                clone_detection.model.zero_grad()       
                global_step += 1

        if config.save_after > 0 and iter > config.save_after:
            torch.save(clone_detection.model.state_dict(), config.save_model_path + "." + str(iter))



class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='default.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--len', default=512, type=int, help='max function length')
    argparser.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    pos_data = read_positive_corpus(config.train_file, args.len)

    vocab = creat_vocab(pos_data)
    token_vec = vocab.load_token_embs(config.pretrained_token_file)
    word_vec = vocab.load_word_embs(config.pretrained_word_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    config.use_cuda = False
    if gpu and args.gpu != -1:
        config.use_cuda = True
        torch.cuda.set_device(args.gpu)
        print('GPU ID:' + str(args.gpu))
    print("\nGPU using status: ", config.use_cuda)
    # print(config.use_cuda)

    model = BiLSTMModel(vocab, config, token_vec, word_vec)
    if config.use_cuda:
        #torch.backends.cudnn.enabled = True
        model = model.cuda(args.gpu)

    classifier = CloneDetection(model, vocab, config.loss_margin)

    train(pos_data, classifier, vocab, config)

