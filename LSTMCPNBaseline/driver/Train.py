import sys
sys.path.extend(["../", "./"])
import time
import argparse
import torch.optim.lr_scheduler
import random
from driver.Config import *
from model.BiLSTMModel import *
from driver.CDHelper import *
from data.Dataloader import *
import pickle
from premodel.BiLSTMModel import BiLSTMModel as PreModel
from premodel.Config import Configurable as PreConfig
from premodel.Vocab import Vocab as PreVocab

def train(train_data, clone_data, non_clone_data, seq_codes, clone_detection, vocab, config):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, clone_detection.model.parameters()), config)

    global_step = 0
    best_acc = 0
    batch_num = int(np.ceil(len(train_data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num = 0, 0
        for onebatch in data_iter(train_data, config.train_batch_size, False):
            tinst = batch_data_variable_train(onebatch, vocab)
            clone_detection.model.train()
            if clone_detection.use_cuda:
                tinst.to_cuda(clone_detection.device)
            clone_detection.forward(tinst.inputs)
            loss = clone_detection.compute_loss(tinst.outputs)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            cur_correct, cur_count = clone_detection.compute_accuracy(tinst.outputs)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num
            during_time = float(time.time() - start_time)
            print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                %(global_step, acc, iter, batch_iter, during_time, loss_value))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, clone_detection.model.parameters()), \
                                        max_norm=config.clip)
                optimizer.step()
                clone_detection.model.zero_grad()       
                global_step += 1

            if (iter+1) % config.validate_every == 0 and batch_iter == batch_num:
                code_reprs = code2vec(seq_codes, clone_detection, vocab, config)
                output = open(config.test_file + '.' + str(global_step), 'w', encoding='utf-8')
                clone_correct, clone_total, clone_acc = \
                    evaluate(clone_data, clone_detection, vocab, code_reprs, output, config)
                non_clone_correct, non_clone_total, non_clone_acc = \
                    evaluate(non_clone_data, clone_detection, vocab, code_reprs, output, config)
                output.close()

                print("clone: acc = %d/%d = %.2f" % (clone_correct, clone_total, clone_acc))
                print("non clone: acc = %d/%d = %.2f" % (non_clone_correct, non_clone_total, non_clone_acc))
                dev_tag_acc = (clone_correct + non_clone_correct) * 100.0 / (clone_total + non_clone_total)
                print("Total: acc = %d/%d = %.2f" % (clone_correct + non_clone_correct, \
                                                     clone_total + non_clone_total, dev_tag_acc))
                if dev_tag_acc > best_acc:
                    print("Exceed best acc: history = %.2f, current = %.2f" %(best_acc, dev_tag_acc))
                    best_acc = dev_tag_acc
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(clone_detection.model.state_dict(), config.save_model_path)


def evaluate(data, clone_detection, vocab, code_reprs, output, config):
    start = time.time()
    clone_detection.model.eval()
    tag_correct, tag_total = 0, 0
    final_sent_dim = clone_detection.model.sent_dim + vocab.embedding_dim

    for onebatch in data_iter(data, config.test_batch_size, False):
        tinst = batch_data_variable_test(onebatch, vocab, final_sent_dim, code_reprs)
        if clone_detection.use_cuda:
            tinst.to_cuda(clone_detection.device)
        count = 0
        pred_tags, probs = clone_detection.classifier(tinst.inputs)
        for inst, bmatch, str_prob, gold_tag in batch_variable_inst(onebatch, pred_tags, probs, vocab):
            output.write(str(inst) + ' ' + str_prob + ' ' + gold_tag + '\n')
            tag_total += 1
            if bmatch: tag_correct += 1
            count += 1
            if tag_total % config.display_interval == 0:
                acc = tag_correct * 100.0 / tag_total
                end = time.time()
                during_time = float(end - start)
                print("processing: acc=%d/%d=%.2f, classifier time=%.2f" % (tag_correct, tag_total, acc, during_time))

    output.flush()
    acc = tag_correct * 100.0 / tag_total
    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  classifier time=%.2f " % (len(data), during_time))

    return tag_correct, tag_total, acc


def code2vec(seq_codes, clone_detection, vocab, config):
    start = time.time()
    clone_detection.model.eval()
    code_reprs = {}
    count = 0

    for onebatch in data_iter(seq_codes, config.code_batch_size, False):
        tinst = batch_data_variable_code(onebatch, vocab)
        if clone_detection.use_cuda:
            tinst.to_cuda(clone_detection.device)
        represents = clone_detection.code2vec(tinst.inputs)
        represents = represents.detach().cpu()
        cur_batch_size = len(onebatch)
        for idx in range(cur_batch_size):
            cur_inst = onebatch[idx]
            cur_represent = represents[idx]
            index = cur_inst.id
            code_reprs[index] = cur_represent
            count += 1

            if count % config.code_display_interval == 0:
                end = time.time()
                during_time = float(end - start)
                print("processing num: %d, classifier time=%.2f" % (count, during_time))

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  classifier time=%.2f " % (count, during_time))

    return code_reprs

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
    argparser.add_argument('--preconfig_file', default='code.cfg')
    argparser.add_argument('--max_train', default=100, type=int, help='max training clone num')
    argparser.add_argument('--weight', default=1, type=int, help='1:w, where w is the fold of non clones')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    pre_config = PreConfig(args.preconfig_file)
    torch.set_num_threads(args.thread)

    pre_vocab = pickle.load(open(pre_config.load_vocab_path, 'rb'))
    token_vec = pre_vocab.create_token_embs(pre_config.pretrained_token_file)
    word_vec = pre_vocab.create_word_embs(pre_config.pretrained_word_file)

    pre_model = PreModel(pre_vocab, pre_config, token_vec, word_vec)
    pre_model.load_state_dict(torch.load(pre_config.load_model_path, map_location=lambda storage, loc: storage))

    codes, seq_codes = read_codes(config.code_file)
    train_clone_data, train_non_clones = read_corpus(codes, config.train_file)

    train_clone_num, train_non_clone_num = len(train_clone_data), len(train_non_clones)
    if args.max_train < train_clone_num: train_clone_num = args.max_train
    if args.weight * args.max_train < train_non_clone_num: train_non_clone_num = args.weight * args.max_train
    train_data = train_clone_data[:train_clone_num] + train_non_clones[:train_non_clone_num]
    print("Used clone %d, non clone: %d" %(train_clone_num,train_non_clone_num ))

    vocab = creatVocab(train_data)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    vocab.load_pretrained_reprs(config.pretrained_repr_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    config.use_cuda = False
    if gpu and args.gpu != -1:
        config.use_cuda = True
        torch.cuda.set_device(args.gpu)
        print('GPU ID:' + str(args.gpu))
    print("\nGPU using status: ", config.use_cuda)

    model = BiLSTMModel(vocab, config, vec)
    model.part_copy(pre_model)
    if config.use_cuda:
        #torch.backends.cudnn.enabled = True
        model = model.cuda()

    clone_detection = CloneDetection(model, vocab)
    test_clone_data, test_non_clones = read_corpus(codes, config.test_file)
    train(train_data, test_clone_data, test_non_clones, seq_codes, clone_detection, vocab, config)
