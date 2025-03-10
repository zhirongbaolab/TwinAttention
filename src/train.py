import argparse
import torch.utils.data
from torch.utils.data import DataLoader
from model import NIT_Registration, neuron_data_pytorch
import torch
import math
import time
import os
import numpy as np
import datetime

# ADDED BY ERIN TO USE TENSORBOARD
import torch
torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter("/bao_data_zrc/baolab/erinhaus/hpc_runs")

def evaluate_ppl(model, dev_data_loader):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        #for pt1_batch, pt2_batch, match_batch in batch_iter(dev_data, batch_size=batch_size):
        for batch_idx, data_batch in enumerate(dev_data_loader):
            #for pt_batch, match_dict in train_data.batch_iter():
            pt_batch = data_batch['pt_batch']
            match_dict = data_batch['match_dict']
        #for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            #loss = -model(src_sents, tgt_sents).sum()
            _, loss, _ = model(pt_batch, match_dict=match_dict, mode='train')


            cum_loss += loss['loss'].item()
            tgt_word_num_to_predict = loss['num']  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl, cum_loss


if __name__ == "__main__":
    # train the model.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--log_every", default=500, type=int)
    # parser.add_argument("--valid_niter", default=500, type=int,
    #                     help="perform validation after how many iterations")
    parser.add_argument("--valid_niter", default=10000, type=int,
                        help="perform validation after how many iterations")
    parser.add_argument("--model_path", default="../model", type=str)
    parser.add_argument("--lr_decay", default=0.5, type=float, help="learning rate decay")
    parser.add_argument("--max_num_trial", default=10, type=int,
                        help="terminate training after how many trials")
    parser.add_argument("--patience", default=5, type=int,
                        help="wait for how many iterations to decay learning rate")
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--n_hidden", default=48, type=int) # DEFAULT IS 48 IN lEIFER CODE
    parser.add_argument("--n_layer", default=8, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--stn_lam", default=1, type=float)
    parser.add_argument("--fstn_lam", default=1, type=float)
    parser.add_argument("--p_rotate", default=1, type=int)
    parser.add_argument("--f_trans", default=1, type=int)
    parser.add_argument("--cuda", default=1, type=int) #DEFAULT IS 1 IN LEIFER CODE
    parser.add_argument("--train_path", default="../Data/train", type=str)
    parser.add_argument("--eval_path", default="../Data/test", type=str)
    parser.add_argument("--data_mode", default="all", type=str)
    parser.add_argument("--model_idx", default=627, type=int)
    parser.add_argument("--lamb_entropy", default=0.1, type=float)
    parser.add_argument("--model_name", default='nitReg', type=str)
    parser.add_argument("--use_pretrain", default=1, type=int)
    # 25 JAN ADDED BY ERIN TO ADD WEIGHT DECAY
    parser.add_argument("--wd",default=0,type=float)
    # 30 jan adding option to use AdamW for weight decay
    parser.add_argument("--opt_algorithm",default="Adam",type=str)
    tic = time.time()
    args = parser.parse_args()
    print('n_hidden:{}\n'.format(args.n_hidden))
    print('n_layer:{}\n'.format(args.n_layer))
    print('learn rate:{}\n'.format(args.lr))
    print('data mode:{}\n'.format(args.data_mode))
    cuda = args.cuda

    # loading the data
    train_data = neuron_data_pytorch(args.train_path, batch_sz=args.batch_size, shuffle=True, rotate=True, mode=args.data_mode)
    
    dev_data = neuron_data_pytorch(args.eval_path, batch_sz=args.batch_size, shuffle=True, rotate=True, mode=args.data_mode)


    train_data_loader = DataLoader(train_data, shuffle=False, num_workers=1, collate_fn=train_data.custom_collate_fn)
    dev_data_loader = DataLoader(dev_data, shuffle=False, num_workers=1, collate_fn=dev_data.custom_collate_fn)

    # CHANGED BY ERIN 06/21/2023 first (commented out) line is Leifer repo version (wouldn't switch to cpu for some reason)
    device = torch.device("cuda" if cuda else "cpu")
    #device = torch.device("cpu")

    model = NIT_Registration(input_dim=3, n_hidden=args.n_hidden, n_layer=args.n_layer, p_rotate=args.p_rotate,
                feat_trans=args.f_trans)
    if args.use_pretrain:
        print("CONFIRM CORRECT PRETRAINED MODEL LOADED")
        pretrain_path = './model/late_embryo_final.bin'
        params = torch.load(pretrain_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])

        model_name=pretrain_path.split("/")[-1].split(".")[0] + '_realretrain_L1_model.bin'#_testing8block_19dec.bin'#"_13oct_adaptive_c.bin"
    else:
        model_name = '{}_nh{}_nl{}_ft{}_data{}_elam_{}_{}_L1_Model_pretrained.bin'.format(args.model_name, args.n_hidden, args.n_layer,
                                                                  args.f_trans, args.data_mode, args.lamb_entropy,
                                                                  args.model_idx)
    model_save_path = os.path.join(args.model_path, model_name)
    model.train()
    model = model.to(device)
    # 30 jan edit by erin to add AdamW as option for optimizer algorithm
    if args.opt_algorithm == "AdamW":
        print("USING ADAM-W!")
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.wd)
    else:
        # 25 JAN CHANGE BY ERIN TO ADD WEIGHT DECAY
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.wd)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    while True:
        epoch += 1
        batch_num = train_data.batch_num
        current_iter = 0

        if train_data_loader.dataset.shuffle:
            train_data_loader.dataset.shuffle_batch()
        #for pt1_batch, pt2_batch, match_batch in batch_iter(train_data, batch_size=args.batch_size, shuffle=True):
        for batch_idx, data_batch in enumerate(train_data_loader):
            # print("TRAINING BATCH IDX: ", batch_idx) # This line is commented out by Ruohan on 8 Oct 2024
            #print("training data batch: ", print(data_batch))
            #Erin TROUBLESHOOTING - line below produces {'pt_batch': [], 'match_dict': {}, 'pt_label': [], 'ref_i': 0}
            #print(data_batch)
            #for pt_batch, match_dict in train_data.batch_iter():
            pt_batch = data_batch['pt_batch']
            match_dict = data_batch['match_dict']
            # print('batch to batch time:{}'.format(time.time() - tic)) # This line is commented out by Ruohan on 8 Oct 2024
            # ADDED BY ERIN
            now = datetime.datetime.now()
            # print("current time: ") # This line is commented out by Ruohan on 8 Oct 2024
            # print(now) # This line is commented out by Ruohan on 8 Oct 2024
            #writer.add_scalar("Time",now,epoch)
            #writer.add_scalar("Time",now,batch_num)
            #tic = time.time()
            current_iter += 1
            train_iter += 1

            optimizer.zero_grad()
            batch_size = len(pt_batch)

            _, batch_loss, _ = model(pt_batch, match_dict=match_dict, ref_idx=data_batch['ref_i'], mode='train')
             #batch_loss = example_losses.sum()
            loss = batch_loss['loss'] / batch_loss['num'] + args.stn_lam * batch_loss['reg_stn'] + \
                   args.fstn_lam * batch_loss['reg_fstn'] + args.lamb_entropy * batch_loss['loss_entropy'] / batch_loss['num_unlabel']
            loss.backward()
            # EDIT BY ERIN - these were commented out (line 145-148):
            # print('batch loss:{}'.format(batch_loss['loss'] / batch_loss['num'])) # This line is commented out by Ruohan on 8 Oct 2024
            # print(batch_loss) # This line is commented out by Ruohan on 8 Oct 2024
            # print('reg_stn:{}'.format(batch_loss['reg_stn'].item())) # This line is commented out by Ruohan on 8 Oct 2024
            # Erin troubleshooting - line below provides same val as line above?
            #print('reg_stn:{}'.format(batch_loss['reg_stn']))
            # print('reg_fstn:{}'.format(batch_loss['reg_fstn'].item())) # This line is commented out by Ruohan on 8 Oct 2024
            # train_logdir="/bao_data_zrc/baolab/erinhaus/hpc_runs/get_rot_metrics/50set_real_train" # This line is commented out by Ruohan on 8 Oct 2024
            #train_writer = SummaryWriter("/bao_data_zrc/baolab/erinhaus/hpc_runs/train")
            #ADDED BY ERIN FOR TENSORBOARD
            #with tf.summary.create_file_writer(train_logdir).as_default():
            #train_writer = SummaryWriter(train_logdir)
            #train_writer.add_scalar("Loss/train", loss,train_iter)
            #train_writer.add_scalar("Reg stn", batch_loss['reg_stn'].item(),train_iter)
            #train_writer.add_scalar("Reg fstn", batch_loss['reg_fstn'].item(),train_iter)
            #train_writer.flush()
            #train_writer.close()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            batch_losses_val = batch_loss['loss'].item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            # omitting leading `<s>`
            tgt_words_num_to_predict = batch_loss['num']
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % args.log_every == 0:
                print('epoch %d (%d / %d), iter %d, avg. loss %.2f, avg. ppl %.2f '
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' %
                      (epoch, current_iter, batch_num, train_iter,
                       report_loss / report_examples,
                       math.exp(report_loss / report_tgt_words),
                       cum_examples,
                       report_tgt_words / (time.time() - train_time),
                       time.time() - begin_time))
                #eval_logdir='/bao_data_zrc/baolab/erinhaus/hpc_runs/50set_real_train_eval'
                #eval_writer = SummaryWriter(eval_logdir)
                #eval_writer.add_scalar("avg. ppl", math.exp(report_loss / report_tgt_words),batch_num)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % args.valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                      cum_loss / cum_examples,
                      np.exp(cum_loss / cum_tgt_words),
                      cum_examples))

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...')

                # compute dev. ppl and bleu
                dev_ppl,valid_loss = evaluate_ppl(model, dev_data_loader)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl
                #eval_logdir='/bao_data_zrc/baolab/erinhaus/hpc_runs/rerun_20Sept/eval'
                #with tf.summary.create_file_writer(eval_logdir).as_default():
                #eval_writer.add_scalar("Dev_ppl", dev_ppl, valid_num)
                #eval_writer.add_scalar("Valid_metric",valid_metric,valid_num)
                #eval_writer.add_scalar("Dev_loss",valid_loss,valid_num)
                #eval_writer.flush()
                #eval_writer.close()
                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl))

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('epoch %d, iter %d: save currently the best model to [%s]' %
                          (epoch, train_iter, model_save_path))
                    print("last epoch:",epoch)
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < args.patience:
                    patience += 1
                    print('hit patience %d' % patience)

                    if patience == args.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial)
                        if num_trial == args.max_num_trial:
                            print('early stop!')
                            #writer.flush()
                            print("final dev ppl:",dev_ppl)
                            print("final valid loss:",valid_loss)
                            print("last epoch:",epoch)
                            # edit by erin 14Aug - adding model.save, torch.saves here to try to get state_dict in this case...
                            model.save(model_save_path)
                            print('saving to ',model_save_path)
                            torch.save(optimizer.state_dict(), model_save_path + '.optim')
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                        print('load previously best model and decay learning rate to %f' % lr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers')
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

            if epoch == args.max_epoch:
                print('reached maximum number of epochs!')
                print('time:', now)
                #writer.flush()
                exit(0)
                # EDIT BY ERIN - these were commented out (line 145-148):
            # print('finish one batch:{}'.format(time.time() - tic)) # This line is commented out by Ruohan on 9 Oct 2024
            # ADDED BY ERIN FOR TENSORBOARD - right place?
            #writer.flush()
    #train_data

    # print('Total Run time:{}'.format(time.time()-tic)) # This line is commented out by Ruohan on 9 Oct 2024
# ADDED BY ERIN FOR TENSORBOARD - right place?
#writer.flush()
