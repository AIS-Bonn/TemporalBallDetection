import torch
from matplotlib import pyplot as plt
import os as os
from arguments import opt
from py_utils.utils_multi import post_processing, tp_fp_tn_fn_alt, performance_metric
from py_utils.logging_setup import logger
from py_utils.util_functions import Averaging
import re

import h5py

from py_utils.util_functions import dir_check


class ModelEvaluator:
    def __init__(self, model, min_radius, threshold, optim_seq, optim_both, loss):
        '''
        model: instance of pytorch model class
        epochs: number of training epochs
        lr: learning rate
        use_gpu: to use gpu
        optim: optimizer used for training, SGD or adam
        '''
        self.model = model
        self.epochs = opt.nm_epochs
        self.lr = opt.lr
        self.use_gpu = True if opt.device == 'cuda' else False
        self.batch_size = opt.batch_size
        self.train_loss = []
        self.test_loss = []
        self.iter_loss_train = []
        self.iter_loss_test = []
        self.loss = loss
        self.min_radius = min_radius
        self.threshold = threshold
        self.fdr_train = []
        self.RC_train = []
        self.accuracy_train = []
        self.fdr_test = []
        self.RC_test = []
        self.accuracy_test = []        
        if self.use_gpu:
            self.model.cuda()

        self.optim_seq = optim_seq
        self.optim_both = optim_both
        self.both = False

    def train(self, epoch, trainloader, print_every=100):
        '''
        method for training
        '''
        self.model.train()
        self.model = self.model.cuda()
        losses = Averaging()
        loss_batch = 0
        # TP, FP, FN, TN = 0, 0, 0, 0
        for b_idx, (train_data, train_labels) in enumerate(trainloader):
            train_data = train_data.float()
            train_labels = train_labels.float()
            if opt.real_balls:
                train_data = train_data.squeeze()
                train_labels = train_labels.view(-1, opt.seq_predict, opt.map_size_x, opt.map_size_y)
            if self.use_gpu:
                train_data = train_data.cuda(non_blocking=True)
                train_labels = train_labels.cuda()
            if opt.seq_model == 'lstm':
                output, (h, cc) = self.model(train_data)
                loss = self.loss(output[0], train_labels)
            if opt.seq_model == 'tcn':
                output = self.model(train_data)
                loss = self.loss(output, train_labels)

            losses.update(loss.item(), train_data.size(0))

            self.model.zero_grad()
            loss.backward()
            if self.both:
                self.optim_both.step()
            else:
                self.optim_seq.step()

            if b_idx % opt.print_every == 0:
                # logger.debug('%s | %s' % (str(train_labels), str(output)))
                logger.debug('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\t Loss {4:<10.3f} \ {5:>10.3f}'.
                             format(epoch, b_idx * len(train_data),
                                    len(trainloader) * len(train_data),
                                    100. * b_idx / len(trainloader), loss, losses.avg))

            loss_ = loss.item()
            self.iter_loss_train.append(loss_)
            loss_batch += loss_
        losses.reset()

        loss_batch /= len(trainloader)

        logger.debug('Epoch = {} '.format(epoch))
        logger.debug('Train loss = {0}'.format(loss_batch))
        self.train_loss.append(loss_batch)

    def test(self,model,  testloader):
        '''
        method for testing
        '''
        self.model.eval()
        TP, FP, FN, TN = 0, 0, 0, 0
        l = []
        with torch.no_grad():
            batch_loss = 0
            for idx, (test_data, test_labels, actual_centers, path) in enumerate(testloader):
                if 'new' in opt.dataset or opt.dataset == 'multi':
                    centers = []
                    for x, y in zip(actual_centers[0], actual_centers[1]):
                        # centers.append((x, y))
                        centers.append((x.numpy(), y.numpy()))
                    actual_centers = centers
                if self.use_gpu:
                    test_data, test_labels = test_data.cuda(), test_labels.cuda()
                # output = self.model(test_data)
                output = self.model.test(test_data)
                if len(output.shape) < 3:
                    output = output.unsqueeze(0)
                loss_ = self.loss(output, test_labels.float())
                # output = output.cpu().squeeze()
                output = output.cpu()

                out, predicted_centers, maps_area = post_processing(output.numpy(), self.threshold)
                TP_test, FP_test, TN_test, FN_test = tp_fp_tn_fn_alt(actual_centers, predicted_centers, maps_area,
                                                                     self.min_radius)
                TP += TP_test
                FP += FP_test
                FN += FN_test
                TN += TN_test

                # if 'new' in opt.dataset or opt.dataset == 'multi':
                #     if opt.batch_size > 1:
                #         path = path[0]
                #         output = output[0]
                #     n = int(re.search(r'frame(\d*)', path).group(1))
                #     plt.axis('off')
                #     output = output.squeeze().numpy()
                #     output[0,0] = 1
                #     img = plt.imshow(output)
                #     dir_check(opt.save_out)
                #     dir_check(os.path.join(opt.save_out, opt.seq_model))
                #     dir_check(os.path.join(opt.save_out, opt.seq_model, opt.suffix))
                #     plt.savefig(os.path.join(opt.save_out, opt.seq_model, opt.suffix, '%d_sweaty_output.png'%n))

                self.iter_loss_test.append(loss_)
                batch_loss += loss_

            batch_loss /= len(testloader)
            FDR_test, RC_test, accuracy_test = performance_metric(TP, FP, FN, TN)

            self.fdr_test.append(FDR_test)
            self.accuracy_test.append(accuracy_test)
            self.RC_test.append(RC_test)

            logger.debug('model {} Test TP {} FP {} TN {} FN {}'.format(model, TP, FP, TN, FN))
            logger.debug('Test loss = {0} FDR = {1:.4f} , RC {2:.4f} =, accuracy = {3:.4f}'.format(batch_loss, FDR_test,
                                                                                            RC_test, accuracy_test))

            self.test_loss.append(batch_loss)

    def evaluator(self, trainloader, testloader, print_every=1000, both=0):
        '''
        train and validate model
        '''
        resume_epoch = 0
        logger.debug('Model')
        self.model.off_sweaty()
        for epoch in range(resume_epoch, self.epochs):
            if epoch == both:
                self.both = True
                self.model.on_sweaty()
            self.test(epoch, testloader)
            self.train(epoch, trainloader, print_every=print_every)
            logger.debug('threshold: %s' % str(self.threshold))
            if epoch % opt.save_every==0:
                save_model = {'threshold': self.threshold,
                              'epoch': epoch,
                              'state_dict_model': self.model.state_dict()}
                model_name = '{}_lr_{}_opt_{}_epoch_{}'.format(opt.seq_save_model,
                                                               self.lr, self.optim, epoch)
                model_dir = opt.model_root + '/' + model_name
                torch.save(save_model, model_dir)

    def plot_loss(self):
        '''
        to visualize loss
        '''
        plt.plot(range(len(self.train_loss)), self.train_loss,
                 label='Training Loss')
        plt.plot(range(len(self.test_loss)), self.test_loss,
                     label='Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('{}/loss_evaluation_epoch_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
        plt.cla()
        plt.clf()
        plt.plot(range(len(self.iter_loss_train)), self.iter_loss_train,
                 label='Training Loss')
        plt.plot(range(len(self.iter_loss_test)), self.iter_loss_test,
                     label='Testing Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('{}/loss_evaluation_iter_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
        plt.cla()
        plt.clf()
        # plt.plot(range(len(self.fdr_train)), self.fdr_train,
        #          label='Training FDR')
        plt.plot(range(len(self.fdr_test)), self.fdr_test,
                     label='Testing FDR')
        plt.xlabel('Epoch')
        plt.ylabel('FDR')
        plt.legend()
        plt.savefig('{}/FDR_evaluation_epoch_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))   
        plt.cla()
        plt.clf()
        #
        # plt.plot(range(len(self.RC_train)), self.RC_train,
        #          label='Training RC')
        plt.plot(range(len(self.RC_test)), self.RC_test,
                     label='Testing RC')
        plt.xlabel('Epoch')
        plt.ylabel('RC')
        plt.legend()
        plt.savefig('{}/rc_evaluation_epoch_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
        plt.cla()
        plt.clf()
        # plt.plot(range(len(self.accuracy_train)), self.accuracy_train,
        #          label='Training Acc')
        plt.plot(range(len(self.accuracy_test)), self.accuracy_test,
                     label='Testing Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('{}/accuracy_evaluation_epoch_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
                 
    def load_model(self, model_name):
        '''
        load model checkpoint
        '''
        model_dir = opt.model_root + '/' + model_name
        checkpoint = torch.load(model_dir)
        model, epoch = checkpoint['state_dict_model'], checkpoint['epoch']
        return model, epoch

    def save_output(self):
        '''
        save results
        '''
        filename = '{}/evaluation_epoch_{}_drop_{}.h5'.format(opt.result_root, opt.net, opt.drop_p)
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('RC_train', data = self.RC_train)
            hf.create_dataset('RC_test', data = self.RC_test)
            hf.create_dataset('fdr_train', data = self.fdr_train)
            hf.create_dataset('fdr_test', data = self.fdr_test)
            hf.create_dataset('accuracy_train', data = self.accuracy_train)
            hf.create_dataset('accuracy_test', data = self.accuracy_test)