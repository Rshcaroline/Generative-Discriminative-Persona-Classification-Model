from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim
from torch.autograd import Variable

from ..evaluator import Evaluator
from ..loss import NLLLoss
from ..optim import Optimizer
from ..util.checkpoint import Checkpoint

from ..dataset import spk_field_name, tgt_field_name, src_field_name

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2mlp.loss.loss.Loss, optional): loss for training, (default: seq2mlp.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
    """
    def __init__(self, args, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100):
        self.args = args
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.cuda = args.cuda

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, spk_input, model, teacher_forcing_ratio):
        loss = self.loss
        # Forward propagation
        results = model(input_variable, input_lengths, teacher_forcing_ratio=teacher_forcing_ratio)

        # Get loss
        loss.reset()
        # loss = criterion(results, Variable(spk_input))
        for step, step_output in enumerate(results):
            batch_size = step_output.size(0)
            labels = torch.LongTensor(step_output.size()).zero_()
            if self.cuda:
                labels = torch.cuda.LongTensor(step_output.size()).zero_()
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), Variable(spk_input.data[step,:]))  # Variable(labels)

        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        device = None if self.args.cuda else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator) # Q: just 15,084 conversations
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step): # Q: don't understand
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1
                input_variables, input_lengths = getattr(batch, src_field_name)
                input_variables, input_lengths = [input_variables], [input_lengths.tolist()]
                if hasattr(batch, tgt_field_name):
                    input_variable, input_length = getattr(batch, tgt_field_name)
                    input_variables.append(input_variable)
                    input_lengths.append(input_length.tolist())

                spk_input = getattr(batch, spk_field_name)

                loss = self._train_batch(input_variables, input_lengths, spk_input, model, teacher_forcing_ratio) # tokenization about symbols

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                assert step == step_elapsed

                if step % self.print_every == 0: # and step_elapsed >= self.print_every: # bug
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)

                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=data.fields[src_field_name].vocab,
                               output_vocab=data.fields[spk_field_name].vocab).save(self.expt_dir)

            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            _, accuracy = self.evaluator.evaluate(model, data)
            log_msg = "Finished epoch %d: Train %s: %.4f, Accuracy: %.4f" % (
                                    epoch, self.loss.name, epoch_loss_avg, accuracy)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

        Args:
            model (seq2mlp.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2mlp.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2mlp.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2mlp.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2mlp.models): trained model.
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model
