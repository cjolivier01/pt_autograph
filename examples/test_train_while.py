"""
Configurable simple model runner for PyTorch
Based upon the train_test_mnist.py script from the pytorch/xla package
"""

def _preempt_uname():
    """
    Avoid a fork() when calling uname() in order to aid gdb
    Importing torch loads pltform and calls uname() in order to determine if the
    system is Windows.  This causes a fork, which makes debugging tedious
    for multi-processing workflows since a trap has to occur in order to manually
    set follow-fork-mode to child *after* uname is called, or else gdb can't
    debug the forked ptxla processes (following child follows the uname() fork)
    
    ...at least how I have mine configured :)

    """
    import platform

    platform._uname_cache = platform.uname_result(
        system='Linux',
        node='chriso-monster',
        release='5.3.0-62-generic',
        version='#56~18.04.1-Ubuntu SMP Wed Jun 24 16:17:03 UTC 2020',
        machine='x86_64',
        processor='',
    )


_preempt_uname()

import os
from statistics import mean
import time
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.fx as fx
from torchvision import datasets, transforms
import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.metrics as met
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import unittest
import shutil
import logging

import ptwse
import ptwse.scope
import ptwse.flow.runner as runner
import ptwse.stats as stats


import args_parse

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/mnist-data',
    batch_size=4,
    momentum=0.5,
    lr=0.01,
    target_accuracy=98.0,
    num_epochs=2)


FLAGS.steps_per_epoch = 50
FLAGS.run_test = False
FLAGS.step_print_interval = 10
FLAGS.use_autograph = True
FLAGS.with_while = False
FLAGS.log_steps = 1
FLAGS.use_fx = False

class MNIST(nn.Module):
    def __init__(self, flags):
        super(MNIST, self).__init__()
        self._flags = flags
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(size=[-1, 784])
        x = F.relu(self.fc1(x))
        counter = torch.tensor(0, device=x.device)
        if self._flags.with_while:
            # Try a "while" statement
            while counter < 10:
                x = F.relu(x)
                counter = counter + 1

        # Try an "if" statement
        if counter < 4:
            counter = counter + 1
        else:
            counter = counter - 1
        return F.log_softmax(x, dim=1)


def _train_update(device, step, loss, tracker, epoch, writer):
    st = time.time()
    loss.item()
    dt = time.time() - st
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer,
    )
    print(f'Getting loss took {dt} seconds')


def _save_checkpoint(args, device, step, model, is_epoch=False):
    if is_epoch:
        xm.master_print(f"Saving checkpoint at end of epoch")
    else:
        xm.master_print(f"Saving checkpoint as step closure of step : {step}")
    file_name = f"test_train_mnist_cpk_{step}.mdl"
    xm.save(model, file_name)
    xm.master_print('done...')
    xm.master_print(f"Checkpoint saved for device: {device}")


def train_mnist(FLAGS):

    DTYPE = torch.float32

    torch.manual_seed(1)

    dims = (
        FLAGS.batch_size,
        1,
        784,
    )

    train_dataset_len = FLAGS.steps_per_epoch if FLAGS.steps_per_epoch else 60000
    train_loader = xu.SampleGenerator(
        data=(
            torch.ones(dims, dtype=DTYPE,),
            torch.ones(
                FLAGS.batch_size,
                dtype=torch.int64,
            ),
        ),
        sample_count=train_dataset_len
        // FLAGS.batch_size
        // xm.xrt_world_size(),
    )
    test_loader = xu.SampleGenerator(
        data=(
            torch.ones(dims, dtype=DTYPE,),
            torch.ones(
                FLAGS.batch_size,
                dtype=torch.int64,
            ),
        ),
        sample_count=10000 // FLAGS.batch_size // xm.xrt_world_size(),
    )

    devices = (
        xm.get_xla_supported_devices(max_devices=FLAGS.num_cores)
        if FLAGS.num_cores != 0
        else []
    )

    """ 
    Non multi-processing
    """
    # Scale learning rate to num cores
    lr = FLAGS.lr * max(len(devices), 1)

    model = MNIST(FLAGS)
    model_parallel = dp.DataParallel(
        model,
        device_ids=devices,
    )

    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)

    # Just some step closure output
    def train_output_fn(outputs, ctx, args, tracker):
        if ctx.step > 0 and args.log_steps and ctx.step % args.log_steps == 0:
            now_time = time.time()
            if hasattr(ctx, 'start_time') and ctx.start_time:
                per_step_time = (now_time - ctx.start_time) / (
                    ctx.step - ctx.last_step_timed
                )
                steps_per_second = 1 / per_step_time
                print(
                    f'[{xm.get_ordinal()}] Round-trip step time: '
                    f'{per_step_time} seconds, steps per second: {steps_per_second}'
                )
                if tracker:
                    _train_update(
                        device=device,
                        step=ctx.step,
                        loss=outputs[0],
                        tracker=tracker,
                        epoch=epoch,
                        writer=writer,
                    )
                print(f'BEGIN Train step {ctx.step}')
                ctx.start_time = time.time()
                ctx.last_step_timed = ctx.step
            else:
                ctx.start_time = time.time()
                ctx.last_step_timed = ctx.step
        ctx.step += 1

    def train_loop_fn(model, loader, device=None, context=None):
        lr_adder = 0.0

        loss_fn = nn.NLLLoss()
        optimizer = context.getattr_or(
            'optimizer',
            lambda: optim.SGD(
                model.parameters(),
                lr=lr + lr_adder,
                momentum=FLAGS.momentum,
            ),
        )

        tracker = xm.RateTracker()

        model.train()

        def train_inner_loop_fn(batch, ctx):
            step = ctx.step
            print(f'Step {step}')
            data = batch[0]
            target = batch[1]
            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, target)
            loss.backward()

            xm.optimizer_step(
                optimizer,
                barrier=False,
            )

            if (
                FLAGS.log_steps != 0
                and (
                    FLAGS.log_steps == 1
                    or (step > 0 and step % FLAGS.log_steps == 0)
                )
            ):
                xm.add_step_closure(
                    _train_update,
                    args=(device, step, loss, tracker, epoch, writer),
                )

            if step == 0:
                xm.master_print(f"End TRAIN step {step}")

            ctx.step += 1
            return [loss]

        step = 0
        # Train
        print('Starting new epoch train loop... (epoch={epoch})')
        for step, (data, target) in enumerate(loader):
            if step % FLAGS.step_print_interval == 0:
                xm.master_print(f"Begin TRAIN Step: {step}")
            context.step = step

            if FLAGS.use_fx:
                assert not FLAGS.use_autograph
                outputs = ptwse.flow.runner.maybe_run_converted(
                    train_inner_loop_fn,
                    (data, target),
                    context
                )
            elif FLAGS.use_autograph:
                outputs = ptwse.flow.runner.maybe_run_converted(
                    train_inner_loop_fn,
                    (data, target),
                    context
                )
            else:
                outputs = train_inner_loop_fn((data, target), context)

        xm.master_print(f"Saving model...")
        _save_checkpoint(FLAGS, device, None, model, is_epoch=True)
        xm.master_print(f"Model saved")

    def test_loop_fn(model, loader, device, context):
        print("***********************")
        print("ENTERING TEST FUNCTION")
        print("***********************")
        print('Evaluating...')
        total_samples = 0
        correct = 0
        model.eval()
        for step, (data, target) in enumerate(loader):
            if step >= FLAGS.test_max_step:
                break
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            if FLAGS.mp:
                correct += pred.eq(target.view_as(pred)).sum()
            else:
                correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size()[0]

        if FLAGS.mp:
            this_accuracy = 100.0 * correct.item() / total_samples
            print("CALLING: mesh_reduce('test_accuracy')")
            this_accuracy = xm.mesh_reduce(
                'test_accuracy', this_accuracy, np.mean
            )
            print("BACK FROM: mesh_reduce('test_accuracy')")
        else:
            this_accuracy = 100.0 * correct / total_samples
            test_utils.print_test_update(device, this_accuracy)
        print("***********************")
        print("LEAVING TEST FUNCTION")
        print("***********************")
        return this_accuracy

    #
    # Set up for
    #
    accuracy = 0.0

    num_devices = (
        len(xm.xla_replication_devices(devices)) if len(devices) > 1 else 1
    )

    if not FLAGS.steps_per_epoch:
        num_training_steps_per_epoch = train_dataset_len // (
            FLAGS.batch_size * num_devices
        )
    else:
        num_training_steps_per_epoch = FLAGS.steps_per_epoch
    max_accuracy = 0.0

    #
    # Epoch loop
    #
    for epoch in range(1, FLAGS.num_epochs + 1):
        #
        # Train
        #
        device = xm.xla_device()
        ctx = dp.Context(device=device)
        ctx.tracker = xm.RateTracker()
        ctx.step = 0
        train_loop_fn(model, train_loader, device, ctx)

        #
        # Test
        #
        if FLAGS.run_test:
            with ptwse.scope.proxy_disabled(disabled=FLAGS.test_off_proxy):
                accuracies = model_parallel(test_loop_fn, test_loader)
            accuracy = mean(accuracies)
            print(
                'Epoch: {}, Mean Accuracy: {:.2f}%'.format(epoch, accuracy)
            )

        global_step = (epoch - 1) * num_training_steps_per_epoch
        max_accuracy = max(accuracy, max_accuracy)

        test_utils.write_to_summary(
            writer,
            global_step,
            dict_to_write={'Accuracy/test': accuracy},
            write_xla_metrics=True,
        )

        if FLAGS.metrics_debug:
            xm.master_print(met.metrics_report())

    test_utils.close_summary_writer(writer)
    xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
    return max_accuracy


class TrainMnist(unittest.TestCase):
    def __init__(self, flags):
        self._flags = flags

    def tearDown(self):
        super(TrainMnist, self).tearDown()
        if self._flags.tidy and os.path.isdir(self._flags.datadir):
            shutil.rmtree(self._flags.datadir)

    def test_accurracy(self):
        if not self._flags.fake_data:
            self.assertGreaterEqual(train_mnist(), self._flags.target_accuracy)
        else:
            train_mnist()


def main(args):
    import argparse
    import traceback
    
    os.environ['XRT_DEVICE_MAP'] = 'CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0'
    os.environ['XRT_WORKERS'] = 'localservice:0;grpc://localhost:40934'

    try:
        train_mnist(FLAGS)
        print("Exiting...")

    except Exception as e:
        msg = f"Unhandled exception: {e}"
        print(msg)
        traceback.print_exc()
        time.sleep(1)  #
        logging.getLogger().error(msg)
        stats.print_stats(
            prefix_text="%ERROR_STATS_BEGIN%",
            suffix_text="%ERROR_STATS_END%",
            include_metrics=True,
        )
        raise


# Run the tests.
if __name__ == '__main__':
    main(sys.argv[1:])
