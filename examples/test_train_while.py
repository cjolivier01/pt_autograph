"""
Simple model runner for PyTorch
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
        system="Linux",
        node="chriso-monster",
        release="5.3.0-62-generic",
        version="#56~18.04.1-Ubuntu SMP Wed Jun 24 16:17:03 UTC 2020",
        machine="x86_64",
        processor="",
    )


_preempt_uname()

# BOILERPLATE
import os
import time
import sys
import logging
import numpy as np

# TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PTXLA
# import torch_xla
# import torch_xla.distributed.data_parallel as dp
# import torch_xla.utils.utils as xu
# import torch_xla.core.xla_model as xm
# import torch_xla.test.test_utils as test_utils
# from torch_xla.debug.graph_saver import save_tensors_graph

# PT_AUTOGRAPH
#import pt_autograph
#from pt_autograph.flow.runner import ag_function
from pt_autograph import ag_function

# PT_AUTOGRAPH.PTXLA
# import pt_autograph.ptxla.scope
# import pt_autograph.ptxla.stats as stats

# TEST UTILS (from ptxla)
import args_parse

FLAGS = args_parse.parse_common_options(
    datadir="/tmp/mnist-data",
    batch_size=4,
    momentum=0.5,
    lr=0.01,
    target_accuracy=98.0,
    num_epochs=2,
)

FLAGS.steps_per_epoch = 50
FLAGS.run_test = False
FLAGS.step_print_interval = 10
FLAGS.use_autograph = True
#FLAGS.use_autograph = False
FLAGS.with_while = False
FLAGS.with_if = True
FLAGS.log_steps = 1
FLAGS.use_fx = False
FLAGS.save_graph = True

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
        if FLAGS.with_if:
            if counter < 4:
                counter = counter + 1
            else:
                counter = counter - 1
            x = x + counter
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
    print(f"Getting loss took {dt} seconds")


def _save_checkpoint(args, device, step, model, is_epoch=False):
    # if is_epoch:
    #     xm.master_print(f"Saving checkpoint at end of epoch")
    # else:
    #     xm.master_print(f"Saving checkpoint as step closure of step : {step}")
    file_name = f"test_train_mnist_cpk_{step}.mdl"
    # xm.save(model, file_name)
    # xm.master_print('done...')
    # xm.master_print(f"Checkpoint saved for device: {device}")


def fake_dataset(sample_count, batch_size, dims, dtype):
    counter = 0
    while counter < sample_count:
        yield torch.ones(dims, dtype=dtype), torch.ones(batch_size, dtype=torch.int64)
        counter += 1


def train_mnist(FLAGS):

    DTYPE = torch.float32

    torch.manual_seed(1)

    dims = (FLAGS.batch_size, 1, 784)

    train_dataset_len = FLAGS.steps_per_epoch if FLAGS.steps_per_epoch else 60000
    train_loader = fake_dataset(
        train_dataset_len, FLAGS.batch_size, dims=dims, dtype=DTYPE
    )
    # train_loader = xu.SampleGenerator(
    #     data=(
    #         torch.ones(
    #             dims,
    #             dtype=DTYPE,
    #         ),
    #         torch.ones(
    #             FLAGS.batch_size,
    #             dtype=torch.int64,
    #         ),
    #     ),
    #     sample_count=train_dataset_len // FLAGS.batch_size //
    #     xm.xrt_world_size(),
    # )

    # devices = (xm.get_xla_supported_devices(
    #     max_devices=FLAGS.num_cores) if FLAGS.num_cores != 0 else [])

    # # Scale learning rate to num cores
    # lr = FLAGS.lr * max(len(devices), 1)
    lr = FLAGS.lr

    model = MNIST(FLAGS)
    # model_parallel = dp.DataParallel(
    #     model,
    #     device_ids=devices,
    # )

    writer = None
    # if xm.is_master_ordinal():
    #     writer = test_utils.get_summary_writer(FLAGS.logdir)

    #
    # Just some step closure output
    #
    def train_output_fn(outputs, ctx, args, tracker):
        if ctx.step > 0 and args.log_steps and ctx.step % args.log_steps == 0:
            now_time = time.time()
            if hasattr(ctx, "start_time") and ctx.start_time:
                per_step_time = (now_time - ctx.start_time) / (
                    ctx.step - ctx.last_step_timed
                )
                steps_per_second = 1 / per_step_time
                print(
                    f"[{xm.get_ordinal()}] Round-trip step time: "
                    f"{per_step_time} seconds, steps per second: {steps_per_second}"
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
                print(f"BEGIN Train step {ctx.step}")
                ctx.start_time = time.time()
                ctx.last_step_timed = ctx.step
            else:
                ctx.start_time = time.time()
                ctx.last_step_timed = ctx.step
        ctx.step += 1

    #
    # Train Epoch Function
    #
    def train_loop_fn(model, loader, device=None, context=None):
        lr_adder = 0.0

        loss_fn = nn.NLLLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr + lr_adder,
            momentum=FLAGS.momentum,
        )

        # tracker = xm.RateTracker()

        model.train()
        loss = None

        #
        # Train Step Function
        #
        @ag_function(enabled=FLAGS.use_autograph)
        def train_inner_loop_fn(batch, ctx):
            step = ctx.step
            print(f"Step {step}")
            data = batch[0]
            target = batch[1]
            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            # xm.optimizer_step(
            #     optimizer,
            #     barrier=False,
            # )

            # if (FLAGS.log_steps != 0
            #         and (FLAGS.log_steps == 1 or
            #              (step > 0 and step % FLAGS.log_steps == 0))):
            # xm.add_step_closure(
            #     _train_update,
            #     args=(device, step, loss, tracker, epoch, writer),
            # )

            # if step == 0:
            #     xm.master_print(f"End TRAIN step {step}")

            ctx.step += 1
            print(f"loss is on device: {loss.device}")

            if FLAGS.save_graph and ctx.step == 2:
                tensors = [loss] + list(model.parameters())
                # save_tensors_graph(os.getcwd(), "loss", tensors)

            return [loss]

        #
        # Train Step Loop
        #
        print("Starting new epoch train loop... (epoch={epoch})")
        for step, (data, target) in enumerate(loader):
            # if step % FLAGS.step_print_interval == 0:
            #     xm.master_print(f"Begin TRAIN Step: {step}")
            context.step = step

            # if FLAGS.use_fx:
            #     assert not FLAGS.use_autograph
            #     assert False  # Will do this shortly
            # elif FLAGS.use_autograph:
            #     outputs = pt_autograph.flow.runner.maybe_run_converted(
            #         train_inner_loop_fn, (data, target), context
            #     )
            # else:
            outputs = train_inner_loop_fn((data, target), context)

        # xm.master_print(f"Saving model...")
        _save_checkpoint(FLAGS, device, None, model, is_epoch=True)
        # xm.master_print(f"Model saved")
        return loss

    #
    # Epoch loop
    #
    for epoch in range(1, FLAGS.num_epochs + 1):
        # device = xm.xla_device()
        device = "cpu"

        class Context(object):
            pass

        ctx = Context()
        # ctx = dp.Context(device=device)
        # ctx.tracker = xm.RateTracker()
        ctx.step = 0
        loss = train_loop_fn(model, train_loader, device=device, context=ctx)

    # test_utils.close_summary_writer(writer)
    return loss


def main(args):
    import argparse
    import traceback

    # os.environ[
    #     'XRT_DEVICE_MAP'] = 'CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0'
    # os.environ['XRT_WORKERS'] = 'localservice:0;grpc://localhost:40934'
    # os.environ['SAVE_GRAPH_FMT'] = 'dot'

    try:
        train_mnist(FLAGS)
        print("Exiting...")

    except Exception as e:
        msg = f"Unhandled exception: {e}"
        print(msg)
        traceback.print_exc()
        time.sleep(1)  # Let any async stuff chill for a bit
        logging.getLogger().error(msg)
        # stats.print_stats(
        #     prefix_text="%ERROR_STATS_BEGIN%",
        #     suffix_text="%ERROR_STATS_END%",
        #     include_metrics=True,
        # )
        raise


# Run the tests.
if __name__ == "__main__":
    main(sys.argv[1:])
