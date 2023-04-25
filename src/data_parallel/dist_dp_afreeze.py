import os

import cupy
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from comm.comm_utils import *
from compress import flag
from compress.fixpoint import *
from compress.sparsification import *

from .flatten_utils import flatten_params, flatten_tensors

if 'SYNC_STEPS' not in os.environ:
    sync_steps = 25
    sync_prob = 1.0 / sync_steps
    global_sync_steps = 1
else:
    sync_steps = int(os.environ['SYNC_STEPS'])
    sync_prob = 1.0 / sync_steps
    global_sync_steps = 1

global_lr = float(os.environ.get('GLOBAL_LR', 1.0))
quantization_bits = int(os.environ.get('QUANT_BITS', 8))
quantization_bucket_size = int(os.environ.get('QUANT_BUCKET_SIZE', 128))
top_k_ratio = float(os.environ.get('TOPK_RATIO', 0.5))


class AFreezeDP:

    def __init__(self,
                 args,
                 device,
                 module: torch.nn.Module,
                 optimizer: torch.optim.Optimizer = None,
                 flatten=False):
        assert not flatten
        self.dp_bits = args.dp_bits
        self.flatten = flatten
        self.global_rank = args.rank
        self.dp_group_size = args.data_group_size
        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.dp_comm = get_data_parallel_comm()
        self.dp_rank = get_data_parallel_rank()
        self.pp_comm = get_pipeline_parallel_comm()
        self.pp_rank = get_pipeline_parallel_rank()
        self.dp_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_optim_comp_stream = torch.cuda.default_stream(device=device)
        self.backward_ready_event = torch.cuda.Event(
            enable_timing=self.enable_tidy_profiling, blocking=False)
        self.sync_gradients_ready_event = torch.cuda.Event(
            enable_timing=self.enable_tidy_profiling, blocking=False)
        self.optimizer_step_ready_event = torch.cuda.Event(
            enable_timing=self.enable_tidy_profiling, blocking=False)

        self.mmt_start_event = torch.cuda.Event(
            enable_timing=self.enable_tidy_profiling, blocking=False)
        self.mmt_ready_event = torch.cuda.Event(
            enable_timing=self.enable_tidy_profiling, blocking=False)

        self.module = module
        assert optimizer is not None
        self.optimizer = optimizer
        num_paras, element_size = self._compute_total_para_num()
        print(
            "Total number of parameters: {}, element size: {}, total size {} MB."
            .format(num_paras, element_size,
                    num_paras * element_size // 1024 // 1024))

        if self.enable_tidy_profiling:
            self.global_rank = args.rank
            self.init_event = None
            self.init_time_stamp = None

            # assert self.flatten
            self.sync_gradients_start_event = torch.cuda.Event(
                enable_timing=True, blocking=False)
            self.optimizer_step_start_event = torch.cuda.Event(
                enable_timing=True, blocking=False)

            self.gather_start_event = torch.cuda.Event(enable_timing=True,
                                                       blocking=False)
            self.sync_start_event = torch.cuda.Event(enable_timing=True,
                                                     blocking=False)
            self.gather_end_event = torch.cuda.Event(enable_timing=True,
                                                     blocking=False)
            self.sync_end_event = torch.cuda.Event(enable_timing=True,
                                                   blocking=False)

            self.worker_compress_start_event = torch.cuda.Event(
                enable_timing=True, blocking=False)
            self.server_compress_start_event = torch.cuda.Event(
                enable_timing=True, blocking=False)
            self.worker_compress_end_event = torch.cuda.Event(
                enable_timing=True, blocking=False)
            self.server_compress_end_event = torch.cuda.Event(
                enable_timing=True, blocking=False)

        self.dp_state_dict = {}

    def _compute_total_para_num(self):
        total_count = 0
        element_size = 0
        for para in self.module.parameters():
            # print("Parameter: ", para.data.shape)
            total_count += torch.numel(para.data)
            element_size = para.element_size()
        return total_count, element_size

    def profile_mark_sync_grad_start(self):
        if self.enable_tidy_profiling:
            self.dp_comm_stream.record_event(self.sync_gradients_start_event)

    def profile_mark_allreduce_end(self):
        pass

    def profile_mark_optimizer_step_start(self):
        if self.enable_tidy_profiling:
            self.torch_optim_comp_stream.record_event(
                self.optimizer_step_start_event)

    def allreduce_parameters(self):
        self._local_parameters_backup = [
            p.data.clone() for p in self.module.parameters()
        ]
        torch.cuda.synchronize()
        self.dp_comm.barrier()
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(
                self.dp_comm_stream.cuda_stream)
            # self.dp_comm_stream.wait_event(self.backward_ready_event)
            for name, para in self.module.named_parameters():
                # self.profile_mark_allreduce_start(name)
                para.data /= self.dp_group_size
                self.dp_comm.all_reduce(para.data, stream=cupy_dp_stream)
                # self.profile_mark_allreduce_end(name)
            # self.dp_comm_stream.record_event(self.allreduce_grad_ready_event)
        torch.cuda.synchronize()
        self.dp_comm.barrier()

    def rollback_parameters(self):
        if not hasattr(self, '_local_parameters_backup'):
            return

        for p, p_local in zip(self.module.parameters(),
                              self._local_parameters_backup):
            p.data[:] = p_local.data

        del self._local_parameters_backup

    def _allreduce_gradients(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(
                self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            if self.flatten:
                # self.profile_mark_allreduce_start()
                self.dp_comm.all_reduce(self.flatten_para.grad,
                                        stream=cupy_dp_stream)
                self.profile_mark_allreduce_end()
            else:
                for name, para in self.module.named_parameters():
                    if para.grad is None:
                        continue
                    # self.profile_mark_allreduce_start(name)
                    self.dp_comm.all_reduce(para.grad, stream=cupy_dp_stream)
                    # self.profile_mark_allreduce_end(name)
            self.dp_comm_stream.record_event(self.sync_gradients_ready_event)

    def _compress(self, x):
        # return x
        dtype = x.dtype
        shape = x.shape
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(
                self.dp_comm_stream.cuda_stream)
            with cupy_dp_stream:

                k = max(int(top_k_ratio * x.numel()), 1)
                if k >= quantization_bucket_size:
                    # ensure dividable
                    k = k // quantization_bucket_size * quantization_bucket_size
                else:
                    # bucket_size will be set to k internally
                    pass

                values, masks, indices = compress_topk(x,
                                                       k,
                                                       return_indices=True)

                # if k > 10 and self.dp_rank==0:
                #     print('indices:', indices)

                values_q, scales_q = compress_flexible_nbits_by_bucket(
                    values,
                    bits=quantization_bits,
                    scale_method='max',
                    bucket_size=quantization_bucket_size)

                # if k > 10 and self.dp_rank==0:
                #     print('integers:', unpack_low_bit_tensor(values_q, quantization_bits, values.shape))

                values = decompress_flexible_nbits_by_bucket(
                    values_q,
                    scales_q,
                    bits=quantization_bits,
                    original_shape=values.shape,
                    bucket_size=quantization_bucket_size)

                x = decompress_topk(values, masks, x.shape)
                x = x.to(dtype)

        return x

    def _decompress(self, x_hat):
        return x_hat

    def _partial_sync(self):

        if self.flatten:

            with torch.cuda.stream(self.dp_comm_stream):
                cupy_dp_stream = cupy.cuda.ExternalStream(
                    self.dp_comm_stream.cuda_stream)

                name = 'model'
                para = self.flatten_para

                if name not in self.comm_mask_dict:
                    self.comm_mask_dict[name] = torch.zeros_like(
                        para, dtype=torch.bool)
                    self.comm_mask_dict[name].view(-1)[::sync_steps] = True
                    self.buffer_dict[name] = torch.zeros(
                        self.comm_mask_dict[name].sum().item(),
                        dtype=torch.float16,
                        device=para.device)
                else:
                    self.comm_mask_dict[name] = self.comm_mask_dict[name].roll(
                        1)

                comm_mask = self.comm_mask_dict[name]
                buffer = self.buffer_dict[name]
                buffer.data[:] = (para[comm_mask] / self.dp_group_size).half()
                # print(buffer.data.numel() / para.data.numel())
                print('not implemented yet, just do allreduce')
                self.dp_comm.all_reduce(buffer)

                self.dp_comm_stream.record_event(
                    self.sync_gradients_ready_event)

        else:
            cupy_dp_stream = cupy.cuda.ExternalStream(
                self.dp_comm_stream.cuda_stream)
            with torch.cuda.stream(self.dp_comm_stream), cupy_dp_stream:

                # for name, para in self.module.named_parameters():
                for i_group, group in enumerate(
                        self.optimizer.optimizer.param_groups):
                    for i_para, para in enumerate(group["params"]):

                        name = f"{i_group}-{i_para}"

                        dp_state_dict = self.dp_state_dict

                        if name not in dp_state_dict:

                            comm_mask = torch.zeros_like(para,
                                                         dtype=torch.bool)
                            comm_mask.view(-1)[::sync_steps] = True
                            if not comm_mask.any().item():
                                comm_mask[:] = True

                            dp_state_dict[name] = {
                                "comm_mask": comm_mask,
                            }

                        comm_mask = dp_state_dict[name]["comm_mask"]

                        data = para.data[comm_mask]
                        # print(data.numel() / para.numel())
                        data /= self.dp_group_size
                        self.dp_comm.all_reduce(data)
                        para.data[comm_mask] = data
                        dp_state_dict[name]["comm_mask"] = dp_state_dict[name][
                            "comm_mask"].roll(1)

                self.dp_comm_stream.record_event(
                    self.sync_gradients_ready_event)


#     def _copy_to_model(self):

#         if self.flatten:
#             name = 'model'
#             para = self.flatten_para
#             if name in self.comm_mask_dict:
#                 comm_mask = self.comm_mask_dict[name]
#                 buffer = self.buffer_dict[name]
#                 para.data[comm_mask] = buffer.to(para.dtype)
#         else:
#             for name, para in self.module.named_parameters():
#                 if name not in self.comm_mask_dict:
#                     continue
#                 comm_mask = self.comm_mask_dict[name]
#                 buffer = self.buffer_dict[name]
#                 para.data[comm_mask] = buffer.to(para.dtype)

    def optimizer_step(self):
        if flag.FLAG_DISABLE_COMPRESSION:
            self._allreduce_gradients()
        else:
            self._partial_sync()
        with torch.cuda.stream(self.torch_optim_comp_stream):
            self.torch_optim_comp_stream.wait_event(
                self.sync_gradients_ready_event)
            self.profile_mark_optimizer_step_start()
            # self._copy_to_model()
            self.optimizer.step()
            self.torch_optim_comp_stream.record_event(
                self.optimizer_step_ready_event)

    def set_time_stamp(self, init_time_stamp, init_event):
        self.init_event = init_event
        self.init_time_stamp = init_time_stamp

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(
            event) * 1e+3

    def profiling_data_parallel(self, init_time_stamp, init_event):
        self.set_time_stamp(init_time_stamp, init_event)
        profiling_log = []

        # assert self.flatten
        allreduce_slot = self.sync_gradients_start_event.elapsed_time(
            self.sync_gradients_ready_event) * 1e+3
        allreduce_log = {
            "name": "opt_shardedPS_sync",
            "ph": "X",
            "pid": self.global_rank,
            "tid": "7. optimizer-comm",
            "ts": self.get_ts(self.sync_gradients_start_event),
            "dur": allreduce_slot,
            "cname": "cq_build_passed",
            "args": {
                'para': 'flattened_grad',
                'size': self.flatten_para.grad.numel()
            }
        }
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        optimizer_slot = self.optimizer_step_start_event.elapsed_time(
            self.optimizer_step_ready_event) * 1e+3
        optimizer_log = {
            "name": "opt_comp",
            "ph": "X",
            "pid": self.global_rank,
            "tid": "8. optimizer-comp",
            "ts": self.get_ts(self.optimizer_step_start_event),
            "dur": optimizer_slot,
            "cname": "bad"
        }
        # print(optimizer_log)
        profiling_log.append(optimizer_log)

        allreduce_slot = self.gather_start_event.elapsed_time(
            self.gather_end_event) * 1e+3
        allreduce_log = {
            "name": "gather grads",
            "ph": "X",
            "pid": self.global_rank,
            "tid": "9. optimizer-comm",
            "ts": self.get_ts(self.gather_start_event),
            "dur": allreduce_slot,
            "cname": "cq_build_passed",
            "args": {
                'para': 'flattened_grad',
                'size': self.flatten_para.grad.numel()
            }
        }
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        allreduce_slot = self.sync_start_event.elapsed_time(
            self.sync_end_event) * 1e+3
        allreduce_log = {
            "name": "distribute grads",
            "ph": "X",
            "pid": self.global_rank,
            "tid": "10. optimizer-comm",
            "ts": self.get_ts(self.sync_start_event),
            "dur": allreduce_slot,
            "cname": "cq_build_passed",
            "args": {
                'para': 'flattened_grad',
                'size': self.flatten_para.grad.numel()
            }
        }
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        allreduce_slot = self.worker_compress_start_event.elapsed_time(
            self.worker_compress_end_event) * 1e+3
        allreduce_log = {
            "name": "worker compress",
            "ph": "X",
            "pid": self.global_rank,
            "tid": "11. optimizer-comm",
            "ts": self.get_ts(self.worker_compress_start_event),
            "dur": allreduce_slot,
            "cname": "cq_build_passed",
            "args": {
                'para': 'flattened_grad',
                'size': self.flatten_para.grad.numel()
            }
        }
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        allreduce_slot = self.server_compress_start_event.elapsed_time(
            self.server_compress_end_event) * 1e+3
        allreduce_log = {
            "name": "server compress",
            "ph": "X",
            "pid": self.global_rank,
            "tid": "12. optimizer-comm",
            "ts": self.get_ts(self.server_compress_start_event),
            "dur": allreduce_slot,
            "cname": "cq_build_passed",
            "args": {
                'para': 'flattened_grad',
                'size': self.flatten_para.grad.numel()
            }
        }
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        return profiling_log
