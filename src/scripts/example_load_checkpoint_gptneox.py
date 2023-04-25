import torch
import torch.nn as nn

from transformers import GPTNeoXForCausalLM

from transformers import AutoConfig, AutoTokenizer

from transformers.modeling_utils import no_init_weights
import os


def create_emtpy_gptneox(config):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = GPTNeoXForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def load_decentralized_checkpoint(model, checkpoint_path, n_stages=2, n_layer_per_stage=14):
    input_path = checkpoint_path

    assert n_stages * n_layer_per_stage >= len(model.gpt_neox.layers)
    # assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.gpt_neox.embed_in.weight.data[:] = _tmp['embed_in.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                model.gpt_neox.layers[j].load_state_dict(_tmp)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                if i*n_layer_per_stage + j == 44:
                    break
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.gpt_neox.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)

            _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
            if len(_tmp) == 0:
                break
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.gpt_neox.final_layer_norm.weight.data[:] = _tmp['final_layer_norm.weight']
            model.gpt_neox.final_layer_norm.bias.data[:] = _tmp['final_layer_norm.bias']
            model.embed_out.weight.data[:] = _tmp['embed_out.weight']
            if 'lm_head.bias' in _tmp:
                model.embed_out.bias.data[:] = _tmp['embed_out.bias']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.gpt_neox.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)

    return model


if __name__ == '__main__':

    config = AutoConfig.from_pretrained('EleutherAI/gpt-neo-1.3B')
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model = create_emtpy_gptneox(config)
    load_decentralized_checkpoint(model, 'model_checkpoints/gptneo-baseline/checkpoint_2000', n_stages=2, n_layer_per_stage=12)

    # test on cpu
    ret = model.generate(**tokenizer('you are not', return_tensors='pt'), max_new_tokens=4)

    print(tokenizer.batch_decode(ret))
