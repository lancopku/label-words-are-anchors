import warnings

import torch


class Predictor:
    def __init__(self, label_id_dict, pad_token_id, task_name, tokenizer, layer,
                 naive_class_embs=None,
                 naive_final_emb=None) -> None:
        self.naive_class_embs = naive_class_embs
        self.naive_final_emb = naive_final_emb
        self.label_id_dict = label_id_dict
        self.pad_token_id = pad_token_id
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.layer = layer

        if task_name == 'sst2':
            self.prefix_idxs = [tokenizer.encode('Sentiment', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        elif task_name == 'agnews':
            self.prefix_idxs = [tokenizer.encode('Answer', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        elif task_name == 'trec':
            self.prefix_idxs = [tokenizer.encode(' Type', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        elif task_name == 'emo':
            self.prefix_idxs = [tokenizer.encode('Emotion', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        else:
            raise NotImplementedError(f"task_name: {task_name}")

    def get_pos(self, inputs):
        label_id_dict = self.label_id_dict
        pad_token_id = self.pad_token_id
        final_pos = (inputs['input_ids'] != pad_token_id).int().sum(-1) - 1
        device = inputs['input_ids'].device
        bsz, sql = inputs['input_ids'].shape
        class_poss = []
        for idx in label_id_dict.values():
            class_idx = idx
            for offset, prefix_idx in enumerate(reversed(self.prefix_idxs)):
                class_idx += prefix_idx * 100000 ** (offset + 1)
            input_ids = inputs['input_ids'].detach().clone()
            input_ids[:, 1:] += inputs['input_ids'][:, :-1] * 100000
            input_ids[:, 2:] += inputs['input_ids'][:, :-2] * 100000 * 100000
            class_pos = torch.arange(sql, device=device).unsqueeze(0).repeat(bsz, 1)[
                input_ids == class_idx].squeeze()
            class_poss.append(class_pos)
        return class_poss, final_pos

    def _cal_all_key_and_values_of_class(self, inputs, past_key_values, one_class_one_list=False,
                                         include_final=False):
        class_poss, final_pos = self.get_pos(inputs)

        if include_final:
            class_poss.append(final_pos)

        def get_vecs(ker_or_value, class_poss):
            batch_idx = torch.arange(inputs['input_ids'].shape[0])
            class_vecs = []
            for poss in class_poss:
                class_vec = ker_or_value[batch_idx, :, poss, :]
                class_vecs.append(class_vec.unsqueeze(-2))
            if not one_class_one_list:
                class_vecs = torch.cat(class_vecs, dim=-2)
            return class_vecs

        key_and_values = []
        for layer in range(0, self.layer):
            key_and_values.append(tuple([get_vecs(_, class_poss) for _ in past_key_values[layer]]))
        return key_and_values  # tuple of tuple of tensor (bsz, n_head, num_class, d_head)

    def cal_all_key_and_values_of_class(self, inputs, results, one_class_one_list=False,
                                        include_final=False):
        past_key_values = results.past_key_values
        key_and_values = self._cal_all_key_and_values_of_class(inputs, past_key_values,
                                                               one_class_one_list=one_class_one_list,
                                                               include_final=include_final)
        return key_and_values  # tuple of tuple of tensor (bsz, n_head, num_class, d_head)

    def get_attention(self, inputs, results, layer):
        class_poss, final_pos = self.get_pos(inputs)
        batch_idx = torch.arange(inputs['input_ids'].shape[0])
        scores = []
        for class_pos in class_poss:
            attention = results.attentions[layer][batch_idx, :, final_pos, class_pos]
            score = attention
            if class_pos.numel() == 1:
                score = score.sum(-1)
            else:
                score = score.sum()
            if inputs['input_ids'].shape[0] != 1:
                warnings.warn(f'Only support batch_size=1 now!')
            scores.append(score.unsqueeze(0))
        scores = torch.cat(scores, dim=0)
        return scores

    def cal_all_sim_attn(self, inputs, results):
        sims = []
        for layer in range(0, self.layer):
            sim = self.get_attention(inputs=inputs, results=results, layer=layer)
            sims.append(sim.unsqueeze(1))
        sims = torch.cat(sims, dim=1)
        sims = sims.reshape(inputs['input_ids'].shape[0], -1)
        return sims
