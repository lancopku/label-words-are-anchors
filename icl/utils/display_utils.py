import torch


def apply_on_element(l,fn=None):
    if isinstance(l, torch.Tensor):
        l = l.tolist()
    if isinstance(l, list):
        return [apply_on_element(_,fn) for _ in l]
    elif isinstance(l, dict):
        return {k: apply_on_element(v,fn) for k, v in l.items()}
    else:
        return fn(l)

def show_words(logits, tokenizer,topk=5):
    logits = torch.tensor(logits)
    token_ids = logits.topk(topk,dim=-1)[1]
    words = apply_on_element(token_ids,tokenizer.decode)
    print(words)