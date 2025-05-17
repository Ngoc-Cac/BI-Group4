import torch

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_disbert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
_disbert = AutoModel.from_pretrained('distilbert-base-cased')
_disbert.requires_grad_ = False

tokenize = lambda description: _disbert_tokenizer(
    description, return_tensors='pt',
    padding=True, truncation=True
)

def train_loop(
    model, loss_fn,
    optimizer, dataloader,
    use_gpu: bool = False
):
    model.train()
    if use_gpu: _disbert.cuda()
    else: _disbert.cpu()

    losses = []
    pbar = tqdm(dataloader, total=len(dataloader))
    for stats, des, ratings in pbar:
        tokens = tokenize(des)
        if use_gpu:
            tokens_id = tokens['input_ids'].cuda()
            tokens_mask = tokens['attention_mask'].cuda()
            stats = stats.cuda()
            ratings = ratings.cuda()

        des_embeddings = _disbert(tokens_id, tokens_mask)
        des_embeddings = des_embeddings['last_hidden_state'][:, 0, :]

        rating_preds = model(torch.concat([stats, des_embeddings], axis=1))
        loss = loss_fn(rating_preds.squeeze(), ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix_str(f'Loss: {losses[-1]}')
    return losses

@torch.no_grad()
def eval_loop(
    model, loss_fn,
    test_loader,
    use_gpu: bool = False
):
    model.eval()
    if use_gpu: _disbert.cuda()
    else: _disbert.cpu()

    losses = []
    for stats, des, ratings in test_loader:
        tokens = tokenize(des)
        if use_gpu:
            tokens_id = tokens['input_ids'].cuda()
            tokens_mask = tokens['attention_mask'].cuda()
            stats = stats.cuda()
            ratings = ratings.cuda()
        des_embeddings = _disbert(tokens_id, tokens_mask)
        des_embeddings = des_embeddings['last_hidden_state'][:, 0, :]


        rating_preds = model(torch.concat([stats, des_embeddings], axis=1))
        loss = loss_fn(rating_preds.squeeze(), ratings)

        losses.append(loss.item())

    return losses