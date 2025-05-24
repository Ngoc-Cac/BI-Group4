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
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    use_gpu: bool = False
) -> list[float]:
    """
    Training loop for MovieDataset. This function returns a list of batch-wise mean loss.

    :param torch.nn.Module model: The model to optimize.
    :param torch.nn.Module loss_fn: The loss function to optimize to.
    :param torch.optim.Optimizer optimizer: The optimization object.
    :param torch.utils.data.DataLoader dataloader: The dataloader to load batches
        of training data.
    :param bool use_gpu: Whether or not to train on GPU. This is `False` by default.

    :return: A list of float, each element in this list is the average loss for the
        corresponding batch.
    :rtype: list[float]
    """
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
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    use_gpu: bool = False
) -> list[float]:
    """
    Evaluation loop for MovieDataset. This function returns a list of
    batch-wise mean loss.

    :param torch.nn.Module model: The model used to make inference.
    :param torch.nn.Module loss_fn: The loss function used to compute loss.
    :param torch.utils.data.DataLoader dataloader: The dataloader to load batches
        of test data.
    :param bool use_gpu: Whether or not to do inference on GPU. This is `False` by default.
    
    :return: A list of float, each element in this list is the average loss for the
        corresponding batch.
    :rtype: list[float]
    """
    model.eval()
    if use_gpu: _disbert.cuda()
    else: _disbert.cpu()

    losses = []
    for stats, des, ratings in dataloader:
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