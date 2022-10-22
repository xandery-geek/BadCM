

def l2_loss(x, y, reduction=None):
    loss = ((x - y)**2).sum(1).sqrt()
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


def cosine_similarity(x, y):
    inner_product = x @ y.t()
    x_norm = (x ** 2).sum(1, keepdim=True).sqrt()
    y_norm = (y ** 2).sum(1, keepdim=True).sqrt()

    cos = inner_product / (x_norm @ y_norm.t()).clamp(min=1e-6)
    return cos
