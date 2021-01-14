import torch
import torch.nn as nn


def log_t(u, t):
    """Compute log_t for `u`."""

    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u`."""

    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0).
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    i = 0
    while i < num_iters:
        i += 1
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)

    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    if t < 1.0:
        return None  # not implemented as these values do not occur in the authors experiments...
    else:
        return compute_normalization_fixed_point(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
    Returns:
    A probabilities tensor.
    """

    if t == 1.0:
        normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
    else:
        normalization_constants = compute_normalization(activations, t, num_iters)

    repeat_normalization_constants = normalization_constants.repeat(activations.size(1)).view(activations.size(1), -1).transpose(1, 0)
    return exp_t(activations - repeat_normalization_constants, t)

class bi_tempered_logistic_loss(nn.Module):
    def __init__(self, t1=1.0, t2=1.0, label_smoothing=0.1, num_iters=5):
        '''Bi-Tempered Logistic Loss with custom gradient.
        Args:
        t1: Temperature 1 (< 1.0 for boundedness).
        t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
        label_smoothing: Label smoothing parameter between [0, 1).
        num_iters: Number of iterations to run the method.
        '''
        super(bi_tempered_logistic_loss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters

    def forward(self, logits, labels):
        """
        Inputs:
        logits: A multi-dimensional tensor with last dimension `num_classes`. tensor of shape (N, C)
        label: A tensor with shape and dtype as activations. tensor of shape(N)

        Returns:
        A loss tensor.
        """
        logits = logits.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = labels.clone().detach()
            lb_pos, lb_neg = 1. - self.label_smoothing, self.label_smoothing / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        # if self.label_smoothing > 0.0:
        #     num_classes = logits.shape[-1]
        #     labels = (1 - num_classes / (num_classes - 1) * self.label_smoothing) * labels + self.label_smoothing / (num_classes - 1)

        probabilities = tempered_softmax(logits, self.t2, self.num_iters)

        temp1 = (log_t(lb_one_hot + 1e-10, self.t1) - log_t(probabilities, self.t1)) * lb_one_hot
        temp2 = (1 / (2 - self.t1)) * (torch.pow(lb_one_hot, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss_values = temp1 - temp2

        loss = loss_values.sum() / logits.size(0)

        # return torch.sum(loss_values, dim=-1)
        return loss

if __name__ == "__main__":
    device = "cuda:0"

    activations = torch.FloatTensor([[-0.5, 0.1, 2.0]]).to(device)
    labels = torch.FloatTensor([[0.2, 0.5, 0.3]]).to(device)

    loss_function1 = bi_tempered_logistic_loss(t1=1.0, t2=1.0, label_smoothing=0.1)
    loss_function2 = bi_tempered_logistic_loss(t1=0.7, t2=1.3, label_smoothing=0.1)
    # The standard logistic loss is obtained when t1 = t2 = 1.0
    loss = loss_function1(activations, labels)
    print("Loss, t1=1.0, t2=1.0: ", loss)

    loss = loss_function2(activations, labels)
    print("Loss, t1=0.7, t2=1.3: ", loss)
