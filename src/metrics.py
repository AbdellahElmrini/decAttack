import numpy as np
import torch

def square_loss(gradients, gradients_hat):
    """
    Computes the l2 norm between the reconstruced gradients (gradients_hat) and the ground truth gradients
    """
    return [torch.linalg.vector_norm(gradients_hat[i] - gradients[i]).item() for i in range(len(gradients))]

def relative_square_loss(gradients, gradients_hat):
    """
    Computes the relative l2 norm error between the reconstruced gradients (gradients_hat) and the ground truth gradients
    """
    return [torch.linalg.vector_norm(gradients_hat[i] - gradients[i]).item()/torch.linalg.norm(gradients[i]).item() for i in range(len(gradients))]

def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """
    Code from invertinggradient library 
    Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0)#.mean()

    return psnr#.item()


def compute_metrics(img_batch, ref_batch):

    psnr_val = psnr(img_batch, ref_batch)
    square_loss_val = square_loss(img_batch, ref_batch)
    relative_square_loss_val = relative_square_loss(img_batch, ref_batch)

    return psnr_val, square_loss_val, relative_square_loss_val