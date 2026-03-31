"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        assert reduction in {"none","mean","sum"}
        self.eps = eps
        self.reduction = reduction
        # TODO: validate reduction in {"none", "mean", "sum"}.

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        px,py,pw,ph=pred_boxes[:,0],pred_boxes[:,1],pred_boxes[:,2],pred_boxes[:,3]
        tx,ty,tw,th=target_boxes[:,0],target_boxes[:,1],target_boxes[:,2],target_boxes[:,3]
        px1,py1,px2,py2=px-pw/2,py-ph/2,px+pw/2,py+ph/2
        tx1,ty1,tx2,ty2=tx-tw/2,ty-th/2,tx+tw/2,ty+th/2
        ix1=torch.max(px1,tx1)
        iy1=torch.max(py1,ty1)
        ix2=torch.min(px2,tx2)
        iy2=torch.min(py2,ty2)

        inter=(ix2-ix1).clamp(0)*(iy2-iy1).clamp(0)
        pred_area=pw*ph
        target_area=tw*th
        union=pred_area+target_area-inter+self.eps

        iou=inter/union
        loss=1-iou

        if self.reduction=="mean":
            return loss.mean()
        elif self.reduction=="sum":
            return loss.sum()
        return loss