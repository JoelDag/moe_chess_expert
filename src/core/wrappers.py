import torch
import torch.nn as nn

class AuxLossWrapper(nn.Module):
    def __init__(self, base: nn.Module, coef: float = 1.0, getter_name: str = "get_aux_loss"):
        super().__init__()
        self.base = base
        self.coef = coef
        self.getter_name = getter_name

    def forward(self, *args, **kwargs):
        out = self.base(*args, **kwargs)
        if hasattr(out, "loss") and out.loss is not None:
            getter = getattr(self.base, self.getter_name, None)
            if callable(getter):
                aux = getter()
                if isinstance(aux, torch.Tensor): #TODO:
                    #print(f"[AuxLossWrapper] out.loss.requires_grad: {out.loss.requires_grad}")
                    #print(f"[AuxLossWrapper] aux.requires_grad: {aux.requires_grad}")
                    print(f"[AuxLossWrapper] Adding aux={aux.item()} with coef={self.coef}")
                    out.loss = out.loss + self.coef * aux
                    #print(f"[AuxLossWrapper] new out.loss.requires_grad: {out.loss.requires_grad}")

                    # otherwise aux loss
                    for _, module in self.base.named_modules():
                        if hasattr(module, "layer_loss"):
                            module.layer_loss = None
        return out

    def __getattr__(self, n):
        try:
            return super().__getattr__(n)
        except AttributeError:
            return getattr(self.base, n)
