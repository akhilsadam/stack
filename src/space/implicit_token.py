from .tokenizer import Operator as BaseOperator

class Operator(BaseOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rep = Rep(**kwargs)

    def freeze(self):
        for param in self.rep.parameters():
            param.requires_grad = False

    def working_set(self, stack, **kwargs):
        items = [op(stack, **kwargs) for i, op in enumerate(self.ops)]
        target_shape = torch.broadcast_shapes(*[t.shape for t in items])
        items = [t.broadcast_to(target_shape) for t in items]
        # stack_depth, CHW # or NCHW

        working_set = torch.stack(
            items,
            dim=-1,
        ) # BCHW, op
        return working_set