from .tokenizer import OpRep as BaseOpRep



class Operator(BaseOpRep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        wrapper = ContinuousSurrogateWrapper(
            original_module_list=self.ops,
            input_dim=1, output_dim=1, embed_dim=self.n+4
        )
        wrapper.train_and_freeze_surrogate(epochs=40)

        self.rep = wrapper

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