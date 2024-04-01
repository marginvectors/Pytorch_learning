class Adam(GenericAdaptiveOptimizer):
    def __init__(self,params,
                 lr:float=1e-3,betas:Tuple[float,float]=(0.9,0.999),eps:float = 1e-16,
                 weight_decay: WeightDecay=WeightDecay(),
                 Optimized_update: bool=True,
                 defaults:Optional[Dict[str,Any]]=None):
        defaults = 