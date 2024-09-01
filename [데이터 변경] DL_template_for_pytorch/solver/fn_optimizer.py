import torch.optim as optim

def build_optimizer(cfg, model):
    if cfg['solver']['name'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['solver']['lr0'],
            momentum=cfg['solver']['momentum'],
            nesterov=True,
            weight_decay=5e-4
        )
    else:
        raise NotImplementedError

    return optimizer
