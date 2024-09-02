import torch.optim as optim


def build_optimizer(cfg, model):
    optim_name = cfg['solver']['name']
    if optim_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['solver']['lr0'],
            momentum=cfg['solver']['momentum'],
            nesterov=True,
            weight_decay=5e-4
        )
    else:
        raise NotImplementedError
    print(f'Optimizer 초기화 완료 !!\n'
          f'Optimizer : {optim_name} \t Params : {optimizer.defaults}\n')
    return optimizer
