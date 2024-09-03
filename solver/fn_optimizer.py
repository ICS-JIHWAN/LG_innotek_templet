import torch.optim as optim


def build_optimizer(cfg, model):
    # Optimizer torch.optim 공식 홈페이지에서 parameter 확인 및 변경
    optim_name = cfg['solver']['name']
    if optim_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['solver']['lr0'],
            momentum=cfg['solver']['momentum'],
            weight_decay=cfg['solver']['weight_decay'],
        )
    elif optim_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['solver']['lr0'],
            weight_decay=cfg['solver']['weight_decay'],
        )
    else:
        raise NotImplementedError
    print(f'Optimizer 초기화 완료 !!\n'
          f'Optimizer : {optim_name} \t Params : {optimizer.defaults}\n')
    return optimizer
