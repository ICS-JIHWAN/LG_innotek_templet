def get_config_dict():
    path = dict(
        save_base_path='runs'
    )

    model = dict(
        name='resnet',  # [lenet|alexnet|resnet]
        num_class=3
    )

    solver = dict(
        name='sgd',
        gpu_id=0,
        lr0=1e-3,
        momentum=0.937,
        weight_decay=5e-4
    )

    scheduler = dict(
        name='steplr'
    )

    # Merge all information into a dictionary variable
    config = dict(
        path=path,
        model=model,
        solver=solver,
        scheduler=scheduler,
    )

    return config
