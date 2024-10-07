def get_config_dict():
    path = dict(
        save_base_path='runs'
    )

    model = dict(
        name='resnet',  # [lenet|alexnet|resnet]
        num_channels=3
    )

    solver = dict(
        name='adam',     # [sgd|adam]
        gpu_id=0,
        lr0=1e-3,
        momentum=0,
        weight_decay=0,
    )

    scheduler = dict(
        name=None,      # [None|steplr]
    )

    # Merge all information into a dictionary variable
    config = dict(
        path=path,
        model=model,
        solver=solver,
        scheduler=scheduler,
    )

    return config
