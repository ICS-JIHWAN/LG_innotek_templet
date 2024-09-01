from torch.utils.data.dataloader import DataLoader


def create_dataloader(cfg_dataset):
    if cfg_dataset['name'] == 'wdm':
        from dataset.data_loader import wdm as dataset_class
    else:
        raise ValueError('Invalid dataset name, currently supported [ wdm ]')
    #
    train_path = cfg_dataset['train_path']
    batch_size = cfg_dataset['batch_size']
    num_workers = cfg_dataset['num_workers']
    height, width = cfg_dataset['height'], cfg_dataset['width']
    #
    train_object = dataset_class(
        path=train_path,
        height=height,
        width=width,
        augmentation=True,
        task='train'
    )
    #
    train_loader = DataLoader(
        train_object,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset_class.collate_fn
    )
    #
    val_object = dataset_class(
        path=train_path,
        height=height,
        width=width,
        task='val'
    )
    val_loader = DataLoader(
        val_object,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset_class.collate_fn
    )
    return train_loader, val_loader
