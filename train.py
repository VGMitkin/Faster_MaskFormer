import torch
from tqdm.auto import tqdm

def train(model, 
          device, 
          optimizer_conf, 
          encoder_type, 
          num_epochs, 
          lr, 
          weight_decay,
          train_dataloader,
          val_dataloader,
          dice_weight,
          num_queries, 
          encoder=None, 
          unfreeze_layers=None,
          unfreeze_ratio=None, 
          cons_unfreeze=False,
          full_unfreeze=False, 
          unfreeze_interval=None
          ):
    """
    Trains the model using the specified parameters.

    Args:
        model (torch.nn.Module): The model to be trained.
        device (str): The device to be used for training.
        optimizer_conf (str): The optimizer configuration.
        encoder_type (str): The type of encoder.
        num_epochs (int): The number of training epochs.
        lr (float): The learning rate.
        weight_decay (float): The weight decay.
        train_dataloader (torch.utils.data.DataLoader): The training dataloader.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        dice_weight (float): The weight for the dice loss.
        num_queries (int): The number of queries in transformer decoder block.
        encoder (torch.nn.Module, optional): The encoder module. Required for 'ours' encoder type.
        unfreeze_layers (list, optional): The ordered list of layers to unfreeze. Required for cons_unfreeze.
        unfreeze_ratio (int, optional): The ratio for unfreezing layers. Required for 'ours' encoder type.
        cons_unfreeze (bool, optional): Whether to perform consistent unfreezing.
        full_unfreeze (bool, optional): Whether to perform full unfreezing.
        unfreeze_interval (int, optional): The interval for unfreezing layers. Required for cons_unfreeze.

    Returns:
        None
    """

    model.to(device)

    if full_unfreeze:
        # Enable gradient computation for all parameters
        for p in model.parameters():
            p.requires_grad = True
    else:
        assert encoder_type == 'ours' and encoder is not None, "Please provide encoder for 'ours' type"

        if encoder_type == 'ours':
            # Disable gradient computation for all parameters
            for p in model.parameters():
                p.requires_grad = False

            # Enable gradient computation for encoder parameters
            for p in model.model.pixel_level_module.encoder.parameters():
                p.requires_grad = True

        else:
            # Enable gradient computation for all parameters
            for p in model.parameters():
                p.requires_grad = True

    if optimizer_conf == 'multi':
        assert encoder_type == 'ours', 'Different optimizators avaliable only with FasterViT encoder type'

        optimizer = torch.optim.AdamW([
            {'params': [model.model.pixel_level_module.decoder.parameters(),
                        model.model.transformer_module.parameters(),
                        model.class_predictor.parameters(),
                        model.mask_embedder.parameters(),
                        model.matcher.parameters()
                        ], 'lr': int(lr/100), 'weight_decay': int(weight_decay/5)
            },
            {'params': model.model.pixel_level_module.encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay}
        ])

    else:   
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    running_loss = 0.0
    num_samples = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        
        if cons_unfreeze:
            assert unfreeze_interval is not None and unfreeze_layers is not None, "Please provide unfreeze interval and/or ordered list of layers to unfeeze"

            if epoch % unfreeze_interval == 0:
                unfreeze_layer = unfreeze_layers.pop()
                for p in unfreeze_layer:
                    p.requires_grad = True
        else:
            unfreeze_ratio = unfreeze_ratio if unfreeze_ratio is not None else 10
            if encoder_type == 'ours' and epoch > num_epochs - int(num_epochs/unfreeze_ratio):
                for p in model.parameters():
                    p.requires_grad = True
                
        model.train()
        for _, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()
            
            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            # Optimization
            optimizer.step()

        if epoch % 5 == 0:
            print("Train Loss:", torch.round(running_loss/num_samples, 3))
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_samples = 0
            for idx, batch in enumerate(tqdm(val_dataloader)):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
                )

                loss = outputs.loss
                batch_size = batch["pixel_values"].size(0)
                val_loss += loss.item()
                val_samples += batch_size

        if epoch % 5 == 0:
            print("Validation Loss:", torch.round(val_loss/val_samples, 3))


        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            model.save_pretrained(f"./checkpoints_faster_maskformer/{encoder_type}_encoder_{optimizer_conf}_optimizer_lr_{lr}_decay_{weight_decay}_num_querries_{num_queries}_dice_weight_{dice_weight}_{num_epochs}_epochs/{best_epoch}_{torch.round(best_loss, 3)}")
