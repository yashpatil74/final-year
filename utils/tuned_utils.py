from itertools import product

def hyperparameter_tuning(search_space, train_loader, val_loader, model, criterion, device, train_func):
    best_model = None
    best_params = {}
    best_val_acc = 0

    for params in product(*search_space.values()):
        lr, optimizer_type = params
        optimizer = optimizer_type(model.parameters(), lr=lr)
        scheduler = None  # Add schedulers if needed

        model_copy = model.to(device)
        history = train_func(model_copy, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5, device=device)
        val_acc = history["val_acc"][-1]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model_copy
            best_params = {"lr": lr, "optimizer": optimizer_type.__name__}

    return best_model, best_params
