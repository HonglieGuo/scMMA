import re

log_path = r"d:\my-projects\scMMA\wandb\run-20260202_212157-kdxxtazv\files\output.log"

loss_history = {} # epoch -> val_loss

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    current_epoch = -1
    for line in f:
        # Remove control characters to make regex easier
        # But wait, progress bars use \r to overwrite. 
        # Python's 'for line in f' usually reads until \n.
        # If \r is used without \n, it might be one huge line or split weirdly?
        # Usually file buffering splits by \n.
        
        # Check for Epoch start/progress
        # Format: Epoch 0/19 ...
        epoch_match = re.search(r"Epoch (\d+)/(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # Check for val_loss in the line
        # Format: val_loss: 5.000
        val_loss_match = re.search(r"val_loss:\s*([\d\.]+)", line)
        if val_loss_match and current_epoch != -1:
            loss = float(val_loss_match.group(1))
            loss_history[current_epoch] = loss

print("Epoch | Val Loss")
print("---|---")
for epoch in sorted(loss_history.keys()):
    print(f"{epoch} | {loss_history[epoch]}")
