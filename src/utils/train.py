# 5
import time

# --- Helper Function for F0.5 Score ---
def calculate_fbeta(preds, labels, beta=0.5, threshold=0.5):
    # preds: (B, H, W) probabilities
    # labels: (B, H, W) binary
    preds_bin = (preds > threshold).float()
    
    tp = (preds_bin * labels).sum()
    fp = (preds_bin * (1 - labels)).sum()
    fn = ((1 - preds_bin) * labels).sum()
    
    beta_sq = beta ** 2
    fbeta = (1 + beta_sq) * tp / ((1 + beta_sq) * tp + beta_sq * fn + fp + 1e-8)
    return fbeta.item()

# --- Training Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 100  # <--- The key to winning
CENTER_SLICE_IDX = 2 

# --- Setup ---
train_ds = GeometricInkDataset(DATA_ROOT, split='train')
val_ds = GeometricInkDataset(DATA_ROOT, split='test')
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = UNet3D(in_channels=4, out_channels=1).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss() # Standard loss is best for balanced data

# --- Training Loop ---
best_val_f05 = 0.0
print(f"Starting High-Performance Training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    # Train
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs[:, :, CENTER_SLICE_IDX, :, :], masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_loss, val_f05 = 0.0, 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            center_preds = outputs[:, :, CENTER_SLICE_IDX, :, :]
            val_loss += criterion(center_preds, masks).item()
            val_f05 += calculate_fbeta(torch.sigmoid(center_preds), masks)
            
    avg_val_f05 = val_f05 / len(val_loader)
    
    # Logging (Clean output)
    if (epoch+1) % 5 == 0 or avg_val_f05 > best_val_f05:
        print(f"Epoch {epoch+1}: Val F0.5 = {avg_val_f05:.4f} (Best: {best_val_f05:.4f})")
    
    # Save Best
    if avg_val_f05 > best_val_f05:
        best_val_f05 = avg_val_f05
        torch.save(model.state_dict(), "best_model_m3.pth")
        
print(f"Training Complete. Best F0.5: {best_val_f05}")
