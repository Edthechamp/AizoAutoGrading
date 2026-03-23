import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


#------
# MODEL
#------

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

#-----
# DATA
#-----

def main():
    torch.multiprocessing.freeze_support()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),           # MNIST is 28x28, resize to match our input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))   # normalize to [-1, 1]
    ])

    augment_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomAffine(
            degrees=10,                        # slight rotation
            translate=(0.1, 0.1),             # slight shift
            scale=(0.85, 1.15),               # slight zoom
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train = datasets.MNIST(root='./data', train=True,  download=True, transform=augment_transform)
    test_set   = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split training into train / validation (90/10)
    train_size = int(0.9 * len(full_train))
    val_size   = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=2)


    #---------------
    # TRAINING SETUP
    #---------------

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model     = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)



    def train_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
    
        return total_loss / len(loader.dataset), correct / len(loader.dataset)


    def eval_epoch(model, loader, criterion, device):
        model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs     = model(images)
                total_loss += criterion(outputs, labels).item() * images.size(0)
                correct    += (outputs.argmax(1) == labels).sum().item()
    
        return total_loss / len(loader.dataset), correct / len(loader.dataset)


    #--------------
    # TRAINING LOOP
    #--------------

    EPOCHS        = 30
    PATIENCE      = 5            # stop if val loss doesn't improve for 5 epochs
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
    
        print(f"Epoch {epoch:02d} | "
            f"train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} acc: {val_acc:.4f}")
    
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'mnist_pretrained.pth')
            print("  → saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Final test accuracy
    model.load_state_dict(torch.load('mnist_pretrained.pth'))
    _, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")


    scan_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    def predict(image_path, model_path='finetuned.pth'):
        """
        Given a path to a cropped cell image, returns the predicted digit
        and the confidence score.
        """
        from PIL import Image
    
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    
        img = Image.open(image_path).convert('L')   # force grayscale
        tensor = scan_transform(img).unsqueeze(0).to(device)  # add batch dim
    
        with torch.no_grad():
            logits     = model(tensor)
            probs      = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)
    
        digit      = predicted.item()
        confidence = confidence.item()
    
        if confidence < 0.85:
            print(f"  WARNING: low confidence ({confidence:.2f}) — flag for manual review")
    
        return digit, confidence





if __name__ == '__main__':
    main()