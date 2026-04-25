import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std  = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"


class AddEdgeSpeckles:
    """
    Simulates extraction artifacts: white speckles that appear more often
    near image edges, plus short speckle lines (broken streaks).

    Applied after Normalize so white = 1.0 in [-1, 1] space.

    Args:
        speckle_prob  : base probability of any pixel becoming a speckle (0–1)
        edge_bias     : how much more likely speckles are at edges vs center
        line_count    : max number of short speckle lines to draw per image
        line_len      : (min, max) pixel length of each speckle line
        line_edge_bias: fraction of lines forced to start inside the edge band
        edge_band     : how many pixels from border count as "edge band"
    """
    def __init__(
        self,
        speckle_prob=0.03,
        edge_bias=4.0,
        line_count=3,
        line_len=(2, 6),
        line_edge_bias=0.7,
        edge_band=3,
    ):
        self.speckle_prob   = speckle_prob
        self.edge_bias      = edge_bias
        self.line_count     = line_count
        self.line_len       = line_len
        self.line_edge_bias = line_edge_bias
        self.edge_band      = edge_band

        # Directions: horizontal, vertical, two diagonals
        self._dirs = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    def _make_prob_map(self, h, w):
        """Build (H, W) probability map — high at edges, low at center."""
        ys = torch.arange(h, dtype=torch.float32)
        xs = torch.arange(w, dtype=torch.float32)
        # normalised distance from nearest edge: 0 = at edge, 1 = at centre
        dist_y = torch.min(ys, h - 1 - ys) / (h / 2.0)
        dist_x = torch.min(xs, w - 1 - xs) / (w / 2.0)
        dist   = torch.min(dist_y.unsqueeze(1), dist_x.unsqueeze(0))   # (H, W)
        edge_w = 1.0 - dist                                             # 1 at edge
        return (self.speckle_prob * (1.0 + self.edge_bias * edge_w)).clamp(0.0, 1.0)

    def _edge_start(self, h, w):
        """Return a (y, x) start point biased toward the edge band."""
        band = self.edge_band
        side = random.randint(0, 3)          # top / bottom / left / right
        if side == 0:                        # top band
            return random.randint(0, band), random.randint(0, w - 1)
        elif side == 1:                      # bottom band
            return random.randint(h - band, h - 1), random.randint(0, w - 1)
        elif side == 2:                      # left band
            return random.randint(0, h - 1), random.randint(0, band)
        else:                                # right band
            return random.randint(0, h - 1), random.randint(w - band, w - 1)

    def __call__(self, tensor):
        # tensor: (1, H, W) in [-1, 1]
        _, h, w = tensor.shape
        result  = tensor.clone()

        # --- scattered speckles with edge bias ---
        prob_map     = self._make_prob_map(h, w)
        speckle_mask = torch.bernoulli(prob_map).bool()
        result[0]    = torch.where(speckle_mask, torch.ones_like(result[0]), result[0])

        # --- short speckle lines ---
        n_lines = random.randint(0, self.line_count)
        for _ in range(n_lines):
            length   = random.randint(*self.line_len)
            dy, dx   = random.choice(self._dirs)

            # bias toward edge band most of the time
            if random.random() < self.line_edge_bias:
                y, x = self._edge_start(h, w)
            else:
                y, x = random.randint(0, h - 1), random.randint(0, w - 1)

            for i in range(length):
                ny, nx = y + dy * i, x + dx * i
                if 0 <= ny < h and 0 <= nx < w:
                    result[0, ny, nx] = 1.0

        return result

    def __repr__(self):
        return (f"AddEdgeSpeckles(speckle_prob={self.speckle_prob}, "
                f"edge_bias={self.edge_bias}, line_count={self.line_count})")



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
        transforms.Grayscale(),                        # USPS can come in as RGB
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    augment_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        AddGaussianNoise(mean=0.0, std=0.08),
        AddEdgeSpeckles(speckle_prob=0.04, edge_bias=4.0, line_count=3, line_len=(2, 6)),
    ])

    # EMNIST Digits — 280k samples, high writer diversity
    emnist_train = datasets.EMNIST(root='./data', split='digits', train=True,  download=True, transform=augment_transform)
    emnist_test  = datasets.EMNIST(root='./data', split='digits', train=False, download=True, transform=transform)

    # USPS — rushed postal handwriting, different style from EMNIST
    # Native resolution is 16x16, Resize in transform handles this
    usps_train = datasets.USPS(root='./data', train=True,  download=True, transform=augment_transform)
    usps_test  = datasets.USPS(root='./data', train=False, download=True, transform=transform)

    combined_train = ConcatDataset([emnist_train, usps_train])
    combined_test  = ConcatDataset([emnist_test,  usps_test])

    print(f"Training samples : {len(combined_train):,}  (EMNIST: {len(emnist_train):,} + USPS: {len(usps_train):,})")
    print(f"Test samples     : {len(combined_test):,}  (EMNIST: {len(emnist_test):,} + USPS: {len(usps_test):,})")

    # Split training into train / validation (90/10)
    train_size = int(0.9 * len(combined_train))
    val_size   = len(combined_train) - train_size
    train_set, val_set = random_split(combined_train, [train_size, val_size])

    train_loader = DataLoader(train_set,     batch_size=64, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,       batch_size=64, shuffle=False, num_workers=2)
    test_loader  = DataLoader(combined_test, batch_size=64, shuffle=False, num_workers=2)


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
    PATIENCE      = 5
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
            torch.save(model.state_dict(), 'emnist_usps_trained.pth')
            print("  → saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Final test accuracy
    model.load_state_dict(torch.load('emnist_trained.pth'))
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
    
        img = Image.open(image_path).convert('L')
        tensor = scan_transform(img).unsqueeze(0).to(device)
    
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