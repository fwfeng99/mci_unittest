# %% [markdown]
# ## Model training

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.tensorboard import SummaryWriter

# %% [markdown]
# ### Prepare dataset

# %%
# training data
training_data = torchvision.datasets.CIFAR10(
    root="../data/train/",
    train=True,
    download=True,
    transform=ToTensor(),
)

# testing data
testing_data = torchvision.datasets.CIFAR10(
    root="../data/val/",
    train=False,
    download=True,
    transform=ToTensor(),
)

# size of data
training_data_size = len(training_data)
testing_data_size = len(testing_data)
print(f"Training data size: {training_data_size}")
print(f"Testing data size: {testing_data_size}")

# data loaders
batch_size = 64
training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

# through one batch
for X, y in training_loader:
    print("Shape of X [N, C, H, W]: ", X.shape)     # 每个batch数据的形状
    print("Shape of y: ", y.shape)                  # 每个batch标签的形状
    break

# %% [markdown]
# ### Define the model

# %%
# define the model
class MyNN(nn.Module):

    def __init__(self):
        super(MyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# Determine the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Move the model to the designated device
mynn = MyNN().to(device)
print(mynn)

# %% [markdown]
# ### Training parameters

# %%
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(mynn.parameters(), lr=0.005)

total_train_step = 0
total_test_step = 0
epochs = 20

# %% [markdown]
# ### Training

# %%
# Define training step
def train_step(model, data_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch % 100 == 0:
            print(f"Training Batch: {batch} Loss: {loss.item()}")
            writer.add_scalar("Train Loss", loss.item(), batch)
    return train_loss


# Define validation step
def val_step(model, data_loader, loss_fn, device, epoch):
    model.eval()
    val_loss = 0
    val_accuarcy = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss+= loss.item()
            val_accuarcy += (y_pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= len(data_loader)
    val_accuarcy /= len(data_loader.dataset)

    writer.add_scalar("Val Loss", val_loss, epoch)
    writer.add_scalar("Val Accuracy", val_accuarcy, epoch)
    return val_loss, val_accuarcy


writer = SummaryWriter("../logs/train/")

for epoch in range(epochs):
    print(f"-------Training Epoch ({epoch + 1})-------")

    train_loss = train_step(mynn, training_loader, optimizer, loss_fn, device)
    val_loss, val_accuarcy = val_step(mynn, testing_loader, loss_fn, device, epoch)

    if epoch % 10 == 0:
        torch.save(mynn.state_dict(), f"../checkpoints/model_train_{epoch}.pth")
        print(f"Model saved at {epoch} epoch")

# %%
torch.save(mynn.state_dict(), f"../model_hub/model_train_19.pth")

# %% [markdown]
# ### Testing

# %%
from PIL import Image

image_path = "../data/bird.png"
image = Image.open(image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)

model = MyNN()
model.load_state_dict(torch.load("../checkpoints/model_train_19.pth"))
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)

print(output)
print(output.argmax(1))


