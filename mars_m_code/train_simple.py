import torch
import torch.nn as nn
import torch.optim as optim
import torch_neuronx
import torch_xla.core.xla_model as xm
from mars_m import MARS_M
import numpy as np
# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Create model and move to Neuron device
# model = SimpleNet().to('neuron')
device = xm.xla_device() 
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
muon_params = [p for p in model.parameters() if p.ndim >= 2]
adamw_params = [p for p in model.parameters() if p.ndim < 2]
optimizer = MARS_M(
    lr=0.01,
    wd=0.1,
    muon_params=muon_params,
    momentum=0.95,
    ns_steps=5,
    adamw_params=adamw_params,
    adamw_betas=(0.9, 0.95),
    adamw_eps=1e-8,
    gamma=0.025,
    clip_c=False,
    is_approx=True
)

# Generate dummy training data
batch_size = 32
num_batches = 100

print("Starting training...")
model.train()

def generate_math_data(num_samples=1000):
    X = []
    y = []
    t = np.linspace(0, 1, 784)
    
    for _ in range(num_samples):
        category = np.random.randint(0, 10)
        f = (category + 1) * 5 + np.random.uniform(-1, 1) 
        
        signal = np.sin(2 * np.pi * f * t) + np.random.normal(0, 0.2, 784)
        
        X.append(signal)
        y.append(category)
        
    return torch.tensor(X), torch.tensor(y)

for batch_idx in range(num_batches):
    # Create dummy batch
    # inputs = torch.randn(batch_size, 784).to(device)
    # targets = torch.randint(0, 10, (batch_size,)).to(device)
    inputs, targets = generate_math_data(num_samples=batch_size)
    inputs, targets = inputs.to(device), targets.to(device)

    # Training step
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    xm.mark_step()

    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

print("Training complete!")