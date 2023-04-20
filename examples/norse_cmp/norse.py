import torch
import numpy as np
import torchvision

from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFRecurrentCell
from norse.torch import LICell, LIState

from typing import NamedTuple

from norse.torch import PoissonEncoder

from tqdm import tqdm, trange


BATCH_SIZE = 256

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_data = torchvision.datasets.MNIST(
    root=".",
    train=True,
    download=True,
    transform=transform,
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root=".",
        train=False,
        transform=transform,
    ),
    batch_size=BATCH_SIZE,
)


class SNNState(NamedTuple):
  lif0: LIFState
  readout: LIState


class SNN(torch.nn.Module):
  def __init__(
      self, input_features, hidden_features, output_features, record=False,
      dt=0.001
  ):
    super(SNN, self).__init__()
    self.l1 = LIFRecurrentCell(
        input_features,
        hidden_features,
        p=LIFParameters(alpha=100, v_th=torch.tensor(0.5)),
        dt=dt,
    )
    self.input_features = input_features
    self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
    self.out = LICell(dt=dt)

    self.hidden_features = hidden_features
    self.output_features = output_features
    self.record = record

  def forward(self, x):
    seq_length, batch_size, _, _, _ = x.shape
    s1 = so = None
    voltages = []

    if self.record:
      self.recording = SNNState(
          LIFState(
              z=torch.zeros(seq_length, batch_size, self.hidden_features),
              v=torch.zeros(seq_length, batch_size, self.hidden_features),
              i=torch.zeros(seq_length, batch_size, self.hidden_features),
          ),
          LIState(
              v=torch.zeros(seq_length, batch_size, self.output_features),
              i=torch.zeros(seq_length, batch_size, self.output_features),
          ),
      )

    for ts in range(seq_length):
      z = x[ts, :, :, :].view(-1, self.input_features)
      z, s1 = self.l1(z, s1)
      z = self.fc_out(z)
      vo, so = self.out(z, so)
      if self.record:
        self.recording.lif0.z[ts, :] = s1.z
        self.recording.lif0.v[ts, :] = s1.v
        self.recording.lif0.i[ts, :] = s1.i
        self.recording.readout.v[ts, :] = so.v
        self.recording.readout.i[ts, :] = so.i
      voltages += [vo]

    return torch.stack(voltages)


def decode(x):
  x, _ = torch.max(x, 0)
  log_p_y = torch.nn.functional.log_softmax(x, dim=1)
  return log_p_y


class Model(torch.nn.Module):
  def __init__(self, encoder, snn, decoder):
    super(Model, self).__init__()
    self.encoder = encoder
    self.snn = snn
    self.decoder = decoder

  def forward(self, x):
    x = self.encoder(x)
    x = self.snn(x)
    log_p_y = self.decoder(x)
    return log_p_y


T = 32
LR = 0.002
INPUT_FEATURES = 28 * 28
HIDDEN_FEATURES = 100
OUTPUT_FEATURES = 10

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
else:
  DEVICE = torch.device("cpu")

model = Model(
    encoder=PoissonEncoder(
        seq_length=T,
    ),
    snn=SNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
    ),
    decoder=decode,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


EPOCHS = 5  # Increase this number for better performance


def train(model, device, train_loader, optimizer, epoch, max_epochs):
  model.train()
  losses = []

  for (data, target) in tqdm(train_loader, leave=False):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = torch.nn.functional.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

  mean_loss = np.mean(losses)
  return losses, mean_loss


def test(model, device, test_loader, epoch):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += torch.nn.functional.nll_loss(
          output, target, reduction="sum"
      ).item()  # sum up batch loss
      pred = output.argmax(
          dim=1, keepdim=True
      )  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  accuracy = 100.0 * correct / len(test_loader.dataset)

  return test_loss, accuracy


training_losses = []
mean_losses = []
test_losses = []
accuracies = []

torch.autograd.set_detect_anomaly(True)

for epoch in trange(EPOCHS):
  training_loss, mean_loss = train(
      model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS
  )
  test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
  training_losses += training_loss
  mean_losses.append(mean_loss)
  test_losses.append(test_loss)
  accuracies.append(accuracy)

print(f"final accuracy: {accuracies[-1]}")
