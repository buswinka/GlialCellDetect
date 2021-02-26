import torch
from torchvision import transforms
from tqdm import trange
from src.dataloader import FasterRCNNData
from src.model import faster_rcnn as model
import src.transforms as t
import src.utils

tr = transforms.Compose([
  t.to_cuda(),
  t.gaussian_blur(),
  t.random_affine(),
  t.random_h_flip(),
  t.random_v_flip(),
  t.adjust_contrast(),
  t.adjust_brightness(),
  t.correct_boxes(),
])

data = FasterRCNNData('./data/train', tr)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

lr = 1e-8
epoch = 2000

epoch_range = trange(epoch, desc='Loss: {1.00000}', leave=True)

model.train().to(device)
model.load_state_dict(torch.load('modelfiles/yeet.mdl'))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
norm = t.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
for e in epoch_range:
    epoch_loss = []
    for image, data_dict in data:
        image = norm({'image': image})['image']
        optimizer.zero_grad()
        loss = model(image.unsqueeze(0), [data_dict])
        losses = 0
        for key in loss:
            losses += loss[key]
        losses.backward()
        epoch_loss.append(losses.item())
        optimizer.step()

    epoch_range.desc = 'Loss: ' + '{:.5f}'.format(torch.tensor(epoch_loss).mean().item())

torch.save(model.state_dict(), 'modelfiles/yeet.mdl')

model.eval()
for image, dd in data:
    out = model(norm({'image': image})['image'].unsqueeze(0))[0]
    src.utils.render_boxes(image, out, 0.5)

print(len(data))