import torch
from torchvision import transforms
from tqdm import trange
from src.dataloader import FasterRCNNData
from src.model import faster_rcnn as model
from src.model import feature_extractor
from src.pca import pca
import src.transforms as t
import src.utils
import matplotlib.pyplot as plt

tr = transforms.Compose([
  # t.to_cuda(),
  # t.gaussian_blur(),
  # t.random_affine(),
  t.random_h_flip(),
  t.random_v_flip(),
  # t.adjust_contrast(),
  # t.adjust_brightness(),
  t.correct_boxes(),
])

data = FasterRCNNData('./data/train', tr)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

lr = 1e-6
epoch = 100

epoch_range = trange(epoch, desc='Loss: {1.00000}', leave=True)

model.train().to(device)
# model.load_state_dict(torch.load('modelfiles/yeet.mdl'))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
norm = t.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
for e in epoch_range:
    epoch_loss = []
    for image, data_dict in data:
        # c = torch.zeros((1, image.shape[1], image.shape[2]), device=device)
        # image = torch.cat((image, image, image), dim=0)
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
with torch.no_grad():
    model.eval()
    feature_extractor.to(device).eval()
    reduced_images = torch.zeros((1, 2048), device=device)
    label = []
    for image, dd in data:
        # c = torch.zeros((1, image.shape[1], image.shape[2]), device=device)
        # image = torch.cat((c, image, c), dim=0)
        image = norm({'image': image})['image']
        out = model(image.unsqueeze(0))[0]
        for i in range(out['boxes'].shape[0]):
            [x0, y0, x1, y1] = out['boxes'][i, :]
            im = image[:, int(x0):int(x1), int(y0):int(y1)].unsqueeze(0)

            if im.numel() == 0 or out['scores'][i] < 0.5:
                continue

            feature = feature_extractor(im).squeeze().unsqueeze(0)
            reduced_images = torch.cat((reduced_images, feature), dim=0)
            label.append(out['labels'][i].item())
            print(reduced_images.shape, len(label))

        src.utils.render_boxes(image, out, 0.5)

        reduced = pca(reduced_images[1::, :], 2).cpu()
label = torch.tensor(label)
print(label)
plt.plot(reduced[label==1, 0], reduced[label==1, 1], 'o')
plt.plot(reduced[label==2, 0], reduced[label==2, 1], 'o')
plt.title('PCA of Prediced Cells')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

