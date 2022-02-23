import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms
from progress.bar import Bar
from prettytable import PrettyTable
import sys

LATENT_DIMS = 48

# inspired by https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb#scrollTo=QVpcKoTdOsK7
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5,  stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=6),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, LATENT_DIMS)
        self.fc_logvar = nn.Linear(1024, LATENT_DIMS)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(LATENT_DIMS, 1024)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view((x.size()[0], 1024, 1, 1))
        x = self.conv(x)
        return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent_sample = self.sample(mu, logvar)
        output = self.decoder(latent_sample)
        return output, mu, logvar

    def sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

def vae_loss(output, x, mu, logvar):
    beta = 1
    bce = torch.nn.functional.binary_cross_entropy(output, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta*kld

def compute_validation(loader, ae):
    validation_loss = 0
    for images, _ in loader:
        images = images.cuda()
        outputs, mu, logvar = ae.forward(images)
        loss = vae_loss(outputs, images, mu, logvar)
        validation_loss += loss.item()
    return validation_loss / len(loader)

def main():

    load = False
    if len(sys.argv) > 1 and sys.argv[1] == '--load':
        load = True

    batch_size = 256
    img_transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder('images/biden', transform=img_transform)
    
    if not load:
        vae = VAE().cuda()
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001, weight_decay=1e-5)
        training_size = int(len(dataset)*.8)
        validation_size = len(dataset) - training_size
        [training_data, validation_data] = torch.utils.data.random_split(dataset, [training_size, validation_size])
        starting_epoch = 0
    else:
        model = torch.load('biden_model.pth')
        vae = model['model'].cuda()
        optimizer = model['optimizer']
        starting_epoch = model['epoch'] + 1
        training_data = model["training"]
        validation_data = model["validation"]
    print(vae)


    epochs = 10000

    best = 100000

    # for logging purposes
    loss_log = open("loss.csv", "a")

    # get data
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1, shuffle=True)

    # another useful site analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/ 
    for epoch in range(starting_epoch, epochs):
        vae.train()
        train_loss = 0.0
        prog = Bar(f"Training Epoch {epoch}:", max=len(training_loader))
        for images, _ in training_loader:
            images = images.cuda()

            optimizer.zero_grad()
            output, mu, logvar = vae.forward(images)
            loss = vae_loss(output, images, mu, logvar)
            loss.backward()
            # nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            train_loss += loss.item()

            prog.next()

        prog.finish()
        train_loss = train_loss / (len(training_loader) * batch_size)
        vae.eval()
        val_loss = compute_validation(validation_loader, vae)
        print('Epoch: {} \tTraining Loss: {:6f}\n\t\tVal Loss: {:6f}'.format(epoch, train_loss, val_loss))

        loss_log.write(f"{epoch},{train_loss},{val_loss}\n")
        loss_log.flush()

        if epoch % 10 == 0:
            torch.save({"model": vae, "optimizer": optimizer, "epoch": epoch, "training": training_data, "validation": validation_data}, "biden_model.pth")

        if val_loss < best:
            best = val_loss
            torch.save({"model": vae, "optimizer": optimizer, "epoch": epoch, "training": training_data, "validation": validation_data}, "best_biden.pth")
        
    torch.save({"model": vae, "optimizer": optimizer, "epoch": epoch, "training": training_data, "validation": validation_data}, "biden_model.pth")

    loss_log.close()
    
    count = 0

if __name__ == "__main__":
    main()
