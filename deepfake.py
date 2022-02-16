import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms
from progress.bar import Bar

# https://www.codeproject.com/Articles/5298025/Building-and-Training-Deep-Fake-Autoencoders
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,256, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256, 64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 16, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=2, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        return self.decoder(encode)

def main():
    path = "scott_faces/scott01277.jpg"
    # img = torch.unsqueeze(img, 0)

    img_transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder('images/scott', transform=img_transform)
    training_size = int(len(dataset)*.8)
    validation_size = len(dataset) - training_size
    [training_data, validation_data] = torch.utils.data.random_split(dataset, [training_size, validation_size])
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1, shuffle=True)
    
    ae = AutoEncoder().cuda()

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)

    epochs = 100

    # another useful site analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/ 
    for epoch in range(epochs):
        train_loss = 0.0
        prog = Bar("Training", max=len(training_loader))
        for images, _ in training_loader:
            images = images.cuda()

            optimizer.zero_grad()
            outputs = ae.forward(images)
            loss = loss_fn(outputs, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*images.size(0)

            prog.next()

        prog.finish()
        train_loss = train_loss / len(training_loader)
        print('Epoch: {} \tTraining Loss: {:6f}'.format(epoch, train_loss))
    
    torch.save(ae, "model.pth")
    count = 0
    for images, _ in validation_loader:
        images = images.cuda()
        output = ae.forward(images)
        images = images.cpu()

        images = torch.mul(images, 255).type(torch.uint8)
        images = images.squeeze()

        output = output.cpu().detach()
        output = torch.mul(output, 255).type(torch.uint8)
        output = output.squeeze()

        torchvision.io.write_jpeg(images, 'biden_outputs/real' + str(count) + '.jpg')
        torchvision.io.write_jpeg(output, 'biden_outputs/fake' + str(count) + '.jpg')
        count += 1

if __name__ == "__main__":
    main()
