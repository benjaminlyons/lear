import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms
from progress.bar import Bar
from prettytable import PrettyTable

def count_param(model):
    count = 0
    table = PrettyTable(["Modules", "Parameters"])
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        count += param
    print(table)
    print(f"Total Params: {count}")

# https://www.codeproject.com/Articles/5298025/Building-and-Training-Deep-Fake-Autoencoders
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5,  stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=6)
        )

        self.fc = nn.Sequential(
                nn.Linear(1024, 24),
                nn.ReLU(),
                nn.Linear(24, 1024),
                nn.ReLU()
        )

        self.decoder = nn.Sequential(
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
        encode = self.encoder(x)
        encode = torch.flatten(encode, 1)
        dec_in = self.fc(encode)
        dec_in = dec_in.view((x.size()[0], 1024, 1, 1))
        decode = self.decoder(dec_in)
        return decode

def compute_validation(loader, ae):
    validation_loss = 0
    loss_fn = nn.BCELoss()
    for images, _ in loader:
        images = images.cuda()
        outputs = ae.forward(images)
        loss = loss_fn(outputs, images)
        validation_loss += loss.item()*images.size(0)
    return validation_loss / len(loader)

def main():

    BATCH_SIZE = 256
    img_transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder('images/trump', transform=img_transform)
    
    ae = AutoEncoder().cuda()
    # model = torch.load('biden_model.pth')
    # ae = model['model'].cuda()
    print(ae)
    count_param(ae)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    # optimizer = model['optimizer']

    epochs = 10000
    # starting_epoch = model['epoch'] + 1
    starting_epoch = 0

    best = 1000

    # for logging purposes
    loss_log = open("loss.csv", "a")

    # get data
    training_size = int(len(dataset)*.8)
    validation_size = len(dataset) - training_size
    [training_data, validation_data] = torch.utils.data.random_split(dataset, [training_size, validation_size])
    # training_data = model["training"]
    # validation_data = model["validation"]
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1, shuffle=True)

    # another useful site analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/ 
    for epoch in range(starting_epoch, epochs):
        train_loss = 0.0
        prog = Bar("Training", max=len(training_loader))
        for images, _ in training_loader:
            images = images.cuda()

            optimizer.zero_grad()
            outputs = ae.forward(images)
            loss = loss_fn(outputs, images)
            loss.backward()
            # nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            train_loss += loss.item()*images.size(0)

            prog.next()

        prog.finish()
        train_loss = train_loss / len(training_loader)
        val_loss = compute_validation(validation_loader, ae)
        print('Epoch: {} \tTraining Loss: {:6f}\n\tVal Loss: {:6f}'.format(epoch, train_loss, val_loss))

        loss_log.write(f"{epoch},{train_loss},{val_loss}\n")
        loss_log.flush()

        if epoch % 10 == 0:
            torch.save({"model": ae, "optimizer": optimizer, "epoch": epoch, "training": training_data, "validation": validation_data}, "trump_model.pth")

        if val_loss < best:
            best = val_loss
            torch.save({"model": ae, "optimizer": optimizer, "epoch": epoch, "training": training_data, "validation": validation_data}, "best_trump.pth")
        
    torch.save({"model": ae, "optimizer": optimizer, "epoch": epoch, "training": training_data, "validation": validation_data}, "trump_model.pth")

    loss_log.close()
    
    count = 0

    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1, shuffle=True)
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

def deepfake():
    biden = torch.load("biden_model.pth")["model"]
    trump = torch.load("trump_model.pth")["model"]
    
    
    img_transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder('images/biden', transform=img_transform)
    training_size = int(len(dataset)*.8)
    validation_size = len(dataset) - training_size
    [training_data, validation_data] = torch.utils.data.random_split(dataset, [training_size, validation_size])
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=512, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1, shuffle=True)

    count = 0
    for images, _ in validation_loader:
        images = images.cuda()
        output = trump.encoder(images)
        output = biden.decoder(output)
        output = output.cpu().detach()
        output = torch.mul(output, 255).type(torch.uint8)
        output = output.squeeze()

        images = torch.mul(images, 255).type(torch.uint8)
        images = images.squeeze().cpu()
        torchvision.io.write_jpeg(output, 'fake_biden/fake' + str(count) + '.jpg')
        torchvision.io.write_jpeg(images, 'fake_biden/original' + str(count) + '.jpg')
        count += 1

if __name__ == "__main__":
    main()
    # deepfake()
