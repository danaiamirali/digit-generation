import torch
import torch.nn as nn
import torchvision.datasets as datasets

def loss_function(recon_x, x, mu, logvar):
    """Loss function for VAE"""
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # VAE Loss = BCE + KLD
    return BCE + KLD

class Encoder(nn.Module):
    """
    Encoder for VAE.

    The encoder takes the input and encodes it into the latent space.

    It outputs the mean and log variance of the latent space.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        # define the encoder
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc31(x)
        logvar = self.fc32(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder for VAE.

    The decoder takes the latent space and decodes it into the output space.

    It outputs the reconstructed input.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        # define the decoder
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

class VAE(nn.Module):
    """VAE model"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterize the latent space"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # encode the input
        mu, logvar = self.encoder(x)
        # sample the latent space
        z = self.reparameterize(mu, logvar)
        # decode the latent space
        return self.decoder(z), mu, logvar

    def generate_samples(self, num_samples, output_path):
        """Generate samples from the VAE model"""
        with torch.no_grad():
            z = torch.randn(num_samples, 20)
            samples = self.decoder(z).view(num_samples, 28, 28)
            samples = samples.numpy()
            title_texts = ['Sample {}'.format(i) for i in range(num_samples)]
            self._show_images(samples, title_texts, output_path)

def train_vae(vae, train_loader, optimizer, num_epochs):
    """Train the VAE"""
    vae.train()
    for epoch in range(num_epochs):
        for i, x in enumerate(train_loader):
            x = x.flatten().float() / 255
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i, len(train_loader), loss.item()))

    print('Finished Training')