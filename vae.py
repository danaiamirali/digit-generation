import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def loss_function(recon_x, x, mu, logvar, bce_weight=1.0, kld_weight=0.1):
    """Loss function for VAE"""
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # VAE Loss = BCE + KLD
    return bce_weight * BCE + kld_weight * KLD

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
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc41 = nn.Linear(hidden_dim, latent_dim)
        self.fc42 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        mu = self.fc41(x)
        logvar = self.fc42(x)
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
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return torch.sigmoid(self.fc6(x))

class VAE(nn.Module):
    """VAE model"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterize the latent space"""
        std = torch.exp(0.5*logvar).to("cuda")
        eps = torch.randn_like(std).to("cuda")
        return mu + eps*std

    def forward(self, x):
        # encode the input
        mu, logvar = self.encoder(x)
        # sample the latent space
        z = self.reparameterize(mu, logvar)
        # decode the latent space
        return self.decoder(z), mu, logvar

    def generate_samples(self, num_samples, hidden_layers, output_path):
        """Generate samples from the VAE model"""
        with torch.no_grad():
            z = torch.randn(num_samples, hidden_layers).to("cuda")
            samples = self.decoder(z).view(num_samples, 28, 28)
            samples = samples.cpu().numpy()
            title_texts = ['Sample {}'.format(i) for i in range(num_samples)]

        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Save each sample as an image
        for i, sample in enumerate(samples):
            plt.imshow(sample, cmap='gray')
            plt.title(title_texts[i])
            plt.axis('off')
            plt.savefig(os.path.join(output_path, f'sample_{i}.png'))
            plt.close()

def train_vae(vae, train_loader, optimizer, num_epochs, bce_weight=1.0, kld_weight=0.1):
    """Train the VAE"""
    vae.train()
    vae.to("cuda")
    for epoch in range(num_epochs):
        for i, x in enumerate(train_loader):
            x = x.to("cuda").flatten().float() / 255
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = loss_function(recon_x, x, mu, logvar, bce_weight, kld_weight)
            loss.backward()
            optimizer.step()

            if i % 5000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i, len(train_loader), loss.item()))

    print('Finished Training')