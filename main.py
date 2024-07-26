"""
Entry point for the project.
"""

from vae import VAE, train_vae
from dataset import MnistDataloader
from torch.optim import Adam

if __name__ == "__main__":
    # Load the dataset
    mnist_dataloader = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Define the VAE model
    img_dim_x = x_train[0].shape[0]
    img_dim_y = x_train[0].shape[1]
    input_dim = img_dim_x * img_dim_y

    # Hyperparameters
    hidden_dim = 400
    latent_dim = 20

    vae = VAE(input_dim, hidden_dim, latent_dim)

    optimizer = Adam(vae.parameters(), lr=1e-3)
    # Train the VAE model
    train_vae(vae, x_train, optimizer=optimizer, num_epochs=10)

    # Generate samples from the VAE model
    vae.eval()
    samples = vae.generate_samples(10, output_path="samples.png")