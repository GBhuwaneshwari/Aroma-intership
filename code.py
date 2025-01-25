import numpy as np
import matplotlib.pyplot as plt
# Generator network
class Generator:
    def __init__(self, noise_dim, img_dim):
        self.noise_dim = noise_dim
        self.img_dim = img_dim
    def forward(self, noise):
        # Simple linear transformation
        return np.tanh(np.dot(noise, np.random.randn(self.noise_dim, self.img_dim)))
# Discriminator network
class Discriminator:
    def __init__(self, img_dim):
        self.img_dim = img_dim
        self.weights = np.random.randn(self.img_dim)
        # Initialize weights for scoring
    def forward(self, img):
        # Simple scoring function
        if len(img.shape) == 1:
            score = np.dot(img, self.weights)
        else:
            score = np.dot(img, self.weights)  # Handle batches of images
        return 1 / (1 + np.exp(-score))  # Sigmoid output
# Training loop
def train_gan(generator, discriminator, epochs, batch_size, noise_dim):
    real_data = np.random.rand(1000, generator.img_dim) * 2 - 1  # Random real images
    fake_data_history = []
    for epoch in range(epochs):
        for _ in range(batch_size):
            # Generate noise
            noise = np.random.randn(batch_size, noise_dim)
            # Generate fake images
            fake_imgs = generator.forward(noise)
            # Discriminator decisions
            real_sample = real_data[np.random.randint(0, len(real_data))]
            real_score = discriminator.forward(real_sample)
            fake_score = discriminator.forward(fake_imgs[0])
            # Update Discriminator (fake and real classification)
            d_loss = -(np.log(real_score) + np.log(1 - fake_score))
            # Update Generator (fool Discriminator)
            g_loss = -np.log(fake_score)
            # Collect fake data for visualization
            fake_data_history.append(fake_imgs[0])
        print(f"Epoch {epoch + 1}/{epochs} - D Loss: { d_loss :.4f}, G Loss: {g_loss:.4f}")
    return fake_data_history
# Main script
noise_dim = 10
img_dim = 64
epochs = 10  # Reduced epochs for quicker execution
batch_size = 10
generator = Generator(noise_dim, img_dim)
discriminator=Discriminator(img_dim)
# Train the GAN
generated_data = train_gan(generator, discriminator, epochs, batch_size, noise_dim)
# Visualize some generated images
for i in range(5):
    plt.imshow(generated_data[i].reshape(8, 8), cmap='gray')  # Reshape to 8x8 image
    plt.title(f"Generated Image {i+1}")
    plt.show()