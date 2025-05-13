import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z, condition):
        z_cond = torch.cat([z, condition], dim=1)
        return self.model(z_cond)


class Critic(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, condition):
        x_cond = torch.cat([x, condition], dim=1)
        return self.model(x_cond)


class WGAN_GP:
    def __init__(self, input_dim, output_dim, condition_dim, device=None):
        self.device = device if device is not None else get_device()
        print(f"Using device: {self.device}")
        
        # Initialize models and move to device
        self.generator = Generator(input_dim + condition_dim, output_dim).to(self.device)
        self.critic = Critic(output_dim, condition_dim).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.0, 0.9))
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-4, betas=(0.0, 0.9))
        
        self.condition_dim = condition_dim
        self.input_dim = input_dim

    def compute_gradient_penalty(self, real_samples, fake_samples, condition):
        # Move tensors to device
        real_samples = real_samples.to(self.device)
        fake_samples = fake_samples.to(self.device)
        condition = condition.to(self.device)
        
        alpha = torch.rand(real_samples.size(0), 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.critic(interpolates, condition)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, real_data, condition, n_critic=3):
        # Move input data to device
        real_data = real_data.to(self.device)
        condition = condition.to(self.device)
        batch_size = real_data.size(0)

        for _ in range(n_critic):
            self.c_optimizer.zero_grad()
            z = torch.randn(batch_size, self.input_dim).to(self.device)
            fake_data = self.generator(z, condition).detach()
            real_validity = self.critic(real_data, condition)
            fake_validity = self.critic(fake_data, condition)
            gradient_penalty = self.compute_gradient_penalty(real_data, fake_data, condition)
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 5 * gradient_penalty
            c_loss.backward()
            self.c_optimizer.step()

        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_data = self.generator(z, condition)
        fake_validity = self.critic(fake_data, condition)
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        self.g_optimizer.step()

        return {'c_loss': c_loss.item(), 'g_loss': g_loss.item()}

    @torch.no_grad()
    def generate(self, n_samples, condition_matrix):
        self.generator.eval()
        z = torch.randn(n_samples, self.input_dim).to(self.device)
        condition_tensor = torch.tensor(condition_matrix).float().to(self.device)
        samples = self.generator(z, condition_tensor)
        self.generator.train()
        return samples.cpu().numpy()
