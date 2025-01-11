import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

class WGAN_GP:
    def __init__(self, input_dim, output_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = Generator(input_dim, output_dim).to(device)
        self.critic = Critic(output_dim).to(device)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.critic(interpolates)
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

    def train_step(self, real_data, n_critic=5):
        batch_size = real_data.size(0)
        
        # train the critic
        for _ in range(n_critic):
            self.c_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.generator.model[0].in_features).to(self.device)
            fake_data = self.generator(z).detach()
            
            real_validity = self.critic(real_data)
            fake_validity = self.critic(fake_data)
            
            gradient_penalty = self.compute_gradient_penalty(real_data, fake_data)
            
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
            c_loss.backward()
            self.c_optimizer.step()

        # train the generator
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.generator.model[0].in_features).to(self.device)
        fake_data = self.generator(z)
        fake_validity = self.critic(fake_data)
        
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        self.g_optimizer.step()

        return {'c_loss': c_loss.item(), 'g_loss': g_loss.item()}

    def generate(self, n_samples):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.generator.model[0].in_features).to(self.device)
            samples = self.generator(z)
        self.generator.train()
        return samples.cpu().numpy() 