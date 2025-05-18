import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

def get_device():
    # Force CPU usage for better compatibility
    return torch.device("cpu")

class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Ensure input has correct shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        weights = self.attention(x)
        weighted_input = x * weights
        return weighted_input, weights

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, condition_dim):
        super(Generator, self).__init__()
        
        # Total input dimension is input_dim + condition_dim
        total_input_dim = input_dim + condition_dim
        self.attention = FeatureAttention(total_input_dim)
        self.feature_importance = None
        
        # Model layers with correct dimensions
        self.model = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z, condition):
        # Ensure inputs are on the same device as the model
        if z.device != next(self.parameters()).device:
            z = z.to(next(self.parameters()).device)
        if condition.device != next(self.parameters()).device:
            condition = condition.to(next(self.parameters()).device)
            
        # Ensure inputs have batch dimension
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(0)
            
        # Concatenate input and condition
        z_cond = torch.cat([z, condition], dim=1)
        
        # Apply attention and get weighted input
        weighted_input, attention_weights = self.attention(z_cond)
        self.feature_importance = attention_weights.detach()
        
        # Generate output
        return self.model(weighted_input)
    
    def get_feature_importance(self):
        """Return feature importance scores."""
        if self.feature_importance is None:
            return None
        # Move to CPU before converting to numpy
        importance = self.feature_importance.cpu()
        # Synchronize if using MPS device
        if importance.device.type == 'mps':
            torch.mps.synchronize()
        return importance.numpy()


class Critic(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Critic, self).__init__()
        
        # Total input dimension is input_dim + condition_dim
        total_input_dim = input_dim + condition_dim
        
        self.model = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, condition):
        # Ensure inputs are on the same device as the model
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        if condition.device != next(self.parameters()).device:
            condition = condition.to(next(self.parameters()).device)
            
        # Ensure inputs have batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(0)
            
        # Concatenate input and condition
        x_cond = torch.cat([x, condition], dim=1)
        return self.model(x_cond)


class WGAN_GP:
    def __init__(self, input_dim, output_dim, condition_dim, device=None):
        self.device = device if device is not None else get_device()
        print(f"Using device: {self.device}")
        print(f"Input dim: {input_dim}, Output dim: {output_dim}, Condition dim: {condition_dim}")
        
        # Initialize models and move to device
        self.generator = Generator(input_dim, output_dim, condition_dim).to(self.device)
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

    @torch.no_grad()  # Add decorator to disable gradient tracking
    def generate(self, n_samples, condition_matrix):
        self.generator.eval()
        # Print shapes for debugging
        print(f"Generating {n_samples} samples")
        print(f"Input dim: {self.input_dim}")
        print(f"Condition matrix shape: {condition_matrix.shape}")
        
        # Move inputs to device
        z = torch.randn(n_samples, self.input_dim).to(self.device)
        condition_tensor = torch.tensor(condition_matrix, dtype=torch.float32).to(self.device)
        
        print(f"z shape: {z.shape}")
        print(f"condition_tensor shape: {condition_tensor.shape}")
        
        try:
            # Generate samples
            samples = self.generator(z, condition_tensor)
            print(f"Generated samples shape: {samples.shape}")
            
            # Move results back to CPU and detach from computation graph
            samples = samples.cpu().detach()
            
            # Synchronize if using MPS device
            if self.device.type == 'mps':
                torch.mps.synchronize()
            
            self.generator.train()
            return samples.numpy()
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            raise
