import torch
import torch.nn as nn
import torch.nn.functional as F


# Hook for gradients
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_full_backward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_classes):
        self.model.zero_grad()
        output = self.model(input_tensor).requires_grad_(True)

        # Create a one-hot encoded tensor for the target classes
        one_hot_output = torch.zeros_like(output)
        for i in range(input_tensor.size(0)):  # Batch size loop
            one_hot_output[i, target_classes[i]] = 1

        # Get score for target class and backward to get gradients
        output.backward(retain_graph=True, gradient=one_hot_output)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted sum of feature maps
        cams = torch.sum(weights * self.feature_maps[0], dim=1)

        # ReLU activation to retain positive values
        cams = F.relu(cams)

        # Normalize CAM
        cams_min = cams.view(cams.size(0), -1).min(dim=1)[0].view(cams.size(0), 1, 1)
        cams_max = cams.view(cams.size(0), -1).max(dim=1)[0].view(cams.size(0), 1, 1)

        # Normalize each cam
        cams = (cams - cams_min) / (cams_max - cams_min + 1e-8)
        cams = (255 * cams).type(torch.uint8)

        return cams.detach().cpu().numpy()