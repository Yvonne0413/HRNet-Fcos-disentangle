  
"""
Created on Thu Oct 26 11:19:58 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch

from .misc_functions import get_example_params, convert_to_grayscale, save_gradient_images, save_image, save_gradient_images_depu


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            # print(grad_in[0])

        # Register hook to the first layer
        # first_layer = self.model.module.conv1
        first_layer = self.model.backbone.body.conv1
        print(first_layer)
        first_layer.register_backward_hook(hook_function)


    def forward(self, input_image, targets):
        # Forward
        # from torch.autograd import Variable
        # input_img = Variable(input_image, requires_grad=True)
        loss_dict = self.model(input_image, targets)
        # print(loss_dict)
        # Zero grads
        self.model.zero_grad()

        return loss_dict

    def generate_gradients_save_image(self, input_image, target, file_name):
        vanilla_grads = self.generate_gradients(input_image, target)
        save_gradient_images(vanilla_grads, file_name + '_Vanilla_BP_color')
        grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
        save_gradient_images(grayscale_vanilla_grads, file_name + '_Vanilla_BP_gray')

    def save_image(self, input_image, file_name):
        save_image(input_image, file_name + '_ori_image.jpg')

    def save_gradient(self, vanilla_grads, file_name):
        # save_gradient_images(vanilla_grads, file_name + '_Vanilla_BP_color')
        grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
        save_gradient_images(grayscale_vanilla_grads, file_name + '_Vanilla_BP_gray')
        
    def save_gradient_depu(self, vanilla_grads, file_name):
        # save_gradient_images(vanilla_grads, file_name + '_Vanilla_BP_color')
        grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
        save_gradient_images_depu(grayscale_vanilla_grads, file_name + '_Vanilla_BP_gray')


def save_gradient_depu(vanilla_grads, file_name):
    # save_gradient_images(vanilla_grads, file_name + '_Vanilla_BP_color')
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    save_gradient_images_depu(grayscale_vanilla_grads, file_name + '_Vanilla_BP_gray')

if __name__ == '__main__':
    # Get params
    target_example = 1  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    print('Vanilla backprop completed')