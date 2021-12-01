import torch
import torchvision.models as models
import pytorch_lightning as pl
import pdb
class SpectrResnet(pl.LightningModule):
    def __init__(self):
        """Simple wrapper for using resnet18 as the image feature extractor for masked transformer decoder
        """
        super(SpectrResnet, self).__init__()
        self.resnet = models.resnet18(pretrained = True)
    
    def forward(self, image):
        """Forward method for resnet18 wrapper class

        Args:
            image: A batch of images of shape [batch_size, num_channels, height, width]
        
        Returns:
            output: A feature map of shape [batch_size, 256, height / 16, width / 16]
        """
        output = self.resnet.conv1(image)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)

        output = self.resnet.layer1(output)
        output = self.resnet.layer2(output)
        #output = self.resnet.layer3(output)

        return output

if __name__ == "__main__":
    pdb.set_trace()
    image = torch.randn(size = [4, 3, 480, 304])
    feature_map_extractor = SpectrResnet()
    output = feature_map_extractor(image)
