# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from torch import nn

from practical_deep_stereo import modules


class Embedding(nn.Module):
    """Embedding module."""

    def __init__(self,
                 number_of_input_features=3,
                 number_of_embedding_features=64,
                 number_of_redirect_features=8,
                 number_of_residual_blocks=2):
        """Returns initialized embedding module.

        Args:
            number_of_input_features: number of channels in the input image;
            number_of_embedding_features: number of channels in image's
                                          descriptor;
            number_of_redirect_features: number of channels in the redirect
                                         connection descriptor;
            number_of_residual_blocks: number of residual blocks in embedding
                                       network.
        """
        super(Embedding, self).__init__()
        embedding_modules = [
            modules.convolutional_block_5x5_stride_2(
                number_of_input_features, number_of_embedding_features),
            modules.convolutional_block_5x5_stride_2(
                number_of_embedding_features, number_of_embedding_features),
        ]
        embedding_modules += [
            modules.ResidualBlock(number_of_embedding_features)
            for _ in range(number_of_residual_blocks)
        ]
        self._embedding_modules = nn.ModuleList(embedding_modules)
        self._redirect_modules = modules.convolutional_block_3x3(
            number_of_embedding_features, number_of_redirect_features)

    def forward(self, image):
        """Returns image's descriptor and redirect connection descriptor.

        Args:
            image: color image of size
                   batch_size x 3 x height x width;

        Returns:
            descriptor: image's descriptor of size
                        batch_size x 64 x (height / 4) x (width / 4);
            redirect: redirect connection descriptor of size
                      batch_size x 8 x (height / 4) x (width / 4).
        """
        descriptor = image
        for embedding_module in self._embedding_modules:
            descriptor = embedding_module(descriptor)

        return descriptor, self._redirect_modules(descriptor)
