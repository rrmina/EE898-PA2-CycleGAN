import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_scheduler(optimizer, opt):
    """
    Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler


def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


###################################################################################
## Your Implementation is from Here ##


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'attention_basic':
        net = AttentionGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # you can modify the input variables and types
    elif netG == 'basic':
        net = BaselineGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # you can modify the input variables and types
    elif netG == 'advanced':
        net = None  # you can modify the input variables and types (PA Step4)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'attention_basic':  # self attention based discriminator
        net = AttentionDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'basic':  # baseline discriminator
        net = BaselineDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'advanced':
        net = None  # you can modify the input variables and types (PA Step4)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        """
        super(GANLoss, self).__init__()
        ## Your Implementation Here ##
        self.register_buffer('real_label', torch.tensor(target_real_label))     # registering as a buffer eliminates inflexibility to device (e.g. cpu gpu)
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """

        if (target_is_real):
            self.loss(prediction, self.real_label.expand_as(prediction))
        else:
            self.loss(prediction, self.fake_label.expand_as(prediction))

        return self.loss

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class BaselineGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        """Construct a generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(BaselineGenerator, self).__init__()
        ## Your Implementation Here ##

        # Define Activation and Dropout
        activation_dropout = [nn.ReLU(True)]
        if (use_dropout):
            activation_dropout += [nn.Dropout(0.5)]
        
        c = ngf
        self.ConvBlock = nn.Sequential(
            ConvLayer(  3,   c, 7, 1, norm_layer, padding_type),
            nn.ReLU(True),
            ConvLayer(  c, c*2, 3, 2, norm_layer, padding_type),
            nn.ReLU(True),
            ConvLayer(c*2, c*4, 3, 2, norm_layer, padding_type),
            nn.ReLU(True)
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type)
        )
        self.UpsampleBlock = nn.Sequential(
            DeconvLayer(c*4, c*2, 3, 2, norm_layer, 1, padding_type),
            nn.ReLU(True),
            DeconvLayer(c*2, c*1, 3, 2, norm_layer, 1, padding_type),
            nn.ReLU(True),
            ConvLayer(c*1,   3, 7, 1, None, padding_type),
            nn.Tanh()
        )

    def forward(self, input):
        """Standard forward"""
        ## Your Implementation Here ##
        x = self.ConvBlock(input)
        x = self.ResidualBlock(x)
        x = self.UpsampleBlock(x)
        return x


class AttentionGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        """
        Construct generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        super(AttentionGenerator, self).__init__()
        ## Your Implementation Here ##

        # Define Activation and Dropout
        activation_dropout = [nn.ReLU(True)]
        if (use_dropout):
            activation_dropout += [nn.Dropout(0.5)]
        
        c = ngf
        self.ConvBlock = nn.Sequential(
            SAGAN_ConvLayer(  3,   c, 7, 1, norm_layer, padding_type),
            nn.ReLU(True),
            SAGAN_ConvLayer(  c, c*2, 3, 2, norm_layer, padding_type),
            nn.ReLU(True),
            SAGAN_ConvLayer(c*2, c*4, 3, 2, norm_layer, padding_type),
            nn.ReLU(True)
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type),
            ResidualLayer(c*4, 3, 1, use_dropout, norm_layer, padding_type)
        )
        self.UpsampleBlock = nn.Sequential(
            SAGAN_DeconvLayer(c*4, c*2, 3, 2, norm_layer, 1, padding_type),
            nn.ReLU(True),
            Self_Attn(c*2, None),
            nn.ReLU(),
            SAGAN_DeconvLayer(c*2, c*1, 3, 2, norm_layer, 1, padding_type),
            nn.ReLU(True),
            Self_Attn(c*1, None),
            nn.ReLU(),
            SAGAN_ConvLayer(c*1,   3, 7, 1, None, padding_type),
            nn.Tanh()
        )

    def forward(self, input):
        """Standard forward"""
        ## Your Implementation Here ##
        x = self.ConvBlock(input)
        x = self.ResidualBlock(x)
        x = self.UpsampleBlock(x)
        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        """
        in_dim   -- input feature's channel dim
        activation    -- activation function type
        """
        super(Self_Attn, self).__init__()
        ## Your Implementation Here ##
        
        # Activation is not used because I can not find any reference using activation within self activation modules
        # As far as I know, activation is not necessarily used in Self Attention
        # However, since this is a CV class, in effort activation layer is defined outside instead for readability

        # Reference text SAGAN
        # Page 3 of the reference text "For memory efficiency, we choose k = 8 (i.e., C¯ = C/8) in all our experiments"
        in_channels = in_dim
        out_channels = in_channels // 8

        # Convolution Layers for feature mapping
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size= 1)

        # Affine/scale parameter gamma
        # multiply the output of the attention layer by a scale parameter and add back the input feature map
        # where gamma is a learnable scalar and it is initialized as 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## Your Implementation Here ##
        B,C,H,W = x.shape

        # Query - Feature space F
        query = self.query_conv(x).view(B, -1, H*W).permute(0,2,1)  # [B, H*W, C//8]   - we permute because we take its transpose
                                                                    #                   later, we will do a batch matrix multiplication Q.T * K                 

        # Key - Feature space G
        key = self.key_conv(x).view(B, -1, H*W)                     # [B, C//8, H*W]

        # Attention Proper
        energy = torch.bmm(query, key)                              # [B, H*W, H*W]     In the paper, this is sij
        attention = torch.softmax(energy, dim=-1)                   # [B, H*W, H*W]     in the paperm this is Beta j,i

        # Value - Feature space H
        value = self.value_conv(x).view(B, -1, H*W)                 # [B, C, H*W]

        out = torch.bmm(value, attention.permute(0,2,1))            # [B, C, H*W] * [B, H*W, H*W] 
                                                                    # Permutation is needed even with the same because of the arrangement of the tensors
                                                                    # Tranposing the attention values is convient when reshaping the output to [B,C,H,W]

        out = out.view(B,C,H,W)

        # Apply the affine parameter gamma and Residual as defined in equation 3 of the reference text
        out = self.gamma * out + x

        # Also, in order to be compatible with the API of nn.Sequential(), attention values are not returned

        return out


class BaselineDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(BaselineDiscriminator, self).__init__()
        ## Your Implementation Here ##
        c = ndf

        # Conv layers First layer, no norm
        layers = [
            ConvLayer(3, c, 4, 2, None, padding_type='zero'),
            nn.LeakyReLU(0.2, True)
        ]

        # Conv Layers With norm
        for i in range(n_layers):
            layers += [
                ConvLayer(c, c*2, 4, 2, norm_layer, padding_type='zero'),
                nn.LeakyReLU(0.2, True)
            ]

            # Twice channel
            c *= 2

        # Output layer
        layers += [
            ConvLayer(c, 1, 4, 1, None, padding_type='zero')
        ]

        self.model = nn.Sequential(*layers)


    def forward(self, input):
        """Standard forward."""
        ## Your Implementation Here ##
        return self.model(input)


class AttentionDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(AttentionDiscriminator, self).__init__()
        ## Your Implementation Here ##
        c = ndf

        # Conv layers First layer, no norm
        layers = [
            ConvLayer(3, c, 4, 2, None, padding_type='zero'),
            nn.LeakyReLU(0.2, True),
            Self_Attn(c, None),
            nn.LeakyReLU(0.2, True)
        ]

        # Conv Layers With norm
        for i in range(n_layers):
            layers += [
                ConvLayer(c, c*2, 4, 2, norm_layer, padding_type='zero'),
                nn.LeakyReLU(0.2, True),
                Self_Attn(c*2, None),
                nn.LeakyReLU(0.2, True)
            ]

            # Twice channel
            c *= 2

        # Output layer
        layers += [
            ConvLayer(c, 1, 4, 1, None, padding_type='zero')
        ]

        self.model = nn.Sequential(*layers)


    def forward(self, input):
        """Standard forward."""
        ## Your Implementation Here ##
        return self.model(input)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer, padding_type="reflect"):
        super(ConvLayer, self).__init__()

        # Padding Layers
        self.padding_layer = None
        padding_size = kernel_size // 2
        if (padding_type == "reflect"):
            self.padding_layer = nn.ReflectionPad2d(padding_size)
        elif (padding_type == "replicate"):
            self.padding_layer = nn.ReplicationPad2d(padding_size)
        elif (padding_type == "zero"):
            self.padding_layer = nn.ZeroPad2d(padding_size)

        # Do not use weight bias when using norm layers
        use_bias = False
        if (norm_layer == None):
            use_bias = True

        # Convolution layer
        if (self.padding_layer == None):
            self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding_size, bias=use_bias)
        else:
            self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=use_bias)

        # Normalization Layer
        self.norm_layer = None
        if (norm_layer != None):
            self.norm_layer = norm_layer(out_channels)

    def forward(self, x):
        # Padding 
        if (self.padding_layer != None):
            x = self.padding_layer(x)

        # Conv layer
        x = self.conv_layer(x)

        # Norm Layer
        if (self.norm_layer != None):
            x = self.norm_layer(x)

        return x

# The only difference between regular ConvLayer, and SAGAN_ConvLayer is a spectral norm after the Convolution Layer
class SAGAN_ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer, padding_type="reflect"):
        super(SAGAN_ConvLayer, self).__init__()

        # Padding Layers
        self.padding_layer = None
        padding_size = kernel_size // 2
        if (padding_type == "reflect"):
            self.padding_layer = nn.ReflectionPad2d(padding_size)
        elif (padding_type == "replicate"):
            self.padding_layer = nn.ReplicationPad2d(padding_size)
        elif (padding_type == "zero"):
            self.padding_layer = nn.ZeroPad2d(padding_size)

        # Do not use weight bias when using norm layers
        use_bias = False
        if (norm_layer == None):
            use_bias = True

        # Convolution layer
        if (self.padding_layer == None):
            self.conv_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding_size, bias=use_bias))
        else:
            self.conv_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=use_bias))

        # Normalization Layer
        self.norm_layer = None
        if (norm_layer != None):
            self.norm_layer = norm_layer(out_channels)

    def forward(self, x):
        # Padding 
        if (self.padding_layer != None):
            x = self.padding_layer(x)

        # Conv layer
        x = self.conv_layer(x)

        # Norm Layer
        if (self.norm_layer != None):
            x = self.norm_layer(x)        

        return x

class ResidualLayer(nn.Module):
    def __init__(self, channels=256, kernel_size=3, stride=1, use_dropout=False, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride, norm_layer, padding_type)
        self.relu = nn.ReLU(True)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride, norm_layer, padding_type)
        self.use_dropout = use_dropout
        if (use_dropout):
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        identity = x                        # preserve residual
        out = self.relu(self.conv1(x))      # 1st conv layer -> activation
        if (self.use_dropout):              # (optional) dropout
            out = self.dropout(out)
        out = self.conv2(out)               # 2nd conv layer
        out = out + identity                # add residual
        return out

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer, output_padding, padding_type):
        super(DeconvLayer, self).__init__()

        # Do not use weight bias when using norm layers
        use_bias = False
        if (norm_layer == None):
            use_bias = True

        # Transposed Convolution
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding, bias=use_bias)

        # Normalization Layer
        self.norm_layer = None
        if (norm_layer != None):
            self.norm_layer = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)

        if (self.norm_layer != None):
            x = self.norm_layer(x)

        return x

# The only difference between regular DeconvLayer, and SAGAN_DeconvLayer is a spectral norm after the Transposed Convolution Layer
class SAGAN_DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer, output_padding, padding_type):
        super(SAGAN_DeconvLayer, self).__init__()

        # Do not use weight bias when using norm layers
        use_bias = False
        if (norm_layer == None):
            use_bias = True

        # Transposed Convolution
        padding_size = kernel_size // 2
        self.conv_transpose = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding, bias=use_bias))

        # Normalization Layer
        self.norm_layer = None
        if (norm_layer != None):
            self.norm_layer = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)

        if (self.norm_layer != None):
            x = self.norm_layer(x)

        return x