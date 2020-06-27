import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

## Your Implementation Here ##
import torch.nn as nn
## Your Implementation Here ##

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.norm,
                                            opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.norm,
                                            opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            ## Your Implementation Here ##
            # Remember that the Reconstruction Loss is L1 
            self.criterionCycle = nn.L1Loss().to(self.device)
            ## Your Implementation Here ##
            # From page 8, identity mapping loss is just the sum of expected L1loss for both samples
            self.criterionIdt = nn.L1Loss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        ## Your Implementation Here ##
        self.fake_B = self.netG_A(self.real_A)  # Gab(a)
        self.rec_A = self.netG_B(self.fake_B)   # Gba( Gab(a) )
        self.fake_A = self.netG_B(self.real_B)  # Gba(b)
        self.rec_B = self.netG_A(self.fake_A)   # Gab( Gba(b) )

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        """
        # Discriminator Losses
        # Real Loss
        real_loss = self.criterionGAN( netD(real), target_is_real=True )

        # Fake Loss
        fake_loss = self.criterionGAN( netD(fake), target_is_real=False )

        total_loss_D = (real_loss + fake_loss) / 2
        total_loss_D.backward()

        return total_loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        ## Your Implementation Here ##

        # variable names are copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
        # with the reason being that train script throws attribute error if you DIDN'T USE THE SAME variable names as the Github source 

        # Options
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Generator A -> B Adversarial Loss
        self.loss_G_B = self.criterionGAN( self.netD_B(self.fake_A), target_is_real=True )   # Target is real is ALWAYS true because we want to convince the discriminator that the 
                                                                                        # Fake images are real

        # Generator B -> A Adversarial Loss
        self.loss_G_A = self.criterionGAN( self.netD_A(self.fake_B), target_is_real=True )

        # Cycle Consistency Loss Loss
        self.loss_cycle_A = self.criterionCycle( self.rec_A, self.real_A ) * lambda_A      # A-B-A L1 Cycle Reconstruction Loss
        self.loss_cycle_B = self.criterionCycle( self.rec_B, self.real_B ) * lambda_B      # B-A-B L1 Cycle Reconstruction Loss

        # Identity Mapping Loss
        lambda_idt = self.opt.lambda_identity
        if lambda_idt > 0:
            # Generator B -> A  with Real_B as inputs
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            # Generator A -> B  with Real_A as inputs
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt

        # Total loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        ## Your Implementation Here ##

        # Forward Pass
        self.forward()

        # Generator Backward
        self.set_requires_grad([self.netD_A, self.netD_B], False)           # Fixed Discriminator Parameters when Optimizing G
        self.optimizer_G.zero_grad()                                        # Zero-out gradients
        self.backward_G()                                                   # Backward pass G
        self.optimizer_G.step()                                             # Gradient Descent G

        # Discriminator Backward
        self.set_requires_grad([self.netD_A, self.netD_B], True)            # Optimize D
        self.optimizer_D.zero_grad()                                        # Zero-out gradients
        self.backward_D_A()                                                 # Backward pass D
        self.backward_D_B()                                                 # Backward pass D
        self.optimizer_D.step()                                             # Gradient Descent D