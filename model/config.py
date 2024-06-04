class Config(object):
    def __init__(self):
        self.sequence_length = 400

        # model configs
        self.input_channels = 180
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 512

        self.num_classes = 6
        self.dropout = 0.35
        self.features_len = 52

        # training configs
        self.num_epoch = 200

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128
        self.batch_size_finetune = 4

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.attention_type="ca"  # or sa


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 4


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6
