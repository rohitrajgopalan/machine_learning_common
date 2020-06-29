import enum


class NetworkInitializationType(enum.Enum):
    HE_NORMAL = 1,
    HE_UNIFORM = 2,
    ORTHOGONAL = 3,
    ZEROS = 4,
    ONES = 5,
    LECUN_UNIFORM = 6,
    LECUN_NORMAL = 7

    @staticmethod
    def all():
        return [NetworkInitializationType.HE_NORMAL,
                NetworkInitializationType.HE_UNIFORM,
                NetworkInitializationType.ORTHOGONAL,
                NetworkInitializationType.ZEROS,
                NetworkInitializationType.ONES,
                NetworkInitializationType.LECUN_NORMAL,
                NetworkInitializationType.LECUN_UNIFORM]


class NetworkActivationFunction(enum.Enum):
    ELU = 1,
    RELU = 2,
    SIGMOID = 3,
    SOFTPLUS = 4,
    TANH = 5,
    LEAKY_RELU = 6,
    SOFTMAX = 7,
    SWISH = 8

    @staticmethod
    def all():
        return [NetworkActivationFunction.ELU,
                NetworkActivationFunction.LEAKY_RELU,
                NetworkActivationFunction.RELU,
                NetworkActivationFunction.SIGMOID,
                NetworkActivationFunction.SOFTMAX,
                NetworkActivationFunction.SOFTPLUS,
                NetworkActivationFunction.SWISH,
                NetworkActivationFunction.TANH]


class NetworkOptimizer(enum.Enum):
    ADAM = 1,
    ADAMAX = 2,
    NADAM = 3

    @staticmethod
    def all():
        return [NetworkOptimizer.ADAM, NetworkOptimizer.ADAMAX, NetworkOptimizer.NADAM]
