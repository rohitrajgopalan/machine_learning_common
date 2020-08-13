import enum


class NetworkInitializationType(enum.Enum):
    HE_NORMAL = 1,
    HE_UNIFORM = 2,
    ORTHOGONAL = 3,
    ZEROS = 4,
    ONES = 5,
    LECUN_NORMAL = 6,
    LECUN_UNIFORM = 7,
    NORMAL = 8

    @staticmethod
    def all():
        return [NetworkInitializationType.HE_NORMAL,
                NetworkInitializationType.HE_UNIFORM,
                NetworkInitializationType.ORTHOGONAL,
                NetworkInitializationType.ZEROS,
                NetworkInitializationType.ONES,
                NetworkInitializationType.LECUN_NORMAL,
                NetworkInitializationType.LECUN_UNIFORM,
                NetworkInitializationType.NORMAL]

    @staticmethod
    def get_type_by_name(name):
        for initialization_type in NetworkInitializationType.all():
            if initialization_type.name.lower() == name.lower():
                return initialization_type
        return None


class NetworkActivationFunction(enum.Enum):
    ELU = 1,
    RELU = 2,
    SIGMOID = 3,
    SOFTPLUS = 4,
    TANH = 5,
    LEAKY_RELU = 6,
    SOFTMAX = 7,
    SWISH = 8,
    LINEAR = 9

    @staticmethod
    def all():
        return [NetworkActivationFunction.ELU,
                NetworkActivationFunction.LEAKY_RELU,
                NetworkActivationFunction.LINEAR,
                NetworkActivationFunction.RELU,
                NetworkActivationFunction.SIGMOID,
                NetworkActivationFunction.SOFTMAX,
                NetworkActivationFunction.SOFTPLUS,
                NetworkActivationFunction.SWISH,
                NetworkActivationFunction.TANH]

    @staticmethod
    def get_type_by_name(name):
        for activation_function in NetworkActivationFunction.all():
            if activation_function.name.lower() == name.lower():
                return activation_function
        return None


class NetworkOptimizer(enum.Enum):
    ADAM = 1,
    ADAMAX = 2,
    NADAM = 3,
    RMSPROP = 4,

    @staticmethod
    def all():
        return [NetworkOptimizer.ADAM, NetworkOptimizer.ADAMAX, NetworkOptimizer.RMSPROP, NetworkOptimizer.NADAM]

    @staticmethod
    def get_type_by_name(name):
        for optimizer_type in NetworkOptimizer.all():
            if optimizer_type.name.lower() == name.lower():
                return optimizer_type
        return None

