class Algorithm:
    discount_factor = 0.0
    algorithm_name = None

    def __init__(self, args={}):
        for key in args:
            setattr(self, key, args[key])

    def calculate_target_value(self, a, s_, r, active, network):
        return 0

    def get_target_error(self, s, a, s_, r, active, network):
        return 0