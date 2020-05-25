class Algorithm:
    discount_factor = 0.0
    policy = None
    lambda_val = 0.0
    e_traces = {}
    enable_e_traces = False

    def __init__(self, args=None):
        if args is None:
            args = {}
        for key in args:
            setattr(self, key, args[key])

    def add_state(self, s, network_type):
        if s is None:
            pass
        s = tuple(s)
        if s not in self.e_traces:
            self.e_traces.update({s: [0] * self.policy.num_actions})

    def calculate_target_error(self, s, a, s_, r, terminal, network, current_q):
        s = tuple(s)
        s_ = tuple(s_)
        self.add_state(s, network.network_type)
        coin_side = network.determine_coin_side()
        if self.enable_e_traces:
            self.e_traces[s][a] += 1

        target_error = self.get_target_value(a, s_, r, terminal, current_q, coin_side) - self.get_scalar(s, a,
                                                                                                         coin_side,
                                                                                                         network)
        if self.enable_e_traces:
            target_error *= self.e_traces[s][a]
            for state in self.e_traces:
                for a_ in range(self.policy.num_actions):
                    self.e_traces[state][a_] *= (self.lambda_val * self.discount_factor)

        return target_error, coin_side

    def get_target_error(self, s, a, s_, r, terminal, network, current_q):
        return self.calculate_target_error(s, a, s_, r, terminal, network, current_q)

    def get_target_value(self, a, s_, r, active, current_q, coin_side):
        a_ = self.get_potential_action(current_q, coin_side, s_, a)
        return r + (self.discount_factor * self.get_scalar(s_, a_, current_q.network_type.value - 1 - coin_side,
                                                           current_q) * active)

    def get_potential_action(self, current_q, coin_side, s, a):
        return -1

    def get_scalar(self, s, a, coin_side, network):
        q_mat = network.get_action_values(s, coin_side)
        return q_mat[0, a]
