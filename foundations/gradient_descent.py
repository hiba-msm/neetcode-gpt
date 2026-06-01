class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        minim =  init

        for _ in range( iterations):
            deriv= 2* minim
            minim = minim - (learning_rate * deriv)
        return round(minim, 5)

        pass
