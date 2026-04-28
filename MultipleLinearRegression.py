import numpy as np
import random as r
import math as m

class MultipleLinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = np.random.rand(1, self.x.shape[1])
        self.b = r.random()

    def f(self):
        return self.b + np.dot(self.x, self.w.T)

    def h(self, x):
        return self.b + np.dot(x, self.w.T)

    def loss(self):
        return (1 / self.y.shape[0]) * np.sum(np.pow(self.y - self.f(), 2))

    def batch_grad_desc(self, lr = 0.001, max_iters = 1000, show_iterations = False):
        iterations = []
        loss = []
        for curr_iters in range(max_iters):
            self.w = self.w - (lr * (-2 / self.y.shape[0]) * np.dot(self.x.T, self.y - self.f())).T
            self.b-=lr * (-2 / self.y.shape[0]) * np.sum(self.y - self.f())

            if show_iterations:
                print(curr_iters, self.loss())

            iterations.append(curr_iters)
            loss.append(self.loss())

        return self.b, self.w, iterations, loss

    def stoch_grad_desc(self, lr = 0.001, show_iterations = False):
        iterations = []
        loss = []
        for i in range(self.y.shape[0]):
            cost = self.h(self.x[i])
            for j in range(self.y.shape[1]):
                self.w[j] = self.w[j] - (lr * (cost - self.y[i]) * self.x[i][j])
            self.b-=lr * (cost - self.y[i])

            if show_iterations:
                print(i, self.loss())

            iterations.append(i)
            loss.append(self.loss())

        return self.b, self.w, iterations, loss


    def error(self, x, y):
        return m.sqrt(np.sum(np.pow(y - (self.b + np.dot(x, self.w.T)), 2)) / y.shape[0])