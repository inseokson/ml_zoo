import numpy as np
import torch
from torch.nn.parameter import Parameter


class LeNet(torch.nn.Module):
    def __init__(self, n_feature):
        super(LeNet, self).__init__()

        self.extractor = torch.nn.Sequential(
            torch.nn.ZeroPad2d(2),
            torch.nn.Conv2d(1, 6, 5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(16, 120, 5),
            torch.nn.Tanh(),
        )

        self.classfier = torch.nn.Sequential(
            torch.nn.Linear(120, 64), torch.nn.Tanh(), torch.nn.Linear(64, n_feature)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.classfier(x)

        return x


class Tree(torch.nn.Module):
    def __init__(self, n_depth, n_feature, n_class):
        super(Tree, self).__init__()

        self.n_depth = n_depth
        self.n_class = n_class

        n_decision_node = 2 ** n_depth - 1
        self.mask = np.random.choice(n_feature, n_decision_node, replace=False)

        self.n_terminal_node = 2 ** n_depth

        self._pi = Parameter(
            torch.rand((self.n_terminal_node, n_class)), requires_grad=False
        )
        self._pi /= self.pi.sum(axis=1).unsqueeze(1)

        self.tree = torch.nn.Sigmoid()

    @property
    def pi(self):
        return self._pi

    @pi.setter
    def pi(self, pi):
        self._pi.data = pi

    def forward(self, x):
        n_batch = x.shape[0]
        x_selected = x[:, self.mask]

        decisions_left = self.tree(x_selected)
        decisions_left = decisions_left.unsqueeze(2)
        decisions_right = 1 - decisions_left
        decisions = torch.cat((decisions_left, decisions_right), axis=2)

        mu = torch.ones(n_batch, 1, 1, device=x.device)

        idx_begin, idx_end = 0, 1
        for i in range(self.n_depth):
            mu_ = mu.reshape(n_batch, -1, 1).repeat(1, 1, 2)
            decision = decisions[:, idx_begin:idx_end, :]

            mu = mu_ * decision

            idx_begin = idx_end
            idx_end = idx_begin + 2 ** (i + 1)

        mu = mu.view(n_batch, self.n_terminal_node)
        prob = torch.mm(mu, self._pi)

        return mu, prob


class Forest(torch.nn.Module):
    def __init__(self, n_tree, n_depth, n_feature, n_class):
        super(Forest, self).__init__()

        self.forest = torch.nn.ModuleList()
        for _ in range(n_tree):
            self.forest.append(Tree(n_depth, n_feature, n_class))

    def forward(self, x):
        probs = []
        for tree in self.forest:
            _, prob_ = tree(x)
            prob_ = prob_.unsqueeze(2)
            probs.append(prob_)
        probs = torch.cat(probs, 2)
        prob = torch.mean(probs, 2)

        return prob


class DeepNeuralDecisionForests(torch.nn.Module):
    def __init__(self, extractor, n_tree, n_depth, n_feature, n_class):
        super(DeepNeuralDecisionForests, self).__init__()

        self.extractor = extractor
        self.classifier = Forest(n_tree, n_depth, n_feature, n_class)

    def forward(self, x):
        out = self.extractor(x)
        out = self.classifier(out)

        return out
