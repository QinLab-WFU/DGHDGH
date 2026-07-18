import torch


class GraphGenerator:
    def __init__(self, thresh=None):
        self.thresh = thresh

    def _get_A(self):
        if self.thresh is None:
            A = torch.ones_like(self.W_to_A)
        else:
            A = torch.ones_like(self.W_to_A).where(self.W_to_A > self.thresh, 0)

        return A

    def _get_W(self, x, ref_x):
        x = x - x.mean(dim=1).unsqueeze(1)
        norms = x.norm(dim=1)

        if ref_x is not None:
            ref_x = ref_x - ref_x.mean(dim=1).unsqueeze(1)
            ref_norms = ref_x.norm(dim=1)
        else:
            ref_x = x
            ref_norms = norms

        self.W_to_A = torch.mm(x, ref_x.t()) / torch.ger(norms, ref_norms)

        x1 = x.transpose(0, 1).unsqueeze(-1)
        x2 = ref_x.transpose(0, 1).unsqueeze(1)

        W = torch.bmm(x1, x2).permute(1, 2, 0)
        W = W / torch.ger(norms, ref_norms).unsqueeze(-1).repeat(1, 1, W.shape[-1])
        return W

    def get_graph(self, x, ref_x=None):
        W = self._get_W(x, ref_x)
        A = self._get_A()
        A = torch.nonzero(A)
        W = W[A[:, 0], A[:, 1]]

        return W, A, (x if ref_x is None else torch.cat([x, ref_x]))


if __name__ == "__main__":
    gg = GraphGenerator()
    x = torch.randn(4, 8)
    W, A, x = gg.get_graph(x)
    print(W.shape)
    print(A.shape)
    print(x.shape)
