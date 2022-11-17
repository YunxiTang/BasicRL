import torch

class LSTMCell(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LSTMCell, self).__init__()
        self.weight_ih = torch.nn.Parameter(torch.rand(4*out_dim, in_dim))
        self.weight_hh = torch.nn.Parameter(torch.rand(4*out_dim, out_dim))
        self.bias = torch.nn.Parameter(torch.zeros(4*out_dim,))

    def forward(self, inputs, h, c):
        ifgo = self.weight_ih @ inputs + self.weight_hh @ h + self.bias
        i, f, g, o = torch.chunk(ifgo, 4)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return (new_h, new_c)

class LSTMLM(torch.nn.Module):
    def __init__(self, vocab_size, dim=17):
        super().__init__()
        self.cell = LSTMCell(dim, dim)
        self.embeddings = torch.nn.Parameter(torch.rand(vocab_size, dim))
        self.c_0 = torch.nn.Parameter(torch.zeros(dim))
    
    @property
    def hc_0(self):
        return (torch.tanh(self.c_0), self.c_0)

    def forward(self, seq, hc):
        loss = torch.tensor(0.)
        for idx in seq:
            loss -= torch.log_softmax(self.embeddings @ hc[0], dim=-1)[idx]
            hc = self.cell(self.embeddings[idx,:], *hc)
        return loss, hc
    
    def greedy_argmax(self, hc, length=6):
        with torch.no_grad():
            idxs = []
            for i in range(length):
                idx = torch.argmax(self.embeddings @ hc[0])
                idxs.append(idx.item())
                hc = self.cell(self.embeddings[idx,:], *hc)
        return idxs

if __name__ == '__main__':
    print('lstm in torch')
    torch.manual_seed(0)

    # As training data, we will have indices of words/wordpieces/characters,
    # we just assume they are tokenized and integerized (toy example obviously).

    import jax.numpy as jnp
    vocab_size = 43  # prime trick! :)
    training_data = jnp.array([4, 8, 15, 16, 23, 42])

    lm = LSTMLM(vocab_size=vocab_size)
    print("Sample before:", lm.greedy_argmax(lm.hc_0))

    bptt_length = 3  # to illustrate hc.detach-ing

    for epoch in range(101):
        hc = lm.hc_0
        totalloss = 0.
        for start in range(0, len(training_data), bptt_length):
            batch = training_data[start:start+bptt_length]
            loss, (h, c) = lm(batch, hc)
            hc = (h.detach(), c.detach())
            if epoch % 50 == 0:
                totalloss += loss.item()
            loss.backward()
            for name, param in lm.named_parameters():
                if param.grad is not None:
                    param.data -= 0.1 * param.grad
                    del param.grad
        if totalloss:
            print("Loss:", totalloss)

    print("Sample after:", lm.greedy_argmax(lm.hc_0))