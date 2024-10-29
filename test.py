"""
To get TP working we need two things:
1. all-gather/identity in forward/backward passes
2. shard along correct axises
3. replicate the data

To do #1, we need to code at the level of forward/backward pass.
This means we need to define new functions for tensor in function.py.
We do not need to touch nn.Linear.

But we do not really want to define new functions. Really we should be able to
override the forward/backward pass to the identity function when we need it.
Overriding is better than making a whole new thing.

Can we also use multilazybuffer.r to reduce? We want the reduction to be done once
in the forward pass and no times in the backwards pass.

Maybe first draft:
Create MLP where the forward pass is correct.


All reduce is incorrect. I'm not sure where those numbers are coming from.
1. How do I see the elements in a tensor on each GPU? If I call x_sharded.numpy() it shows me all of the elements on a single GPU.
2. I need to use pad to add elements, like we do in extra/multitensor.py to get all reduce to work/
3. Maybe I can just use multi.copy_to_device() instead? This automatically takes care of the padding for me.

"""
import math
from tinygrad.nn.datasets import mnist
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, Device
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist
from tinygrad.multi import all_reduce, MultiLazyBuffer
from tinygrad.ops import MetaOps, ReduceOps


import numpy as np
from icecream import ic


GPUS = [f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 4))]

# class FeedForward:
#   def __init__(self, dim:int, hidden_dim:int, linear=nn.Linear):
#     self.w1 = linear(dim, hidden_dim, bias=False)
#     self.w2 = linear(hidden_dim, dim, bias=False)
#     self.w3 = linear(dim, hidden_dim, bias=False) # the gate in Gated Linear Unit

#   def __call__(self, x:Tensor) -> Tensor:
#     return self.w2(self.w1(x).silu() * self.w3(x)) # SwiGLU [arxiv/2002.05202, eq (5)]

# if __name__ == "__main__":
#   X_train, Y_train, X_test, Y_test = mnist()
#   ic(X_test)
#   X_test.shard_(GPUS, axis=0)
#   ic(X_test, X_test.numpy()[0, 0, 10, 10:-5])


B, S = 1, 1
D1 = D2 = D3 = 4
d1 = f"{Device.DEFAULT}:1"
d2 = f"{Device.DEFAULT}:2"
devices = (d1, d2)


def test_data_parallelism():
    X = Tensor.kaiming_uniform(B, S, D1).realize()
    W = Tensor.kaiming_uniform(D1, D2).realize()

    Xs = X.shard(devices, 0) # shard along batch dimension
    Ws = W.shard(devices, None) # don't shard, just turn the lazybuffer into a multilazybuffer for compatability

    O = (Xs@Ws)
    np.testing.assert_allclose(X.numpy() @ W.numpy(), O.to(Device.DEFAULT).numpy(), atol=1e-5)

def test_weight_parallelism():
    """This is not a real parallelism strategy, it is just testing sharding the weights.
    (This is not really model parallelism which is layer-wise or tensor parallelism)."""
    X = Tensor.kaiming_uniform(B, S, D1).realize()
    W = Tensor.kaiming_uniform(D1, D2).realize()

    Xs = X.shard(devices, None) # don't shard, just turn the lazybuffer (lb) into a multilazybuffer (mlb) for compatability
    Ws = W.shard(devices, 0) # shard along rows
    Os = (Xs@Ws) # multiply two sharded matrices (eventhough X is not really sharded)

    O = Os.to(Device.DEFAULT) # unshard (turns mlb to lb)
    np.testing.assert_allclose(X.numpy() @ W.numpy(), O.numpy(), atol=1e-5)


def test_all_reduce_1():
    W = Tensor.kaiming_uniform(D1, D2).realize()
    a = (W[0:2] + W[2:4]).repeat([2,1]).realize()
    ts = W.shard(devices, 0).realize()
    b = Tensor(MultiLazyBuffer(all_reduce(ReduceOps.SUM, ts.lazydata.lbs), 0))
    b.realize()
    np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

def test_all_reduce_2():
    W = Tensor.kaiming_uniform(D1, D2)
    Wshard = W.shard(devices, 0)
    Wred_sum = Tensor(MultiLazyBuffer(all_reduce(ReduceOps.SUM, Wshard.lazydata.lbs), 0))
    W_sum = (W[0:2] + W[2:4]).repeat([2,1])
    np.testing.assert_almost_equal(W_sum.numpy(), Wred_sum.numpy(), decimal=5)

def test_all_reduce_3():

    # class FeedForward:
    #     def __init__(self, dim:int, hidden_dim:int, linear=nn.Linear):
    #         self.w1 = linear(dim, hidden_dim, bias=False)
    #         self.w2 = linear(hidden_dim, dim, bias=False)
    #         self.w3 = linear(dim, hidden_dim, bias=False) # the gate in Gated Linear Unit

    #     def __call__(self, x:Tensor) -> Tensor:
    #         return self.w2(self.w1(x).silu() * self.w3(x)) # SwiGLU [arxiv/2002.05202, eq (5)]

    class Linear:
        def __init__(self, in_features, out_features, bias=True):
            bound = 1 / math.sqrt(in_features)
            self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
            self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

        def __call__(self, x:Tensor):
            return x.linear(self.weight.transpose(), self.bias)

    W = nn.Linear(D1, D2, bias=False).weight
    Wshard = W.shard(devices, 0)
    Wred_sum = Tensor(MultiLazyBuffer(all_reduce(ReduceOps.SUM, Wshard.lazydata.lbs), 0))
    W_sum = (W[0:2] + W[2:4]).repeat([2,1])
    np.testing.assert_almost_equal(W_sum.numpy(), Wred_sum.numpy(), decimal=5)

def test_ffn_1():
    """Check that the unsharded and sharded ffn output match."""

    # initialize tensors
    x = Tensor.kaiming_uniform(B, S, D1)
    w1 = Tensor.kaiming_uniform(D1, D2)
    w2 = Tensor.kaiming_uniform(D2, D3)

    # shard tensors
    x_sharded = x.shard(devices, None) # do not shard the data # todo: are we replicating data here implicitly via None?
    w1_sharded = w1.shard(devices, 1) # shard across columns
    w2_sharded = w2.shard(devices, 0) # shard across rows

    # compute sharded and unsharded ffn output
    o_sharded = x_sharded.linear(w1_sharded.T).gelu().linear(w2_sharded.T)
    o_true = x.linear(w1.T).gelu().linear(w2.T)

    # check that they match
    step = D1 // len(devices)
    for i, lb in enumerate(o_sharded.lazydata.real_lbs):
        o_partial = Tensor(lb) # the output on the ith GPU
        o_true_partial = o_true[:, :, i*step:(i+1)*step]
        np.testing.assert_almost_equal(o_partial.numpy(), o_true_partial.numpy(), decimal=5)


def test_ffn_2():
    """Check that the unsharded and sharded+all-reduce ffn output match."""

    # initialize tensors
    x = Tensor.kaiming_uniform(B, S, D1)
    w1 = Tensor.kaiming_uniform(D1, D2)
    w2 = Tensor.kaiming_uniform(D2, D3)

    # shard tensors
    x_sharded = x.shard(devices, None) # do not shard the data # todo: are we replicating data here implicitly via None?
    w1_sharded = w1.shard(devices, 1) # shard across columns
    w2_sharded = w2.shard(devices, 0) # shard across rows

    # compute unsharded and sharded+all_reduce ffn output
    o_sharded = x_sharded.linear(w1_sharded.T).gelu().linear(w2_sharded.T)
    ic(o_sharded, o_sharded.numpy())
    # o = o_sharded.lazydata.r(ReduceOps.SUM, (2,))
    # o = all_reduce(ReduceOps.SUM, o_sharded.lazydata.lbs)
    # o = Tensor(MultiLazyBuffer(all_reduce(ReduceOps.SUM, o_sharded.lazydata.lbs), 2))

    o_sharded.lazydata.lbs[1], o_sharded.lazydata.lbs[0] = o_sharded.lazydata.lbs[1].copy_to_device(0), o_sharded.lazydata.lbs[0].copy_to_device(1)
    ic(o_sharded, o_sharded.numpy())
    # o_sharded.lazydata.lbs[1].copy_to_device(0)

    o_true = x.linear(w1.T).gelu().linear(w2.T)
    for device in devices:
        ic(o_true, o_true.to(device))

    ic(o_sharded, o_true, o_sharded.numpy(), o_true.numpy())

    # check that they match
    step = D1 // len(devices)
    for i, lb in enumerate(o_sharded.lazydata.real_lbs):
        o_partial = Tensor(lb) # the output on the ith GPU
        o_true_partial = o_true[:, :, i*step:(i+1)*step]
        np.testing.assert_almost_equal(o_partial.numpy(), o_true_partial.numpy(), decimal=5)


def test_rmsnorm():
    B, T, embed_size = 4, 10, 20
    devices_2 = devices

    norm = nn.RMSNorm(embed_size)
    x = Tensor.rand((B, T, embed_size)).contiguous().realize()
    y = norm(x)

    # for norm layers, the correct way to shard weights is duplication
    norm_sharded = nn.RMSNorm(embed_size)
    norm_sharded.weight.shard_(devices_2, axis=None).realize()

    # if x is being sharded, then all-reduce is involved
    #
    # why do we need this realize?
    # if we didn't have realize(), then we'd never actually put the data on the different GPUs
    # no realize:
    # our computation graph says we shard the norm and shard x, run x through norm_sharded,
    # get the output, and then finally realize all of this.
    # Perhaps when we realize, we apply some optimizations
    # that say we don't actually
    x_sharded = x.shard(devices_2, axis=2).realize()
    y_shard = norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

    # if x is being duplicated, then the operations remain inside each GPU
    # which is the common case
    x_sharded = x.shard(devices_2, axis=None).realize()
    y_shard = norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
    # test_weight_parallelism()
    # test_weight_parallelism()
    # test_all_reduce_1()
    # test_all_reduce_2()
    # test_all_reduce_3()
    # test_ffn_1()
    # test_ffn_2()
    test_rmsnorm()
