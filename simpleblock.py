
from kmodule.module_functions import *


class Simple_Block(nn.Module):
    def __init__(
        _,
        nch,
        show='once',
        mdic=None,
    ):
        super(Simple_Block,_).__init__()
        packdict(_,locals())
    def forward(_,x):
        describe_input(x,_,show,0)
        x=conv2d(
            'a',
            x,
            out_channels=8,
            kernel_size=4,
            stride=3,
            padding=2,
            mdic=_.mdic,
            activation=nn.ReLU(True),
            show=show,
        )
        x=conv2d(
            'b',
            x,
            out_channels=16,
            kernel_size=4,
            stride=3,
            padding=2,
            mdic=_.mdic,
            activation=nn.ReLU(True),
            show=show,
        )
        x=conv2d(
            'c',
            x,
            out_channels=32,
            kernel_size=4,
            stride=3,
            padding=2,
            mdic=_.mdic,
            activation=nn.ReLU(True),
            show=show,
        )
        x=conv2d(
            'd',
            x,
            out_channels=64,
            kernel_size=4,
            stride=3,
            padding=2,
            mdic=_.mdic,
            activation=nn.ReLU(True),
            show=show,
        )
        x=conv2d(
            'e',
            x,
            out_channels=29,
            kernel_size=4,
            stride=3,
            padding=1,
            mdic=_.mdic,
            activation=nn.ReLU(True),
            show=show,
        )
        describe_output(x,_,show,0)
        return x





if __name__=='__main__':
    print(10*'\n')
    show='once'
    _show=straskys("""
        Simple_Block
    """)
    bs=1
    nin=3
    nch=8 
    mdic=nn.ModuleDict()
    simple_block=Simple_Block(nch,mdic=mdic,show=show)
    xin=torch.from_numpy(na(rndn(bs,nin,128,128))).float()
    x=simple_block(xin)
    describe_tensor(x,'end',show=show)
    x=simple_block(xin)
    describe_tensor(x,'end',show=show)

#EOF
