
from kmodule.module_functions import *



class Feedforward_Block(nn.Module):
    def __init__(
        _,
        nch,
        mdic=None,
    ):
        super(Feedforward_Block,_).__init__()
        packdict(_,locals())
    def forward(_,x):
        x=fire_module('feedforward A',x,_.nch,_.nch,_.nch,mdic=_.mdic,hide=False)
        x=fire_module('feedforward B',x,_.nch,_.nch,_.nch,mdic=_.mdic,hide=False)
        return x



class Attention_Block(nn.Module):
    def __init__(
        _,
        nch,
        mdic=None,
    ):
        super(Attention_Block,_).__init__()
        packdict(_,locals())
    def forward(_,x):
        h=image_height=x.size()[-2]
        w=image_width=x.size()[-1]
        x_in=x
        x=maxpool('attention input',x_in,kernel_size=7,stride=3,mdic=_.mdic)
        x=conv2d('attention 7x7',x,out_channels=nch*2,kernel_size=7,stride=3,padding=3,mdic=_.mdic,activation=nn.Sigmoid())
        x=upsample(d2n('attention u1 (',h,'x',w,')'),x,image_height=h,image_width=w,mode='bilinear',mdic=_.mdic)
        return x



class Multiscale_Block(nn.Module):
    def __init__(
        _,
        feedforward_block=None,
        attention_block=None,
        h_final=0,
        w_final=0,
        sizes=[],
        mdic=None,
    ):
        super(Multiscale_Block,_).__init__()
        packdict(_,locals())
    def forward(_,x):
        bs=x.size()[0]
        nc=x.size()[1]
        x_base=torch.zeros(bs,nc,_.h_final,_.w_final)
        for h,w in _.sizes:
            cg(h,w,r=0)
            x=upsample(
                d2n('u3 (',h,'x',w,')'),
                x,
                image_height=h,
                image_width=w,
                mode='bilinear',
                mdic=_.mdic,
            )
            x=feedforward_block(x)
            x_attention=attention_block(x)
            x=torch.mul(x,x_attention)
            x=upsample(
                d2n('output final (',_.h_final,'x',_.w_final,')'),
                x,
                image_height=_.h_final,
                image_width=_.w_final,
                mode='bilinear',
                mdic=mdic,
            )
            print(x_base.size(),x.size())
            x_base+=x
            print(2*'\n')
        x_base/=len(_.sizes)
        return x_base




class Skip_Connection_Block(nn.Module):
    def __init__(
        _,
        block=None,
        skip_connections=[],
        h_final=0,
        w_final=0,
        n_out_channels=0,
        mdic=None,
    ):
        super(Skip_Connection_Block,_).__init__()
        packdict(_,locals())
    def forward(_,x):

        x=_.block(x)
        
        xs=[]

        for x in [x]+_.skip_connections:
            xs.append(
                upsample(
                    d2n('skip connection (',_.h_final,'x',_.w_final,')'),
                    x,
                    image_height=_.h_final,
                    image_width=_.w_final,
                    mode='bilinear',
                    mdic=mdic,
                )
            )     
        for y in xs:
            print(y.size())
        x=torch.cat(xs,axis=1)
        x=conv2d('1x1 skip',x,_.n_out_channels,kernel_size=1,mdic=_.mdic,hide=False)
        
        return x


"""
-Feed-forward from input to x

-Attention, large RFs, contracted, then upscaped to multiply with x

-Multi-scape, increase and decrease resolution of input, send through feed-forward
and attention, then resize and add together.

-gather skip connections, blend them together at end with concat and 1x1 convolution
"""



if __name__=='__main__':
    print(10*'\n')
    bs=1
    nin=16
    nch=8 
    mdic=nn.ModuleDict()
    feedforward_block=Feedforward_Block(nch,mdic)
    attention_block=Attention_Block(nch,mdic)
    multiscale_block=Multiscale_Block(
        feedforward_block,
        attention_block,
        67,
        133,
        [(256,512),(128,256),(64,128),(32,64),(16,32)],
        mdic,
    )
    skip_connection_block=Skip_Connection_Block(
        block=multiscale_block,
        skip_connections=[torch.from_numpy(na(rndn(bs,nin,128,256))).float()],
        h_final=56,
        w_final=147,
        n_out_channels=7,
        mdic=mdic,
    )



    x=torch.from_numpy(na(rndn(bs,nin,128,256))).float()
    #x=multiscale_block(x)
    x=skip_connection_block(x)
    describe_tensor(x,'end')


#EOF
