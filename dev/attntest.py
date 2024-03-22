
from knets.nets.utils.setup import *


upsamplers=nn.ModuleDict()
for i in range(2,258,2):
    upsamplers[str(i)]=nn.Upsample((i,i),mode='nearest')

class AttnTest(nn.Module):
    def __init__(
        _,

    ):
        super(AttnTest,_).__init__()

        _.a=nn.Conv2d(3, 16, kernel_size=3,stride=2,padding=1)
        _.b=nn.Conv2d(16, 3, kernel_size=1,stride=3)
        _.bb=nn.Conv2d(3, 3, kernel_size=7,stride=3,padding=3)
        #_.c=nn.Upsample((128,128),mode='nearest')
        _.d=nn.Conv2d(19, 3, kernel_size=1)

        _.a_=dl('\ta',True)
        _.b_=dl('\tb',True)
        _.bb_=dl('\tbb',True)
        _.c_=dl('\tc',True)
        _.d_=dl('\td',True)
        _.in_=dl('in',True)
        _.out_=dl('out',True)


    layers=nn.ModuleDict({
        'a':nn.Conv2d(3, 16, kernel_size=3,stride=2,padding=1),
        'b':nn.Conv2d(16, 3, kernel_size=1,stride=3),
        'bb':nn.Conv2d(3, 3, kernel_size=7,stride=3,padding=3),
        'd':nn.Conv2d(19, 3, kernel_size=1),
    })

    def forward(_, x):
        """
        xs={}
        for k in ['a','b','bb']:
            x=layers[k](x)
            xs[k]=x
        """
        _.in_(x)
        xa=_.a(x);      _.a_(xa)
        x=_.b(xa);      _.b_(x)
        xbb=_.bb(x);    _.bb_(xbb)
        w=xa.size()[-1]
        xc=upsamplers[str(w)](xbb);     _.c_(xc)
        #xc=_.c(xbb);     _.c_(xc)
        x=torch.cat([xa,xc],1); _.d_(x)
        x=_.d(x);       _.out_(x)   
        return x



def unit_test():
    net=AttnTest()
    w=256
    x=torch.from_numpy(na(rndn(1,3,w,w))).float()
    x=net(x)




if __name__=='__main__':
    unit_test()
#EOF
