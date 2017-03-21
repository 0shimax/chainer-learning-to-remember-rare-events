"""
networkの出力サイズが1×1となると想定した場合に、入力のサイズがどの程度になる必要があるか
"""

""" [filter size, stride, padding] """

convnet =[[7,2,3], [3,2,0], \

          [1,1,0], [3,1,1], \
          [1,1,0], [3,1,1], \
          [1,1,0], [3,1,1], \
          [3,2,0], \

          [1,1,0], [3,1,1], \
          [1,1,0], [3,1,1], \
          [1,1,0], [3,1,1], \
          [1,1,0], [3,1,1],
          ]
layer_name = ['c1', 'm1',
                'f1c1', 'f1c2',
                'f2c1', 'f2c2',
                'f3c1', 'f2c2',
                'm2',
                'f4c1', 'f4c2',
                'f5c1', 'f5c2',
                'f6c1', 'f6c2',
                'f7c1', 'f7c2',
            ]

imsize = 352




# def outFromIn(isz, layernum = 9, net = convnet):
#     if layernum>len(net): layernum=len(net)
#
#     totstride = 1
#     insize = isz
#     #for layerparams in net:
#     for layer in range(layernum):
#         fsize, stride, pad = net[layer]
#         outsize = (insize - fsize + 2*pad) / stride + 1
#         insize = outsize
#         totstride = totstride * stride
#
#     RFsize = isz - (outsize - 1) * totstride
#
#     return outsize, totstride, RFsize


def outFromIn(isz, layernum = 9, net = convnet):
    if layernum>len(net): layernum=len(net)

    totstride = 1
    insize = isz
    #for layerparams in net:
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2*pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride

def inFromOut( layernum = 9, net = convnet):
    if layernum>len(net): layernum=len(net)
    outsize = 1
    #for layerparams in net:
    for layer in reversed(range(layernum)):
        ksize, stride, pad = net[layer]
        outsize = ((outsize -1)* stride) + ksize
        # print(layer)
        # print(ksize, stride, pad)
        # print(outsize)
    RFsize = outsize
    return RFsize

if __name__ == '__main__':
    print("layer output sizes given image = %dx%d" % (imsize, imsize))
    for i in range(len(convnet)):
        p = outFromIn(imsize,i+1)
        rf = inFromOut(i+1)
        # print("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d"%(layer_name[i], p[0], p[1], p[2]))
        print("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d"%(layer_name[i], p[0], p[1], rf))
