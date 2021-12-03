from fastai.vision.all import *
from fastbook import *
import sys
sys.path.append('/home/msadmin/notebooks/msc8001/src')
from dense_unet import *
from fastai import *
import argparse
#import dill

class CrossEntropyLossFlat(BaseLoss):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    y_int = True
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)
    
    
if __name__ == '__main__':
    
    # Create the parser
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    my_parser.add_argument('Path',
                           metavar='path',
                           type=str,
                           help='the path to training images')

    my_parser.add_argument('Height',
                           metavar='height',
                           type=int,
                           help='the height of training images')
    
    my_parser.add_argument('Width',
                           metavar='width',
                           type=int,
                           help='the width of training images')
    
    my_parser.add_argument('Projection',
                           metavar='proj',
                           type=str,
                           help='the name of the projection')    
    
    
    # Execute the parse_args() method
    args = my_parser.parse_args()

    input_path = args.Path
    
    img_shape = args.Height, args.Width 
    
    codes = ['Background', 'CV']
    
    #path = Path("../data/usq/imageprojections/all/images")
    
    path = input_path
    #projection = Path(input_path).parents[0].stem
    projection = args.Projection
    
    def get_msk(fname):
        "Grab a mask from a `filename` and adjust the pixels based on `pix2class`"
        fn = str(fname).replace('images','masks')
        msk = np.array(PILMask.create(fn))
        msk = (msk > 100)
        msk = (msk * 1).astype(np.uint8)
        return PILMask.create(msk)

    def get_y(o):
        return get_msk(o)


    dblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=get_y,
                   #item_tfms=Resize(224),
                   item_tfms=Resize(img_shape, method=ResizeMethod.Pad, pad_mode = PadMode.Zeros),
                   batch_tfms=[Normalize.from_stats(*imagenet_stats)])
    
    dls = dblock.dataloaders(path, bs=2)

        # try progressive resizing
    def get_dls(bs, size):
        dblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                       get_items=get_image_files,
                       get_y=get_y,
                       item_tfms=Resize((size), method=ResizeMethod.Pad, pad_mode = PadMode.Zeros),
                       batch_tfms=[Normalize.from_stats(*imagenet_stats)])

        return dblock.dataloaders(path, bs=bs)
    
    encoder = nn.Sequential(*list(models.densenet121().children())[0])
    unet = DynamicUnet(encoder, n_classes=2, img_size=img_shape, blur=False, blur_final=False,
                        self_attention=False, y_range=None, norm_type=NormType,
                        last_cross=True,
                        bottle=False)
    
    weights = torch.tensor([[0.5] + [1.5]]).cuda()
    loss_func = CrossEntropyLossFlat(weight=weights, axis=1)
    
    learn = Learner(dls, unet, loss_func=loss_func, metrics=[foreground_acc, Dice()])
    lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    
    #learn.fit_one_cycle(100, lr_max=slice(lrs.minimum, lrs.valley), 
    #                    cbs=[ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=5, min_lr=lrs.minimum),
    #                                         EarlyStoppingCallback(monitor='dice', min_delta=0.01, patience=15),
    #                                         SaveModelCallback(monitor='dice', min_delta=0.01)])
    
    learn.fit_one_cycle(100, lr_max=slice(lrs.minimum, lrs.valley), 
                        cbs=[ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=5, min_lr=lrs.minimum),
                                             EarlyStoppingCallback(monitor='foreground_acc', min_delta=0.01, patience=15),
                                             SaveModelCallback(monitor='foreground_acc', min_delta=0.01)])
    
    #learn.export(f'dense_unet_mip_images_model_{projection}')
    learn.save(f'dense_unet_mip_images_model_{projection}')
    