import numpy as np
import pdb
import matplotlib as plt
plt.use('AGG')
import pylab as py
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import skimage.io
import skimage.filter
from skimage import feature

def normit(x):
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    return x

def PLOT(mfi,savepath):
    '''

    :param mfi: feature importance map of the digit
    :param savepath: path to result heatmap picture
    :return: -
    '''
    py.imshow(np.reshape(mfi, (16, 16)))
    py.savefig(savepath)

def PLOTnum(mfi, savepath, digit,prediction=1):
    '''

    :param mfi: feature importance map of the digit
    :param savepath: path to result heatmap picture
    :param digit: has to be in the final plotting shape (e.g. 16 x 16 for usps data)
    :return: -
    '''
    py.close("all")
    cmp = plt.cm.get_cmap('Greys')
    cmp._init()

    fig = py.imshow(normit(digit), cmap=cmp, origin='lower')#, vmin=0.4, vmax=0.7)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    cmap_used = py.get_cmap()
    cmap_used._init()
    cmap_used._init()

    py.imshow(np.reshape(normit(mfi), digit.shape))
    alphas = np.linspace(0.2, 0.9, cmap_used.N + 3)
    cmap_used._lut[40:190, 0:3] = [1, 1, 1]
    #
    cmap_used._lut[40:190, -1] = 0.4#alphas[30:180]  #
    py.tight_layout()
    py.colorbar()
    py.title(str(prediction))
    py.savefig(savepath, bbox_inches=0, orientation='landscape', pad_inches=0.1)

def PLOTnum3(mfis, savepath, digits,prediction=None):
    '''

    :param mfi: feature importance map of the digit
    :param savepath: path to result heatmap picture
    :param digit: has to be in the final plotting shape (e.g. 16 x 16 for usps data)
    :return: -
    '''
    py.close("all")
    cmp = plt.cm.get_cmap('Greys')
    cmp._init()

    for i,mfi in enumerate(mfis):
        print i
        py.subplot(2, len(mfis), i + 1)
        fig = py.imshow(vec2im(digits[i]),cmap = cmp)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)


    for i,mfi in enumerate(mfis):
        print i
        rgb = hm_to_rgb(mfi, X=digits[i])
        py.subplot(2, len(mfis), len(mfis) + 1 + i)
        fig = py.imshow(rgb,interpolation='spline16')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        if prediction != None:
            py.title(str(np.round(prediction[i],2)))

    py.savefig(savepath, bbox_inches=0, orientation='landscape', pad_inches=0.1)

def vec2im(V, shape = () ):
    # function taken from https://github.com/sebastian-lapuschkin/lrp_toolbox/blob/master/python/render.py
    '''
    Transform an array V into a specified shape - or if no shape is given assume a square output format.
    Parameters
    ----------
    V : numpy.ndarray
        an array either representing a matrix or vector to be reshaped into an two-dimensional image
    shape : tuple or list
        optional. containing the shape information for the output array if not given, the output is assumed to be square
    Returns
    -------
    W : numpy.ndarray
        with W.shape = shape or W.shape = [np.sqrt(V.size)]*2
    '''
    if len(shape) < 2:
        shape = [np.sqrt(V.size)]*2

    return np.reshape(V, shape)


def enlarge_image(img, scaling = 3):
    # function taken from https://github.com/sebastian-lapuschkin/lrp_toolbox/blob/master/python/render.py
    '''
    Enlarges a given input matrix by replicating each pixel value scaling times in horizontal and vertical direction.
    Parameters
    ----------
    img : numpy.ndarray
        array of shape [H x W] OR [H x W x D]
    scaling : int
        positive integer value > 0
    Returns
    -------
    out : numpy.ndarray
        two-dimensional array of shape [scaling*H x scaling*W]
        OR
        three-dimensional array of shape [scaling*H x scaling*W x D]
        depending on the dimensionality of the input
    '''

    if scaling < 1 or not isinstance(scaling,int):
        print 'scaling factor needs to be an int >= 1'

    if len(img.shape) == 2:
        H,W = img.shape

        out = np.zeros((scaling*H, scaling*W))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling] = img[h,w]

    elif len(img.shape) == 3:
        H,W,D = img.shape

        out = np.zeros((scaling*H, scaling*W,D))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling,:] = img[h,w,:]

    return out


def hm_to_rgb(R, X = None, scaling = 3, shape = (), sigma = 2, cmap = 'jet', normalize = True):
    # function taken from https://github.com/sebastian-lapuschkin/lrp_toolbox/blob/master/python/render.py
    '''
    Takes as input an intensity array and produces a rgb image for the represented heatmap.
    optionally draws the outline of another input on top of it.
    Parameters
    ----------
    R : numpy.ndarray
        the heatmap to be visualized, shaped [M x N]
    X : numpy.ndarray
        optional. some input, usually the data point for which the heatmap R is for, which shall serve
        as a template for a black outline to be drawn on top of the image
        shaped [M x N]
    scaling: int
        factor, on how to enlarge the heatmap (to control resolution and as a inverse way to control outline thickness)
        after reshaping it using shape.
    shape: tuple or list, length = 2
        optional. if not given, X is reshaped to be square.
    sigma : double
        optional. sigma-parameter for the canny algorithm used for edge detection. the found edges are drawn as outlines.
    cmap : str
        optional. color map of choice
    normalize : bool
        optional. whether to normalize the heatmap to [-1 1] prior to colorization or not.
    Returns
    -------
    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    X = normit(X)# (X + 1.) / 2.
    #create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))



    R = normit(R)

    R = enlarge_image(vec2im(R,shape), scaling)
    rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    #rgb = repaint_corner_pixels(rgb, scaling) #obsolete due to directly calling the color map with [0,1]-normalized inputs

    if not X is None: #compute the outline of the input
        X = enlarge_image(vec2im(X,shape), scaling)

        xdims = X.shape
        Rdims = R.shape

        if not np.all(xdims == Rdims):
            print 'transformed heatmap and data dimension mismatch. data dimensions differ?'
            print 'R.shape = ',Rdims, 'X.shape = ', xdims
            print 'skipping drawing of outline\n'
        else:
            edges = feature.canny(X, sigma=2.)
            edges = np.invert(np.dstack([edges]*3))*1.0
            rgb *= edges # set outline pixels to black color
    return rgb

def PFplot(values,names):
    py.cla()
    fs = 34
    py.close("all")
    val_means = []
    val_std   = []
    for r in range(len(names)):
        val_means.append(np.mean(values[r:len(names):len(values)],0))
        val_std.append(np.std(values[r:len(names):len(values)],0))
    for i,val in enumerate(val_means):
        py.plot(val_means[i], linewidth=5.0, alpha=0.6, label=names[i])
    py.xlabel("Flipping pixels", fontsize=fs)
    py.ylabel("Score", fontsize=fs)
    py.xticks(range(50, len(val_means[0]), 100), fontsize=fs)
    py.yticks(fontsize=fs)
    py.legend(loc='best',fontsize=20, fancybox=True, framealpha=0.7)
    py.tight_layout()
    py.savefig("results/pf.pdf")

def PF():
    values=[]
    for r in range(200):
        for s, samples in enumerate(Samplesets):
            for i in range(len(metric)):
                fobj = open("results/mfi_sample_" + str(s) + ".pkl", 'rb')
                mfi = pickle.load(fobj)
                fobj.close()
                values.append(tools.pixel_flipping(clf,np.mean(x,axis=0),mfi,100))
        values.append(tools.pixel_flipping(clf, np.mean(x, axis=0), np.random.uniform(0, 1, len(mfi)), 100))
    names = ["random samples","training samples","random mfi"]
    fobj = open("results/pf.pkl","wb")
    pickle.dump([values,names],fobj)
    fobj.close()