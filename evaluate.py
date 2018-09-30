import numpy
import torch

def image_to_text(captions, images, npts=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] / 5
        npts = int(npts)

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].unsqueeze(0)

        # Compute scores
        d = torch.mm(im, captions.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()

        # Score
        rank = 1e20
        # find the highest ranking
        for i in range(5*index, 5*index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1

    print("		* Image to text scores: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
    return r1+r5+r10

def text_to_image(captions, images, npts=None):
    if npts == None:
    	npts = images.size()[0] / 5
    	npts = int(npts)

    ims = torch.cat([images[i].unsqueeze(0) for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5*index : 5*index + 5]

        # Compute scores
        d = torch.mm(queries, ims.t())
        for i in range(d.size()[0]):
            d_sorted, inds = torch.sort(d[i], descending=True)
            inds = inds.data.squeeze(0).cpu().numpy()
            ranks[5 * index + i] = numpy.where(inds == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    
    print("		* Text to image scores: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
    return r1+r5+r10

 
