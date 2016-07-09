import numpy as np

def overlap_squares(patch_a, patch_b, overlap):
    '''
    checks for overlap of bboxes specified as tuples (x, y, size)
    check if squares overlap by IoU >= overlap
    '''    
    # intersection
    y_in = np.intersect1d(range(patch_a[1], patch_a[1]+patch_a[2]), range(patch_b[1], patch_b[1]+patch_b[2]))
    x_in = np.intersect1d(range(patch_a[0], patch_a[0]+patch_a[2]), range(patch_b[0], patch_b[0]+patch_b[2]))
    intersection = float(len(y_in))*float(len(x_in))

    # union
    union = float(patch_a[2])**2+float(patch_b[2])**2 - intersection

    # print intersection/union
    if intersection/union <= overlap:
        return False
    else:
        return True
