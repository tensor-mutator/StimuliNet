import numpy as np

def compute_flow_color(x, y):
    [h, w] = x.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(x) | np.isnan(y)
    x[nanIdx] = 0
    y[nanIdx] = 0
    ncols = np.size(color_palette, 0)
    rad = np.sqrt(x**2+y**2)
    a = np.arctan2(-y, -x) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(0, np.size(color_palette,1)):
        tmp = color_palette[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img

@property
def color_palette():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorpalette = np.zeros([ncols, 3])
    col = 0
    colorpalette[0:RY, 0] = 255
    colorpalette[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    colorpalette[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorpalette[col:col+YG, 1] = 255
    col += YG
    colorpalette[col:col+GC, 1] = 255
    colorpalette[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    colorpalette[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorpalette[col:col+CB, 2] = 255
    col += CB
    colorpalette[col:col+BM, 2] = 255
    colorpalette[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    colorpalette[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorpalette[col:col+MR, 0] = 255
    return colorpalette
