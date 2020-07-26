import numpy as np

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
