import numpy as np

def normalize(v):
    s = np.std(v)
    v -= np.mean(v)
    return v/s

def square_diff(v1, v2):
    result = []
    # pad v1 so the comparison can be made for all possible v1 x
    v1p = np.pad(v1, (0,len(v2)-1), 'constant', constant_values=0)
    # slide v2 on v1 and compute square of differences
    for i in range(len(v1)):
        v2p = v1p.copy()
        v2p[i:i+len(v2)]=v2
        corr = np.sum(np.square(v1p-v2p))
        result.append(corr)
    return np.array(result)

def get_ref_signal(skyline, fov):
    ref_signal = normalize(np.append(skyline, skyline[:2*fov]))
    ref_signal /=len(ref_signal)
    return ref_signal

def get_best_azimuth(sqd, fov):
    azimuth = (np.argmin(sqd)+fov/2)%360
    return azimuth

def skylines_to_azimuth(ref_skyline, img_skyline):
    # compute sliding square difference between reference signal (panoramic skyline) and shifted/noisy signal (skyline from image)
    ref_signal = get_ref_signal(ref_skyline, len(img_skyline))
    img_signal = normalize(img_skyline)
    sqd = square_diff(ref_signal, img_signal)
    azimuth = get_best_azimuth(sqd, len(img_skyline))
    return azimuth
