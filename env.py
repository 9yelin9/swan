import re

num_thread = 1

path_swann = '/home/9yelin9/R2Ir2O7/hf3/swann'
path_lat, path_band, path_fig = 'swann/lat', 'swann/band', 'swann/fig'
lat_col = ['i', 'j', 'k', 'p', 'q', 't_real', 't_imag']
lat_dtype = {'i':'i', 'j':'i', 'k':'i', 'p':'i', 'q':'i', 't_real':'d', 't_imag':'d'}
lat_format = ['%5d', '%5d', '%5d', '%5d', '%5d', '%12.6f', '%12.6f']

def ReSub(pattern, string):
    return float(re.sub(pattern, '', re.search('%s[-]?\d+[.]\d+' % pattern, string).group()))

def ReSubInt(pattern, string):
    return int(re.sub(pattern, '', re.search('%s[-]?\d+' % pattern, string).group()))

