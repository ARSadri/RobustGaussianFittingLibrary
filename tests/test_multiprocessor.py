import time
import numpy as np
import RobustGaussianFittingLibrary as RGF

def calcMyThing(cnt, inputs, iterable_inputs):
    mask, = inputs
    data, weights = iterable_inputs
    vec_data = data
    vec_mask = mask[cnt]
    vec = vec_data[vec_mask == 1] * weights[vec_mask == 1]
    result = np.mean(vec) + np.std(vec)
    return(np.array([result]))

N = 100
data = np.random.randn(N, 1000000).astype('float32')
mask = np.floor(2*np.random.rand(N, 1000000)).astype('int8')
weights = np.random.randn(N, 1000000).astype('float32')

time_time = time.time()
intensity = RGF.misc.multiprocessor(
    calcMyThing, N, inputs = mask, iteratable_inputs = (data, weights),
    showProgress = True, max_cpu = 12).start() 
time_rec1 = time.time() - time_time
print('Now single CPU', flush = True)
time_time = time.time()
out = np.zeros(N)
for cnt in range(N):
    out[cnt] = calcMyThing(cnt, (mask, ), (data[cnt], weights[cnt]))
time_rec2 = time.time() - time_time

print(time_rec1, time_rec2, time_rec2/time_rec1)