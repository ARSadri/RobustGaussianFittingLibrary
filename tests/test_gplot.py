import RobustGaussianFittingLibrary as rgflib
import numpy as np

def test_gplot():
	print('test_gplot')
	x = np.arange(0, 2, 0.1)
	mu = x**2
	std = mu**0.5

	Gplot = rgflib.misc.plotGaussianGradient()
	Gplot.addPlot(x = x, mu = mu, std = std, 
	              gradient_color = (1, 0, 0), 
				  label = 'red',
				  mu_color = (0.75, 0, 0, 1),
				  mu_linewidth = 3)
	Gplot.show()
	
if __name__=='__main__':
	test_gplot()