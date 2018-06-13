import glob
import numpy  as np

def ModelIt(fromUser  = 'Default', inputfile):
	listofsoundfiles = glob.glob('/Users/adam/projects/soundeffects/data/soundeffectfiles/wav/*.wav')
	a = np.random.rand(len(listofsoundfiles))
	sf_score = list(zip(a, listofsoundfiles))
	sf_score.sort(key = lambda x:x[0], reverse=True)
	return sf_score[:5]
	


