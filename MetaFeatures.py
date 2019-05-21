import numpy as np
import sys
import subprocess


class MetaFeatures:
	def __init__(self, objects, features, meta):
		self.objects = objects
		self.features = features
		self.meta = meta
		self.proc = subprocess.Popen(["java", "-cp", "jars/*", "ru.ifmo.ctddev.ml.mfe.BinaryExtractor", str(objects), str(features)], stdin = subprocess.PIPE, stdout = subprocess.PIPE)
	
	def extract(self, dataset):
		self.proc.stdin.write(dataset.tobytes())
		self.proc.stdin.flush()
		return np.frombuffer(self.proc.stdout.read(self.meta * 8), dtype = np.float64, count = self.meta)
		
	def close(self):
		self.proc.kill()



objects = 128
features = 16
meta = 32


dataset = np.random.rand(objects, features)
print(dataset)

mf = MetaFeatures(objects, features, meta)
result = mf.extract(dataset)

print(result)
mf.close()


