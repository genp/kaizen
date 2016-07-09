import time

class AppTimer:

	def start(self):
		self.st = time.time()

	def stop(self, file):
		print >> file, str(time.time()-self.st)
