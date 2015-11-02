from app.models import *
from tasks import *

b1 = Blob.query.get(1)
p1 = Patch.query.get(1)

ps1 = PatchSpec.query.get(1)
fs1 = FeatureSpec.query.get(1)

f1 = Feature.query.get(1)

e1 = Estimator.query.get(1)

c1 = Classifier.query.get(1)
