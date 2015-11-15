from flask.ext.wtf import Form
from wtforms import Field, BooleanField, IntegerField, FloatField, SelectField, TextField, HiddenField
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from wtforms.widgets import TextInput, HiddenInput
from wtforms.validators import Required, Optional, NumberRange
from flask_wtf.file import FileField, FileAllowed, FileRequired
from models import Blob, User, Dataset, PatchSpec, FeatureSpec, Keyword, Estimator, Classifier, Patch, Round
import extract

class ObjectField(Field):
    widget = HiddenInput()

    def __init__(self, label='', validators=None, model=None, **kwargs):
        super(ObjectField, self).__init__(label, validators, **kwargs)
        self.model = model

    def _value(self):
        if self.data:
            return str(self.data.id)
        else:
            return u''

    def process_formdata(self, valuelist):
        if valuelist:
            self.data = self.model.query.get(valuelist[0])
        else:
            self.data = []

class ObjectsField(Field):
    widget = HiddenInput()
    
    def __init__(self, label='', validators=None, model=None, **kwargs):
        super(ObjectsField, self).__init__(label, validators, **kwargs)
        self.model = model

    def _value(self):
        if self.data:
            return u', '.join(str(o.id) for o in self.data)
        else:
            return u''

    def process_formdata(self, valuelist):
        if valuelist:
            self.data = [self.model.query.get(s.strip())
                         for s in valuelist[0].split(',') if s.strip() != '']
        else:
            self.data = []

        
class FloatsField(Field):
    widget = TextInput()

    def _value(self):
        if self.data:
            return u', '.join(str(f) for f in self.data)
        else:
            return u''

    def process_formdata(self, valuelist):
        if valuelist:
            self.data = [float(s.strip()) for s in valuelist[0].split(',') if s.strip() != '']
        else:
            self.data = []


class LoginForm(Form):
    username = TextField('username', validators = [Required()])
    password = TextField('password', validators = [Required()])
    remember_me = BooleanField('remember_me', default = False)

    def __init__(self, *args, **kwargs):
        Form.__init__(self, *args, **kwargs)
        self.user = None

    def validate(self):
        if not Form.validate(self):
            return False

        user = User.find(self.username.data)
        if user is None:
            self.username.errors.append('Unknown username')
            return False

        if not user.check_password(self.password.data):
            self.password.errors.append('Invalid password')
            return False

        self.user = user
        return True

class SeedForm(Form):
    '''
    Specifies the img, bounding box, and keyword membership of a seed patch
    '''
    keyword = TextField('Keyword Name', validators = [Required()])
    seeds = HiddenField('seeds', validators = [Required()])

class BlobForm(Form):
    '''
    Form for uploading images
    '''
    file = FileField('file',  validators=[
        Required(),
        FileAllowed(['jpg', 'jpeg', 'png', 'gif'],
                    '.jpg, .jpeg, .png, or .gif only!')
    ])

class DatasetForm(Form):
    '''
    Upload an archive file of images for a new dataset
    '''
    file = FileField('archive',  validators=[
        FileRequired(),
        FileAllowed(['zip', 'tar', 'gz', 'bz2', 'txt', 'csv'],
                    'Upload a zip or tar file of images, or a txt file of image urls.')
    ])

    patchspec = QuerySelectField(get_label='name',
                                 allow_blank=True, blank_text='Patch?',
                                 query_factory=lambda:PatchSpec.query.all())
    featurespec = QuerySelectField(get_label='name',
                                   allow_blank=True, blank_text='Feature?',
                                   query_factory=lambda:FeatureSpec.query.all())

class DatasetAddSpecsForm(Form):
    '''
    Attach a new PatchSpec or FeatureSpec
    '''
    dataset = ObjectField(model=Dataset)
    patchspec = QuerySelectField(get_label='name',
                                 allow_blank=True, blank_text='Add PatchSpec',
                                 query_factory=lambda:PatchSpec.query.all())
    featurespec = QuerySelectField(get_label='name',
                                   allow_blank=True, blank_text='Add FeatureSpec',
                                   query_factory=lambda:FeatureSpec.query.all())


class PatchSpecForm(Form):
    '''
    Form for specifying how patches should be made from a dataset
    '''
    name = TextField('Name', validators = [Required()])

    width = IntegerField('Minimum width',
                         validators = [Required(), NumberRange(10,1000)])
    height = IntegerField('Minimum height',
                          validators = [Required(), NumberRange(10,1000)])

    xoverlap =  FloatField('Fraction to overlap when sliding over for next patch.',
                           validators = [Required(), NumberRange(0.01,0.99)])
    yoverlap =  FloatField('Fraction to overlap when sliding down for next patch.',
                           validators = [Required(), NumberRange(0.01,0.99)])


    scale = FloatField('Scale up patches by this factor',
                       validators = [Optional(), NumberRange(1.01,4)])

    flip = BooleanField('Create patches for mirror images')

    dataset = ObjectField(model=Dataset)

class FeatureSpecForm(Form):
    '''
    Form for specifying how to run features on a dataset.
    '''
    name = TextField('Name', validators = [Required()])
    cls = SelectField('Kind',
                      choices = [("extract."+k.__name__, k.__name__)
                                 for k in extract.kinds],
                      validators = [Required()])
    params = FloatsField('Parameters')

    dataset = ObjectField(model=Dataset)


class ClassifierForm(Form):
    '''
    Form for specifying and initializing new classifier for given keyword
    '''
    keyword = ObjectField(model=Keyword)
    dataset = QuerySelectField(get_label='name',
                               query_factory=lambda:Dataset.query.all())
    estimator = QuerySelectField(query_factory=lambda:Estimator.query.all())


class ActiveQueryForm(Form):
    classifier = ObjectField(model=Classifier, validators = [Required()])
    round = ObjectField(model=Round, validators = [Required()])
    time = HiddenField('time', validators = [Required()])
    user = HiddenField()
    location = HiddenField()
    nationality = HiddenField()
    confidence = HiddenField(validators = [Required()])
    pos_patches = ObjectsField(model=Patch)
    neg_patches = ObjectsField(model=Patch)

class DetectForm(Form):
    blobs = ObjectsField(model=Blob, validators = [Required()])
    # Eventually, form should say which to run against. For now, we'll run all
    # rounds = ObjectsField(model=Round, validators = [Required()])
