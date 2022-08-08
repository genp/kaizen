from flask_wtf import FlaskForm as Form
from wtforms import Field, BooleanField, IntegerField, FloatField, SelectField, StringField, HiddenField
from wtforms_sqlalchemy.fields import QuerySelectField
from wtforms.widgets import TextInput, HiddenInput
from wtforms.validators import DataRequired, Optional, NumberRange
from flask_wtf.file import FileField, FileAllowed
from app.models import Blob, User, Dataset, PatchSpec, FeatureSpec, Keyword, Estimator, Classifier, Patch, Round
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
    username = StringField('username', validators = [DataRequired()])
    password = StringField('password', validators = [DataRequired()])
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
    keyword = StringField('Keyword Name', validators = [DataRequired()])
    seeds = HiddenField('seeds', validators = [DataRequired()])
    imgInfos = HiddenField('imgInfos', validators = [DataRequired()])

class BlobForm(Form):
    '''
    Form for uploading images
    '''
    file = FileField('file',  validators=[
        DataRequired(),
        FileAllowed(['jpg', 'jpeg', 'png', 'gif'],
                    '.jpg, .jpeg, .png, or .gif only!')
    ])

class DatasetForm(Form):
    '''
    Upload an archive file of images for a new dataset
    '''
    file = FileField('archive',  validators=[
        FileAllowed(['zip', 'tar', 'gz', 'bz2', 'txt', 'csv'],
                    'Upload a zip or tar file of images, or a txt file of image urls.')
    ])

    patchspec = QuerySelectField(get_label='name',
                                 allow_blank=True, blank_text='Patch?',
                                 query_factory=lambda:PatchSpec.query.all())
    featurespec = QuerySelectField(get_label='name',
                                   allow_blank=True, blank_text='Feature?',
                                   query_factory=lambda:FeatureSpec.query.all())
    val_percent =  FloatField('Fraction of the images to put in the val dataset',
                                default=.2,
                                validators = [DataRequired(), NumberRange(0.01,0.99)])

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
    name = StringField('Name', validators = [DataRequired()])

    width = IntegerField('Minimum width',
                         validators = [DataRequired(), NumberRange(10,1000)])
    height = IntegerField('Minimum height',
                          validators = [DataRequired(), NumberRange(10,1000)])

    xoverlap =  FloatField('Fraction to overlap when sliding over for next patch.',
                           validators = [DataRequired(), NumberRange(0.01,0.99)])
    yoverlap =  FloatField('Fraction to overlap when sliding down for next patch.',
                           validators = [DataRequired(), NumberRange(0.01,0.99)])


    scale = FloatField('Scale up patches by this factor',
                       validators = [Optional(), NumberRange(1.01,4)])

    flip = BooleanField('Create patches for mirror images')

    dataset = ObjectField(model=Dataset)

class FeatureSpecForm(Form):
    '''
    Form for specifying how to run features on a dataset.
    '''
    name = StringField('Name', validators = [DataRequired()])
    cls = SelectField('Kind',
                      choices = [("extract."+k.__name__, k.__name__)
                                 for k in extract.kinds],
                      validators = [DataRequired()])
    params = FloatsField('Parameters')

    dataset = ObjectField(model=Dataset)


class ClassifierForm(Form):
    '''
    Form for specifying and initializing new classifier for given keyword
    '''
    keyword = ObjectField(model=Keyword)
    dataset = QuerySelectField(get_label='name',
                               query_factory=lambda:Dataset.query.filter(Dataset.is_train == True).all())
    estimator = QuerySelectField(query_factory=lambda:Estimator.query.all())


class ActiveQueryForm(Form):
    classifier = ObjectField(model=Classifier, validators = [DataRequired()])
    round = ObjectField(model=Round, validators = [DataRequired()])
    time = HiddenField('time', validators = [DataRequired()])
    user = HiddenField()
    location = HiddenField()
    nationality = HiddenField()
    pos_patches = ObjectsField(model=Patch)
    neg_patches = ObjectsField(model=Patch)

class DetectForm(Form):
    blobs = ObjectsField(model=Blob, validators = [DataRequired()])
    # Eventually, form should say which to run against. For now, we'll run all
    # rounds = ObjectsField(model=Round, validators = [DataRequired()])

class EvaluateForm(Form):
    round = QuerySelectField(get_label=lambda round: "%s in %s @ Iteration: %d" %
                             (round.classifier.keyword.name if round.classifier.keyword else 'Export Clf',
                              round.classifier.dataset.name,
                              round.number),
                                allow_blank=False,
                                query_factory=lambda:Round.query.order_by(Round.classifier_id.asc(),Round.number.desc()).all())
    dataset = QuerySelectField(get_label='name',
                                allow_blank=False,
                                query_factory=lambda:Dataset.query.filter_by(is_train=False).order_by(Dataset.id.desc()))

class ClassifierEvaluateForm(Form):
    note = HiddenField()
    first_incorrect = ObjectField(model=Patch, validators = [DataRequired()])
    last_correct = ObjectField(model=Patch, validators = [DataRequired()])
    classifier = ObjectField(model=Classifier, validators = [DataRequired()])
    round = ObjectField(model=Round, validators = [DataRequired()])
    dataset = ObjectField(model=Dataset, validators = [DataRequired()])
