#!/usr/bin/env python
from app.models import Blob
from app import db

def add_blob(item, dset):
    """Add a blob to a dataset
    Params:
        item (str): path to an image file
        dset (Dataset): Dataset ORM object from app/models.py
    """
    # If this Blob has already been added to this dataset, skip adding it
    blob = Blob.query.filter(Blob.location == item).first()
    if blob is not None:
        if db.session.execute(f'select * from dataset_x_blob where dataset_id = {dset.id} and blob_id = {blob.id}').fetchall() != []:
            return False

    blob = Blob(item)
    db.session.add(blob)
    db.session.commit()
    dset.blobs.append(blob)
    return True
