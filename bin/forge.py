#!/usr/bin/env python
"""
These functions are to create a fake testing dataset for this app.
Example uses the FathomNet-py api to get image urls.
"""
from app import db
from app.models import (
    User,
    PatchSpec,
    FeatureSpec,
    Dataset,
    Keyword,
    Classifier,
    Estimator,
    Example,
    Patch,
    Blob,
)
from setup_database import setup_database_defaults
from dataset_ingestion import dataset_upload_urls
import tasks

import fathomnet.api.images
import fathomnet.models


class ForgeData():

    def __init__(self, **kwargs):
        for arg in kwargs:
            self.dataset_name = kwargs.get('dataset_name')
            self.dataset_url_list = kwargs.get('dataset_url_list')
            self.keyword_name = kwargs.get('keyword_name')
            self.keyword_seed_url = kwargs.get('keyword_seed_url')
            self.keyword_seed_bbox = kwargs.get('keyword_seed_bbox') #[x,y,width,height]


def forge(forgeData):
    try:
        # If the database has not been initialized yet, do so and add default entities
        setup_database_defaults()
    except:
        # otherwise, just add forged data to database
        pass

    # Make a dataset
    dataset_name = forgeData.dataset_name
    list_of_urls = forgeData.dataset_url_list

    # If the dataset exists, i.e. forge was run before, don't make the dataset twice
    if not Dataset.query.filter(Dataset.name == dataset_name).all():
        patchspec_id = PatchSpec.query.filter(PatchSpec.name == "Sparse").first().id
        featurespec_id = (
            FeatureSpec.query.filter(FeatureSpec.name == "TinyImage").first().id
        )
        dataset_upload_urls(
            list_of_urls, dataset_name, featurespec_id, patchspec_id, val_percent=0.1
        )
    dataset = Dataset.query.filter(Dataset.name == dataset_name).first()

    # Get default user
    owner = User.query.get(1)

    # Make a keyword
    name = forgeData.keyword_name
    url = forgeData.keyword_seed_url
    x, y, width, height = forgeData.keyword_seed_bbox

    keyword = Keyword(name=name, owner=owner)
    db.session.add(keyword)

    # Add blob, patch, example to keyword
    blob = Blob(location=url)
    db.session.add(blob)

    patch = Patch(blob=blob, x=int(x), y=int(y),
                  width=int(width), height=int(height))
    db.session.add(patch)

    seed = Example(value=True, patch=patch, keyword=keyword)
    db.session.add(seed)

    # Make a classifier
    # Get default Estimator
    estimator = Estimator.query.get(1)
    classifier = Classifier(
        owner=owner, dataset=dataset, keyword=keyword, estimator=estimator
    )
    db.session.add(classifier)

    # Commit keyword items and new classifier to DB so they have primary keys
    db.session.commit()

    tasks.classifier(classifier.id)


# Example forged data, from the FathomNet API

def get_fathoment_forge():

    # FathomNet Keyword Example
    keyword_name = "Aurelia aurita"
    query = fathomnet.models.GeoImageConstraints(
        concept=keyword_name,
    )

    concept_images = fathomnet.api.images.find(query)
    dataset_url_list = [img.url for img in concept_images]
    dataset_name = "Moon Jellies"

    # Find a starting example (a seed) from a specific gps range
    query = fathomnet.models.GeoImageConstraints(
        concept=keyword_name,
        maxLatitude=37.0538,
        minLatitude=36.4458,
        maxLongitude=-121.7805,
        minLongitude=-122.5073,
        limit=1,
    )
    fn_img = fathomnet.api.images.find(query)
    keyword_seed_url = fn_img[0].url
    x = fn_img[0].boundingBoxes[0].x
    y = fn_img[0].boundingBoxes[0].y
    width = fn_img[0].boundingBoxes[0].width
    height = fn_img[0].boundingBoxes[0].height
    keyword_seed_bbox = [x, y, width, height]

    return ForgeData(dataset_name=dataset_name,
                     dataset_url_list=dataset_url_list,
                     keyword_name=keyword_name,
                     keyword_seed_url=keyword_seed_url,
                     keyword_seed_bbox=keyword_seed_bbox
                     )


if __name__ == "__main__":
    from config import APPNAME

    print(f"Forging dataset to test {APPNAME}")

    # Customize the data loaded for forging!
    forgeData = get_fathoment_forge()
    forge(forgeData)
