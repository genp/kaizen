#!/usr/bin/env python
import argparse
import os
import random

from app import db, manage
from app.models import User, Dataset, Blob, PatchSpec, FeatureSpec
import tasks
from utils.extract_frames import extract_frames
from utils.add_blob import add_blob, add_blobs_batch

_acceptable = [".jpg", ".jpeg", ".png"]
_video_acceptable = [".mp4", ".mov", ".avi"]


def setup_dataset(name, featurespec_id, patchspec_id, val_percent=0.1):
    """
    Make a new empty Dataset.
    """
    owner = User.query.get(1)  # this is the default database user

    # Check if there is a dataset by the input name, else create new Dataset
    dset = Dataset.query.filter(Dataset.name == name).first()
    if not dset:
        dset = Dataset(name=name, owner=owner)
        db.session.add(dset)
        db.session.commit()
        if val_percent:
            vdset = Dataset(name=name + "_val", owner=owner)
            vdset.train_dset_id = dset.id
            vdset.is_train = False
            db.session.add(vdset)
            db.session.commit()
    else:
        vdset = dset.val_dset.first()

    if featurespec_id:
        dset.featurespecs.append(FeatureSpec.query.get(featurespec_id))
        vdset.featurespecs.append(FeatureSpec.query.get(featurespec_id))
    if patchspec_id:
        dset.patchspecs.append(PatchSpec.query.get(patchspec_id))
        vdset.patchspecs.append(PatchSpec.query.get(patchspec_id))

    return dset, vdset


def dataset_upload_urls(
        list_of_urls, name, featurespec_id, patchspec_id, val_percent=0.1,
        use_task_queue=False
):
    """
    Upload all images and or videos from the input list of urls into a new
    Dataset.
    """
    dset, vdset = setup_dataset(name, featurespec_id, patchspec_id, val_percent)
    print(f'Attempint to adding {len(list_of_urls)} images{dset}')
    dset_urls = []
    vdset_urls = []
    for url in list_of_urls:
        # check url is to a valid file type
        _, ext = os.path.splitext(url)
        print(url)
        if ext.lower() in _acceptable:
            print(f"adding image url to dataset: {url}")
            if random.random() > val_percent:
                dset_urls.append(url)
            else:
                vdset_urls.append(url)

        # TODO: if video extract frames first and add them all as blobs
        if ext.lower() in _video_acceptable:
            print("Video url add not currently supported")

        if ext.lower() not in _acceptable and ext.lower() not in _video_acceptable:
            print(f"This url points to an unsupported file type: {url}")

    add_blobs_batch(dset_urls, dset)
    add_blobs_batch(vdset_urls, vdset)
    db.session.commit()

    if not use_task_queue:
        # Execute without Celery so all patches and features will be available at once.
        # Suitable for smaller datasets.
        tasks.dataset(dset.id)
        tasks.dataset(vdset.id)
    else:
        # Execute with Celery for faster processing.
        tasks.dataset_distributed.delay(dset.id)
        tasks.dataset_distributed.delay(vdset.id)

    print(
        f"Added {len(dset.blobs)} images to dataset {dset.name}. "
        + "Extracted patches and features."
    )
    return dset


def dataset_upload_dir(
    dirname, name, featurespec_id, patchspec_id, cache_dir, val_percent=0.1
):
    """
    Upload all images and or videos from the input dirname into a new
    Dataset.
    """

    dset, vdset = setup_dataset(name, featurespec_id, patchspec_id, val_percent)

    for root, directories, filenames in os.walk(dirname):
        for filename in filenames:
            fname, ext = os.path.splitext(filename)
            if "__MACOSX" in filename:
                continue
            item = os.path.join(root, filename)
            print("item: " + item)
            if ext.lower() in _acceptable:
                if random.random() > val_percent:
                    add_blob(item, dset)
                else:
                    add_blob(item, vdset)
            # if video extract frames first and add them all as blobs
            if ext.lower() in _video_acceptable:

                cur_cache_dir = os.path.join(
                    manage.get_hash_dir(cache_dir, os.path.basename(item)), fname
                )
                if (
                    Blob.query.filter(Blob.location.like(cur_cache_dir + "%")).first()
                    is not None
                ):
                    continue
                frames = extract_frames(item, cur_cache_dir)
                print(f"Extracted {len(frames)} frames!")

                # Whole vid goes into either val or train
                if random.random() > val_percent:
                    for fr in frames:
                        add_blob(fr, dset)
                else:
                    for fr in frames:
                        add_blob(fr, vdset)

    db.session.commit()

    # Execute without Celery so all patches and features will be available at once.
    tasks.dataset(dset.id)
    tasks.dataset(vdset.id)

    return dset


if __name__ == "__main__":
    """
    Example of how to call:
    >./datasets_from_dir.py -d ~/AVA_dataset/images
    -n ava_aesthetics_places_redux -f 3 -c ~/kaizen/app/static/datasets
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        help="""the name of the image or video directory you want to
                       add as a Dataset to the database. """,
    )
    parser.add_argument(
        "-n", help="""the name of the new Dataset or Dataset to add to. """
    )
    parser.add_argument(
        "-f",
        type=int,
        help="""the id of the feature spec to associate with this dataset """,
    )
    parser.add_argument(
        "-p",
        type=int,
        help="""the id of the patch spec to associate with this dataset """,
    )
    parser.add_argument(
        "-c", help="""the name of the dir to cache patches and frames to """
    )
    parser.add_argument(
        "-v",
        type=float,
        default=0.1,
        help="""percent of dataset to set aside for validation (0-1.0, float)""",
    )
    args = parser.parse_args()

    if not os.path.exists(args.c):
        print(f"Mkdir for {args.c} or add appropriate symlink")
    else:
        dest = dataset_upload_dir(args.d, args.n, args.f, args.p, args.c, args.v)
