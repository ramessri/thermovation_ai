"""
upload_to_roboflow.py — upload the merged/ dataset (Boiler GT + pipe
proposals) to Roboflow via the SDK as predictions awaiting review.

Uploading with is_prediction=True lands the pipe masks as unapproved
proposals in an Annotate job named after --batch-name, not as final ground
truth — review them there (accept/reshape/delete/add), then export a new
COCO version and swap it in for Boilers.coco/.

Usage:
  python experiments/upload_to_roboflow.py --api-key YOUR_KEY --project boilers-review
"""

import argparse

import roboflow


def main():
    parser = argparse.ArgumentParser(description="Upload merged dataset to Roboflow for review")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project", default="boilers-review",
                        help="New project id (recommended, avoids SHA-256 dedup/merge "
                             "ambiguity against an existing project) or an existing one")
    parser.add_argument("--dataset-path", default="./merged/")
    parser.add_argument("--project-type", default="instance-segmentation")
    parser.add_argument("--batch-name", default="pipe-proposals")
    args = parser.parse_args()

    rf = roboflow.Roboflow(api_key=args.api_key)
    rf.workspace().upload_dataset(
        args.dataset_path,
        args.project,
        project_type=args.project_type,
        batch_name=args.batch_name,
        is_prediction=True,
    )
    print(f"Uploaded {args.dataset_path} to project '{args.project}', "
          f"batch '{args.batch_name}' — review it under Annotate.")


if __name__ == "__main__":
    main()
