import os
import pandas as pd

from unique_classes import get_classes

dataset_path = 'server/object_detection_classification/multiclass_dataset'

classes_to_transfer = get_classes(dataset_path)
print("Classes to transfer:", classes_to_transfer)

