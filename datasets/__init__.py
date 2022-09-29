from cmath import phase
from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .dog_dataset import DogDataset
from .cartype_dataset import CarTypeDataset


def get_trainval_datasets(tag, resize):
    # resize 照片大小
    if tag == 'aircraft':
        return AircraftDataset(phase='train', resize=resize), AircraftDataset(phase='val', resize=resize)
    elif tag == 'bird':
        return BirdDataset(phase='train', resize=resize), BirdDataset(phase='val', resize=resize)
    elif tag == 'car':
        return CarDataset(phase='train', resize=resize), CarDataset(phase='val', resize=resize)
    elif tag == 'dog':
        return DogDataset(phase='train', resize=resize), DogDataset(phase='val', resize=resize)
    elif tag == 'cartype':
        return CarTypeDataset(phase='train', resize=resize), CarTypeDataset(phase='val', resize=resize)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))