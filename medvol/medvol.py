import SimpleITK as sitk
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple
import numpy as np


@dataclass
class MedVol:
    array: Union[np.ndarray, str]
    spacing: Optional[Union[List, Tuple, np.ndarray]] = None
    origin: Optional[Union[List, Tuple, np.ndarray]] = None
    direction: Optional[Union[List, Tuple, np.ndarray]] = None
    header: Optional[Dict] = None
    copy: Optional['MedVol'] = field(default=None, repr=False)

    def __post_init__(self):
        # Validate array: Must be a 3D array
        if not ((isinstance(self.array, np.ndarray) and self.array.ndim == 3) or isinstance(self.array, str)):
            raise ValueError("Array must be a 3D numpy array or a filepath string")
        
        if isinstance(self.array, str):
            self._load(self.array)

        # Validate spacing: Must be None or an array-like with shape (3,)
        if self.spacing is not None:
            if not isinstance(self.spacing, (List, Tuple, np.ndarray)):
                raise ValueError("Spacing must be either None, List, Tuple or np.ndarray")
            self.spacing = np.array(self.spacing).astype(float)
            if not (self.spacing.shape == (3,) and np.issubdtype(self.spacing.dtype, np.floating)):
                raise ValueError("The dtype of spacing must be three floats")

        # Validate origin: Must be None or an array-like with shape (3,)
        if self.origin is not None:
            if not isinstance(self.origin, (List, Tuple, np.ndarray)):
                raise ValueError("Origin must be either None, List, Tuple or np.ndarray")
            self.origin = np.array(self.origin).astype(float)
            if not (self.origin.shape == (3,) and np.issubdtype(self.origin.dtype, np.floating)):
                raise ValueError("The dtype of origin must be three floats")

        # Validate direction: Must be None or an array-like with shape (3, 3)
        if self.direction is not None:
            if not isinstance(self.direction, (List, Tuple, np.ndarray)):
                raise ValueError("Direction must be either None, List, Tuple or np.ndarray")
            self.direction = np.array(self.direction).astype(float)
            if not (self.direction.shape == (3, 3) and np.issubdtype(self.direction.dtype, np.floating)):
                raise ValueError("The dtype of direction must be three floats")

        # Validate header: Must be None or a dictionary
        if self.header is not None and not isinstance(self.header, dict):
            raise ValueError("header must be None or a dictionary")
        
        # If copy is set, copy fields from the other Nifti instance
        if self.copy is not None:
            self._copy_fields_from(self.copy)

    @property
    def affine(self) -> np.ndarray:
        if self.spacing is None or self.origin is None or self.direction is None:
            raise ValueError("spacing, origin, and direction must all be set to compute the affine.")
        
        affine = np.eye(4)
        affine[:3, :3] = self.direction @ np.diag(self.spacing)
        affine[:3, 3] = self.origin
        return affine

    def _copy_fields_from(self, other: 'MedVol'):
        if self.spacing is None:
            self.spacing = other.spacing
        if self.origin is None:
            self.origin = other.origin
        if self.direction is None:
            self.direction = other.direction
        if self.header is None:
            self.header = other.header

    def _load(self, filepath):
        image_sitk = sitk.ReadImage(filepath)
        self.array = sitk.GetArrayFromImage(image_sitk)
        self.spacing = np.array(image_sitk.GetSpacing()[::-1])
        self.origin = np.array(image_sitk.GetOrigin()[::-1])
        self.direction = np.array(image_sitk.GetDirection()[::-1]).reshape(3, 3)
        self.header = {key: image_sitk.GetMetaData(key) for key in image_sitk.GetMetaDataKeys()}

    def save(self, filepath):
        image_sitk = sitk.GetImageFromArray(self.array)
        image_sitk.SetSpacing(self.spacing.tolist()[::-1])
        image_sitk.SetOrigin(self.origin.tolist()[::-1])
        image_sitk.SetDirection(self.direction.flatten().tolist()[::-1])
        for key, value in self.header.items():
            image_sitk.SetMetaData(key, value) 
        sitk.WriteImage(image_sitk, filepath)