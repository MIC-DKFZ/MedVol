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
        # Validate array: Must be a 2D, 3D or 4D array
        if not ((isinstance(self.array, np.ndarray) and (self.array.ndim == 2 or self.array.ndim == 3 or self.array.ndim == 4)) or isinstance(self.array, str)):
            raise ValueError("Array must be a 2D, 3D or 4D numpy array or a filepath string.")
        
        if isinstance(self.array, str):
            self._load(self.array)

        # Validate spacing: Must be None or an array-like with shape (2,), (3,) or (4,) depending on the array dimensionality
        if self.spacing is not None:
            if not isinstance(self.spacing, (List, Tuple, np.ndarray)):
                raise ValueError("Spacing must be either None, List, Tuple or np.ndarray.")
            self.spacing = np.array(self.spacing).astype(float)
            if not (self.spacing.shape == (self.ndims,) and np.issubdtype(self.spacing.dtype, np.floating)):
                raise ValueError(f"The dtype of spacing must be {self.ndims} floats.")

        # Validate origin: Must be None or an array-like with shape (2,), (3,) or (4,) depending on the array dimensionality
        if self.origin is not None:
            if not isinstance(self.origin, (List, Tuple, np.ndarray)):
                raise ValueError("Origin must be either None, List, Tuple or np.ndarray.")
            self.origin = np.array(self.origin).astype(float)
            if not (self.origin.shape == (self.ndims,) and np.issubdtype(self.origin.dtype, np.floating)):
                raise ValueError(f"The dtype of origin must be {self.ndims} floats.")

        # Validate direction: Must be None or an array-like with shape (2, 2), (3, 3) or (4, 4) depending on the array dimensionality
        if self.direction is not None:
            if not isinstance(self.direction, (List, Tuple, np.ndarray)):
                raise ValueError("Direction must be either None, List, Tuple or np.ndarray.")
            self.direction = np.array(self.direction).astype(float)
            if not (self.direction.shape == (self.ndims, self.ndims) and np.issubdtype(self.direction.dtype, np.floating)):
                raise ValueError(f"The dtype of direction must be an array os shape ({self.ndims}, {self.ndims}) floats.")

        # Validate header: Must be None or a dictionary
        if self.header is not None and not isinstance(self.header, dict):
            raise ValueError("Header must be None or a dictionary.")
        
        # If copy is set, copy fields from the other Nifti instance
        if self.copy is not None:
            self._copy_fields_from(self.copy)

    @property
    def affine(self) -> np.ndarray:
        if self.spacing is None or self.origin is None or self.direction is None:
            raise ValueError("Spacing, origin, and direction must all be set to compute the affine.")
        
        affine = np.eye(self.ndims+1)
        affine[:self.ndims, :self.ndims] = self.direction @ np.diag(self.spacing)
        affine[:self.ndims, self.ndims] = self.origin
        return affine
    
    @property
    def ndims(self) -> int:
        if self.array is None:
            raise ValueError("Array be set to compute the number of dimensions.")
        
        ndims = len(self.array.shape)
        return ndims

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
        metadata_ndims = len(image_sitk.GetSpacing())        
        self.array = sitk.GetArrayFromImage(image_sitk)
        self.spacing = np.array(image_sitk.GetSpacing()[::-1])
        self.origin = np.array(image_sitk.GetOrigin()[::-1])
        self.direction = np.array(image_sitk.GetDirection()[::-1])
        self.header = {key: image_sitk.GetMetaData(key) for key in image_sitk.GetMetaDataKeys()}  

        if self.ndims == metadata_ndims:   
            self.direction = self.direction.reshape(self.ndims, self.ndims)
        elif self.ndims == 4 and metadata_ndims == 3: # Some 4D Nifti images might have 3D metadata as SimpleITK can only store 2D and 3D metadata
            # Expand metadata from 3D to 4D
            spacing = np.zeros((self.ndims,))
            spacing[:self.ndims-1] = self.spacing
            spacing[self.ndims-1] = 1
            self.spacing = spacing

            origin = np.zeros((self.ndims,))
            origin[:self.ndims-1] = self.origin
            origin[self.ndims-1] = 0
            self.origin = origin

            direction = np.zeros((self.ndims, self.ndims))
            direction[:self.ndims-1, :self.ndims-1] = self.direction.reshape(self.ndims-1, self.ndims-1)
            direction[self.ndims-1, self.ndims-1] = 1
            self.direction = direction
        else:
            raise RuntimeError("Cannot interpret image metadata. Something is wrong with the dimensionality.")
    def save(self, filepath):
        image_sitk = sitk.GetImageFromArray(self.array)
        # SimpleITK cannot store 4D metadata, only 2D and 3D metadata
        if self.spacing is not None:
            image_sitk.SetSpacing(self.spacing[:3].tolist()[::-1])
        if self.origin is not None:
            image_sitk.SetOrigin(self.origin[:3].tolist()[::-1])
        if self.direction is not None:
            image_sitk.SetDirection(self.direction[:3, :3].flatten().tolist()[::-1])
        if self.header is not None:
            for key, value in self.header.items():
                image_sitk.SetMetaData(key, value)
        sitk.WriteImage(image_sitk, filepath, useCompression=True)