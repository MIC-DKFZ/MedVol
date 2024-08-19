import SimpleITK as sitk
from typing import Dict, Optional, Union, List, Tuple
import numpy as np

# TODO:
# - Enable user to set affine
#   - Reflect changes in affine in all other parameters
# - Add affine_3d function?
# - Create docstrings and docs
# - Write tests
# - Rename into MedImg
# - Fix is_seg
# - Fix napari-nifti loading of 2D and 4D images


class MedVol:
    def __init__(self,
                 array: Union[np.ndarray, str],
                 spacing: Optional[Union[List, Tuple, np.ndarray]] = None,
                 origin: Optional[Union[List, Tuple, np.ndarray]] = None,
                 direction: Optional[Union[List, Tuple, np.ndarray]] = None,
                 header: Optional[Dict] = None,
                 is_seg: Optional[bool] = None,
                 copy: Optional['MedVol'] = None) -> None:        
        # Validate array: Must be a 2D, 3D or 4D array
        if not ((isinstance(array, np.ndarray) and (array.ndim == 2 or array.ndim == 3 or array.ndim == 4)) or isinstance(array, str)):
            raise ValueError("Array must be a 2D, 3D or 4D numpy array or a filepath string.")
        elif isinstance(array, str) and (spacing is not None or origin is not None or direction is not None or header is not None or is_seg is not None or copy is not None):
            raise RuntimeError("Spacing, origin, direction, header, is_seg or copy cannot be set if array is a string to load an image.")
        
        if isinstance(array, str):
            array, spacing, origin, direction, header, is_seg = self._load(array)

        self.array = array

        # Validate spacing: Must be None or an array-like with shape (2,), (3,) or (4,) depending on the array dimensionality
        if spacing is not None:
            if not isinstance(spacing, (List, Tuple, np.ndarray)):
                raise ValueError("Spacing must be either None, List, Tuple or np.ndarray.")
            spacing = np.array(spacing).astype(float)
            if not (spacing.shape == (self.ndims,) and np.issubdtype(spacing.dtype, np.floating)):
                raise ValueError(f"The dtype of spacing must be {self.ndims} floats.")
            self.spacing = spacing
            self._has_spacing = True
        else:
            self.spacing = np.full((self.ndims,), fill_value=1.0)
            self._has_spacing = False

        # Validate origin: Must be None or an array-like with shape (2,), (3,) or (4,) depending on the array dimensionality
        if origin is not None:
            if not isinstance(origin, (List, Tuple, np.ndarray)):
                raise ValueError("Origin must be either None, List, Tuple or np.ndarray.")
            origin = np.array(origin).astype(float)
            if not (origin.shape == (self.ndims,) and np.issubdtype(origin.dtype, np.floating)):
                raise ValueError(f"The dtype of origin must be {self.ndims} floats.")
            self.origin = origin
            self._has_origin = True
        else:
            self.origin = np.zeros((self.ndims,))
            self._has_origin = False

        # Validate direction: Must be None or an array-like with shape (2, 2), (3, 3) or (4, 4) depending on the array dimensionality
        if direction is not None:
            if not isinstance(direction, (List, Tuple, np.ndarray)):
                raise ValueError("Direction must be either None, List, Tuple or np.ndarray.")
            direction = np.array(direction).astype(float)
            if not (direction.shape == (self.ndims, self.ndims) and np.issubdtype(direction.dtype, np.floating)):
                raise ValueError(f"The dtype of direction must be an array os shape ({self.ndims}, {self.ndims}) floats.")
            self.direction = direction
            self._has_direction = True
        else:
            self.direction = np.eye(self.ndims)
            self._has_direction = False

        # Validate header: Must be None or a dictionary
        if header is not None:
            if not isinstance(header, dict):
                raise ValueError("Header must be None or a dictionary.")
            else:
                self.header = header
                self._has_header = True
        else:
            self.header = {}
            self._has_header = False
        
        # Validate is_seg: Must be None or a boolean
        if is_seg is not None:
            if not isinstance(is_seg, bool):
                raise ValueError("is_seg must be None or a boolean.")
            else:
                self.is_seg = is_seg
                self._has_is_seg = True
        else:
            self.is_seg = False
            self._has_is_seg = False
        
        # If copy is set, copy fields from the other Nifti instance
        if copy is not None:
            self._copy_fields_from(copy)

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
        if not self._has_spacing:
            self.spacing = other.spacing
        if not self._has_origin:
            self.origin = other.origin
        if not self._has_direction:
            self.direction = other.direction
        if not self._has_header:
            self.header = other.header
        if not self._has_is_seg:
            self.is_seg = other.is_seg

    def _load(self, filepath):
        image_sitk = sitk.ReadImage(filepath)
        array = sitk.GetArrayFromImage(image_sitk)
        ndims = len(array.shape)
        metadata_ndims = len(image_sitk.GetSpacing())

        if ndims == 4 and np.argmin(array.shape) != 0:
            raise RuntimeError("MedVol expect 4D images to be channel-first.")
        if ndims != metadata_ndims: 
            raise RuntimeError("Cannot interpret image metadata. Something is wrong with the dimensionality.")
        
        spacing = np.array(image_sitk.GetSpacing()[::-1])
        origin = np.array(image_sitk.GetOrigin()[::-1])
        direction = np.array(image_sitk.GetDirection()[::-1]).reshape(ndims, ndims)
        header = {key: image_sitk.GetMetaData(key) for key in image_sitk.GetMetaDataKeys()}          
        is_seg = None
        if "intent_name" in header and header["intent_name"] == "medvol_seg":
            is_seg = True
        elif "intent_name" in header and header["intent_name"] == "medvol_img":
            is_seg = False
        
        return array, spacing, origin, direction, header, is_seg

    def save(self, filepath):
        if self.ndims == 4:
            raise RuntimeError("Saving a 4D image is currently not supported.")
        image_sitk = sitk.GetImageFromArray(self.array)
        # SimpleITK cannot store 4D metadata, only 2D and 3D metadata
        if self.spacing is not None:
            image_sitk.SetSpacing(self.spacing.tolist()[::-1])
        if self.origin is not None:
            image_sitk.SetOrigin(self.origin.tolist()[::-1])
        if self.direction is not None:
            image_sitk.SetDirection(self.direction.flatten().tolist()[::-1])
        if self.is_seg is not None:
            if self.is_seg:
                self.header["intent_name"] = "medvol_seg"
            elif not self.is_seg or self.is_seg is None:
                self.header["intent_name"] = "medvol_img"
        if self.header is not None:
            for key, value in self.header.items():
                image_sitk.SetMetaData(key, value)
        sitk.WriteImage(image_sitk, filepath, useCompression=True)