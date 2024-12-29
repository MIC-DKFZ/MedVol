import SimpleITK as sitk
from typing import Dict, Optional, Union, List, Tuple
import numpy as np
from pathlib import Path

# TODO:
# - Enable user to set affine
#   - Reflect changes in affine in all other parameters
# - Write tests
# - Rename into MedImg


class MedVol:
    def __init__(self,
                 array: Union[np.ndarray, str, Path],
                 spacing: Optional[Union[List, Tuple, np.ndarray]] = None,
                 origin: Optional[Union[List, Tuple, np.ndarray]] = None,
                 direction: Optional[Union[List, Tuple, np.ndarray]] = None,
                 header: Optional[Dict] = None,
                 is_seg: Optional[bool] = None,
                 copy: Optional['MedVol'] = None) -> None:
        """
        Initializes the MedVol object with image data and associated metadata.

        Args:
            array (Union[np.ndarray, str]): A 2D, 3D, or 4D numpy array or a file path string to load an image.
            spacing (Optional[Union[List, Tuple, np.ndarray]]): Voxel spacing in each dimension. Defaults to None.
            origin (Optional[Union[List, Tuple, np.ndarray]]): Origin coordinates in physical space. Defaults to None.
            direction (Optional[Union[List, Tuple, np.ndarray]]): Direction cosines for the image axes. Defaults to None.
            header (Optional[Dict]): Metadata associated with the image. Defaults to None.
            is_seg (Optional[bool]): Indicates if the image is a segmentation. Defaults to None.
            copy (Optional['MedVol']): Another MedVol instance to copy fields from. Defaults to None.

        Raises:
            ValueError: If array is not a valid numpy array or string.
            RuntimeError: If incompatible arguments are provided.
        """ 
        # Validate array: Must be a 2D, 3D or 4D array
        if not ((isinstance(array, np.ndarray) and (array.ndim == 2 or array.ndim == 3 or array.ndim == 4)) or isinstance(array, (str, Path))):
            raise ValueError("Array must be a 2D, 3D or 4D numpy array or a filepath string.")
        elif isinstance(array, str) and (spacing is not None or origin is not None or direction is not None or header is not None or is_seg is not None or copy is not None):
            raise RuntimeError("Spacing, origin, direction, header, is_seg or copy cannot be set if array is a string to load an image.")
        
        if isinstance(array, str) or isinstance(array, Path):
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
            self.is_seg = None
            self._has_is_seg = False
        
        # If copy is set, copy fields from the other Nifti instance
        if copy is not None:
            self._copy_fields_from(copy)

    @property
    def affine(self) -> np.ndarray:
        """
        Computes the affine transformation matrix for the image.

        Returns:
            np.ndarray: The affine matrix representing the translation, scaling, and rotation of the image.
        """        
        affine = np.eye(self.ndims+1)
        affine[:self.ndims, :self.ndims] = self.direction @ np.diag(self.spacing)
        affine[:self.ndims, self.ndims] = self.origin
        return affine
    
    @property
    def translation(self):
        """
        Extracts the translation vector from the affine matrix.

        Returns:
            np.ndarray: The translation vector from the affine matrix.
        """
        return self.affine[:-1, -1]

    @property
    def scale(self):
        """
        Extracts the scaling factors from the affine matrix.

        Returns:
            np.ndarray: The scaling factors for each axis from the affine matrix.
        """
        scales = np.linalg.norm(self.affine[:-1, :-1], axis=0)
        return scales

    @property
    def rotation(self):
        """
        Extracts the rotation matrix from the affine matrix.

        Returns:
            np.ndarray: The rotation matrix from the affine matrix.
        """
        rotation_matrix = self.affine[:-1, :-1] / self.scale
        return rotation_matrix

    @property
    def shear(self):
        """
        Computes the shear matrix from the affine matrix.

        Returns:
            np.ndarray: The shear matrix, representing axis-alignment adjustments.
        """
        scales = self.scale
        rotation_matrix = self.rotation
        shearing_matrix = np.dot(rotation_matrix.T, self.affine[:-1, :-1]) / scales[:, None]
        return shearing_matrix
    
    @property
    def ndims(self) -> int:
        """
        Returns the number of dimensions of the image.

        Returns:
            int: The number of dimensions of the image (2D, 3D, or 4D).
        """        
        return len(self.array.shape)

    def _copy_fields_from(self, other: 'MedVol'):
        """
        Copies fields from another MedVol instance if they are not already set.

        Args:
            other (MedVol): The MedVol instance to copy fields from.
        """
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
        """
        Loads image data and metadata from a file.

        Args:
            filepath (str or Path): The path to the file to load.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Optional[bool]]:
            The image array, spacing, origin, direction, header, and is_seg values.

        Raises:
            RuntimeError: If the dimensionality of the image and metadata do not match.
        """
        image_sitk = sitk.ReadImage(str(filepath))
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
        if "ITK_FileNotes" in header and header["ITK_FileNotes"] == "medvol_seg":
            is_seg = True
        elif "ITK_FileNotes" in header and header["ITK_FileNotes"] == "medvol_img":
            is_seg = False
        
        return array, spacing, origin, direction, header, is_seg

    def save(self, filepath):
        """
        Saves the current image and its metadata to a file.

        Args:
            filepath (str or Path): The path where the file will be saved.

        Raises:
            RuntimeError: If saving a 4D image is attempted.
        """
        if self.ndims == 4:
            raise RuntimeError("Saving a 4D image is currently not supported.")
        image_sitk = sitk.GetImageFromArray(self.array)
        if self.spacing is not None:
            image_sitk.SetSpacing(self.spacing.tolist()[::-1])
        if self.origin is not None:
            image_sitk.SetOrigin(self.origin.tolist()[::-1])
        if self.direction is not None:
            image_sitk.SetDirection(self.direction.flatten().tolist()[::-1])
        if self.is_seg is not None:
            if self.is_seg:
                self.header["ITK_FileNotes"] = "medvol_seg"
            elif not self.is_seg:
                self.header["ITK_FileNotes"] = "medvol_img"
        if self.header is not None:
            for key, value in self.header.items():
                image_sitk.SetMetaData(key, value)
        sitk.WriteImage(image_sitk, str(filepath), useCompression=True)