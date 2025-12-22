import os
import re
import exifread
import numpy as np
from bs4 import BeautifulSoup
from iptcinfo3 import IPTCInfo
from typing import Union


class Metadata_Extractor:
    """
    A class to extract and process metadata from image files.
    
    This class handles extraction of XMP, EXIF, and IPTC metadata from images,
    with capabilities to clean and combine the metadata into a single dictionary.
    
    Attributes:
        img (bytes or file-like): The raw image data or file-like object of the image.
        image_path (str): The file path (or name) of the image.
    """
    
    def __init__(self, image_path):
        """
        Args:
            img (file-like): An open file-like object or bytes of the image.
            image_path (str): Path to the image file. Used to derive basename for saving metadata.
        """
        
        self.image_path = image_path



    def run_metadata_extractor(self) -> None:
        """
        Main method to extract and combine all metadata from the image.
        Serializes the resulting dictionary into a .npz file in a 'tmp' folder.
        Does not return anything.
        """
        with open(self.image_path, 'rb') as img_file:
            self.img = img_file
            exif_dict = self.read_exif_metadata()
            xmp_dict = self.xmp_metadata_cleaner()
            iptc_dict = self.read_iptc_metadata()

            exif_dict = {str(key): str(value) for key, value in exif_dict.items()}
            
            if isinstance(xmp_dict, dict):
                metadata_dict = {**exif_dict, **xmp_dict, **iptc_dict}
            else:
                metadata_dict = {**exif_dict, **iptc_dict, "XMP_error": xmp_dict}

            metadata_dict = {str(k): str(v) for k, v in metadata_dict.items()}

            
            
        return metadata_dict

    def read_exif_metadata(self) -> dict:
        """
        Extract EXIF metadata from the image.

        Returns:
            dict: Dictionary containing EXIF metadata.
        """
        # Reset file pointer if necessary (helpful if 'img' is an open file)
        if hasattr(self.img, 'seek'):
            self.img.seek(0)
        return exifread.process_file(self.img, details=False)

    def read_iptc_metadata(self) -> dict:
        """
        Extract IPTC metadata from the image.
        
        Returns:
            dict: Dictionary containing IPTC metadata.
        """
        # Reset file pointer if necessary
        if hasattr(self.img, 'seek'):
            self.img.seek(0)
        iptc_info = IPTCInfo(self.img)
        # Convert IPTCInfo object to dict; skip empty or None values
        return {k: v for k, v in iptc_info._data.items() if v not in (None, b'', '')}

    def read_xmp_metadata(self) -> Union[dict, str]:
        """
        Extract XMP metadata from the image.
        
        Returns:
            Union[dict, str]: 
                - Dictionary containing XMP metadata if found.
                - Error message string if not found.
        """
        # Reset file pointer if necessary
        if hasattr(self.img, 'seek'):
            self.img.seek(0)

        # Read raw bytes from the image (in case 'img' is a file object)
        if hasattr(self.img, 'read'):
            data = self.img.read()
        else:
            data = self.img  # if already bytes

        xmp_start = data.find(b'<x:xmpmeta')
        xmp_end = data.find(b'</x:xmpmeta>')

        if -1 in (xmp_start, xmp_end):
            return "No XMP metadata found in the image."

        xmp_end += len(b'</x:xmpmeta>')
        xmp_string = data[xmp_start:xmp_end].decode('utf-8', errors='ignore')
        xmp_as_xml = BeautifulSoup(xmp_string, "xml")
        rdf_description = xmp_as_xml.find('rdf:Description')

        if not rdf_description:
            return "No rdf:Description found in the XMP metadata."

        return dict(rdf_description.attrs)

    @staticmethod
    def _convert_value(value: Union[str, int, float]) -> Union[float, str]:
        """
        Convert string values to float where possible.
        
        Args:
            value: The value to convert.
            
        Returns:
            Union[float, str]: 
                - Converted float if possible, otherwise original string.
        """
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and re.match(r'^[+-]?\d*\.?\d+$', value.strip()):
            return float(value)
        return str(value)

    def xmp_metadata_cleaner(self) -> Union[dict, str]:
        """
        Clean XMP metadata by trimming namespace prefixes and converting numeric values.
        
        Returns:
            Union[dict, str]: 
                - Cleaned dictionary with simplified keys and converted values,
                - Or error message if no valid XMP metadata found.
        """
        xmp_data = self.read_xmp_metadata()
        
        if isinstance(xmp_data, str):
            # Error case: XMP metadata not found
            return xmp_data
            
        # Otherwise, xmp_data is a dictionary
        return {
            key.split(':')[-1]: self._convert_value(value) 
            for key, value in xmp_data.items()
        }