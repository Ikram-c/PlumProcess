import lensfunpy
import cv2
from fractions import Fraction
import imageio.v3 as iio
import os

def _conv_fraction(fraction_string):
    """Convert EXIF-style fraction string to float (e.g., '28/10' -> 2.8)."""
    negative = 1
    if fraction_string.startswith('-'):
        fraction_string = fraction_string[1:]
        negative = -1
    try:
        result = negative * float(sum(Fraction(s) for s in fraction_string.split()))
        return result
    except Exception:
        return None

def lens_fix(im_path, metadata_dict, output_dir):
    """
    Corrects lens distortion in an image using lensfunpy and EXIF metadata.

    Args:
        im_path (str): Path to the input image.
        metadata_dict (dict): Metadata for the image.
        output_dir (str): Directory to save the corrected image.

    Returns:
        str: Path to the corrected image, or None on failure.
    """
    print("begun lens_fix")
    db = lensfunpy.Database()
    make = metadata_dict.get('Image Make', '')
    model = metadata_dict.get('Image Model', '')

    cams = db.find_cameras(make, model)
    if not cams:
        print(f"[lens_fix] Camera '{make} {model}' not found in lensfun database.")
        return None
    cam = cams[0]

    lens_model = metadata_dict.get('EXIF LensModel')
    lenses = db.find_lenses(cam, lens_model) if lens_model else db.find_lenses(cam)
    if not lenses:
        print(f"[lens_fix] Lens '{lens_model}' not found for camera '{model}' in lensfun database.")
        return None
    lens = lenses[0]

    focal_str = metadata_dict.get('EXIF FocalLength')
    aperture_str = metadata_dict.get('EXIF ApertureValue')
    if not focal_str or not aperture_str:
        print("[lens_fix] FocalLength or ApertureValue missing in metadata.")
        return None
    focal_length = _conv_fraction(focal_str)
    aperture = _conv_fraction(aperture_str)

    if metadata_dict.get('RelativeAltitude'):
        distance = float(metadata_dict['RelativeAltitude'])
    elif metadata_dict.get('EXIF SubjectDistance'):
        distance = _conv_fraction(metadata_dict['EXIF SubjectDistance'])
    else:
        distance = 0

    img = iio.imread(im_path)
    h, w = img.shape[:2]

    mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
    mod.initialize(focal_length, aperture, distance)
    undistort_map = mod.apply_geometry_distortion()
    img_undistorted = cv2.remap(img, undistort_map, None, cv2.INTER_LANCZOS4)

    base = os.path.splitext(os.path.basename(im_path))[0]
    out_path = os.path.join(output_dir, f"undistorted_{base}.jpg")
    iio.imwrite(out_path, img_undistorted)

    print(f"corrected {os.path.basename(im_path)} -> {out_path}")
    return out_path
