import sys

# Global imports when running the modules for testing
if "-m" not in sys.argv:

    # Import the functions so that they are accesible from the main module
    from .labeling.segmentlabeler import SegmentLabeler  # noqa
    from .labeling.volumelabeler import VolumeLabeler  # noqa
