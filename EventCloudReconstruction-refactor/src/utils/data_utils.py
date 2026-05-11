import numpy as np

def parse_header(fhandle):
    """
    Parses the header of a .npy file
    Args:
        - f file handle to a .npy file
    return :
        - int position of the file cursor after the header
        - int type of event
        - int size of event in bytes
        - size (height, width) tuple of int or (None, None)
    """
    version = np.lib.format.read_magic(fhandle)
    shape, fortran, dtype = np.lib.format._read_array_header(fhandle, version)
    assert not fortran, "Fortran order arrays not supported"
    # Get the number of elements in one 'row' by taking
    # a product over all other dimensions.
    if len(shape) == 0:
        count = 1
    else:
        count = np.multiply.reduce(shape, dtype=np.int64)
    ev_size = dtype.itemsize
    assert ev_size != 0
    start = fhandle.tell()
    # turn numpy.dtype into an iterable list
    ev_type = [(x, str(dtype.fields[x][0])) for x in dtype.names]
    # filter name to have only t and not ts
    ev_type = [(name if name != "ts" else "t", desc) for name, desc in ev_type]
    ev_type = [(name if name != "confidence" else "class_confidence", desc) for name, desc in ev_type]
    size = (None, None)
    size = (None, None)

    return start, ev_type, ev_size, size


def read_npy_chunk(filename, start_row, num_rows):
    """
    Reads a partial array (contiguous chunk along the first
    axis) from an NPY file.
    Parameters
    ----------
    filename : str
        Name/path of the file from which to read.
    start_row : int
        The first row of the chunk you wish to read. Must be
        less than the number of rows (elements along the first
        axis) in the file.
    num_rows : int
        The number of rows you wish to read. The total of
        `start_row + num_rows` must be less than the number of
        rows (elements along the first axis) in the file.
    Returns
    -------
    out : ndarray
        Array with `out.shape[0] == num_rows`, equivalent to
        `arr[start_row:start_row + num_rows]` if `arr` were
        the entire array (note that the entire array is never
        loaded into memory by this function).
    """
    assert start_row >= 0 and num_rows > 0
    with open(filename, 'rb') as fhandle:
        major, minor = np.lib.format.read_magic(fhandle)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fhandle)
        assert not fortran, "Fortran order arrays not supported"
        # Make sure the offsets aren't invalid.
        assert start_row < shape[0], (
            'start_row is beyond end of file'
        )
        assert start_row + num_rows <= shape[0], (
            'start_row + num_rows > shape[0]'
        )
        # Get the number of elements in one 'row' by taking
        # a product over all other dimensions.
        row_size = np.prod(shape[1:])
        start_byte = start_row * row_size * dtype.itemsize
        fhandle.seek(start_byte, 1)
        n_items = row_size * num_rows
        flat = np.fromfile(fhandle, count=n_items, dtype=dtype)
        return flat.reshape((-1,) + shape[1:])


def get_len_without_loading(file_path):
    """
    This function will return the length of the idx-th file without loading it.
    """
    with open(file_path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
        return shape[0]