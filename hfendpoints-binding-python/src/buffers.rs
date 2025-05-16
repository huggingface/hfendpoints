use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::{c_int, c_void, CString};
use std::ptr;

/// # Safety
///
/// `view` must be a valid pointer to ffi::Py_buffer, or null
/// `data` must outlive the Python lifetime of `owner` (i.e., data must be owned by owner, or data
/// must be static data)
pub unsafe fn fill_view_from_readonly_data(
    view: *mut ffi::Py_buffer,
    flags: c_int,
    data: &[u8],
    owner: Bound<'_, PyAny>,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("View is null"));
    }

    if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err("Object is not writable"));
    }

    unsafe {
        (*view).obj = owner.into_ptr();

        (*view).buf = data.as_ptr() as *mut c_void;
        (*view).len = data.len() as isize;
        (*view).readonly = 1;
        (*view).itemsize = 1;

        (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
            let msg = CString::new("B")?;
            msg.into_raw()
        } else {
            ptr::null_mut()
        };

        (*view).ndim = 1;
        (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
            &mut (*view).len
        } else {
            ptr::null_mut()
        };

        (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
            &mut (*view).itemsize
        } else {
            ptr::null_mut()
        };

        (*view).suboffsets = ptr::null_mut();
        (*view).internal = ptr::null_mut();
    }
    Ok(())
}
