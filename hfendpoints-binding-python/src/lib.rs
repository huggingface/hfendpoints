// Heavily inspired from https://github.com/biomancy/biobit/blob/main/modules/core/py/src/bindings/utils/importable_py_module.rs

use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyNone};
use pyo3::{ffi, PyClass};
use std::ffi::CString;

pub struct ImportablePyModuleBuilder<'py> {
    inner: Bound<'py, PyModule>,
}

impl<'py> ImportablePyModuleBuilder<'py> {
    /// Creates a new `ImportablePyModuleBuilder` with the given module name.
    /// The module is registered with the import system (via PyImport_AddModule) and
    /// is recognizable by the Python import machinery.
    ///
    /// Note that the name should be a fully qualified name of the module (for example, "foo.bar.spam").
    pub fn new(py: Python<'py>, name: &str) -> PyResult<Self> {
        let module = unsafe {
            // Create the module via the import-friendly AddModule interface.
            // https://docs.python.org/3/c-api/import.html#c.PyImport_AddModule
            let ptr = ffi::PyImport_AddModule(CString::new(name)?.as_ptr());

            // Upgrade the borrowed pointer to a strong pointer and bind it to the current runtime.
            let bound = Bound::from_borrowed_ptr_or_err(py, ptr)?;

            // Downcast to the PyModule type.
            bound.downcast_into::<PyModule>()?
        };

        // Technically, we don't need to add the __package__ attribute to the module because it is
        // automatically populated by the loader machinery. Besides, __package__ is used to support
        // relative imports that are irrelevant for native extensions.
        // https://docs.python.org/3/reference/import.html#references
        // https://peps.python.org/pep-0366/

        module.setattr("__file__", PyNone::get(py))?;

        Ok(Self { inner: module })
    }

    /// Creates a new `ImportablePyModuleBuilder` from the given module.
    /// The module is assumed to be already registered with the import system.
    pub fn from(module: Bound<'py, PyModule>) -> Self {
        Self { inner: module }
    }

    /// Sets default attributes to the module.
    pub fn defaults(self) -> PyResult<Self> {
        self.inner.gil_used(false)?;
        Ok(self)
    }

    /// Attaches a submodule to the current module. A module with submodules is called a package
    /// and must have a __path__ attribute. If the __path__ attribute is not present, it is added.
    pub fn add_submodule(self, module: &Bound<'_, PyModule>) -> PyResult<Self> {
        // We only need to add the module name to the current attribute dictionary.
        let fully_qualified_name = module.name()?.extract::<String>()?;
        let name = fully_qualified_name.split('.').last().ok_or_else(|| {
            let msg = format!(
                "Can't extract module name from fully qualified name {fully_qualified_name}"
            );

            PyErr::new::<PyImportError, _>(msg)
        })?;
        self.inner.add(name, module)?;

        // A package is a module that can include other modules together with the
        // classes/functions/etc. The only difference between a package and a module is
        // that a package has a __path__ attribute.
        // __path__ attribute is a 'Sequence' and can be empty, but it must be present.
        // https://docs.python.org/3/reference/import.html#packages
        if !self.inner.hasattr("__path__")? {
            self.inner
                .setattr("__path__", PyList::empty(self.inner.py()))?;
        }
        Ok(self)
    }

    /// Attaches a class to the current module and sets its __module__ attribute to the module name.
    /// If the __module__ attribute is already set to a different value, an error is raised.
    pub fn add_class<T: PyClass>(self) -> PyResult<Self> {
        self.inner.add_class::<T>()?;

        // By default, pyo3 sets the __module__ name to builtins assuming that the class can be
        // attached to multiple modules. This is a fairly rare use case, not used in this project.
        // Therefore, we set the __module__ attribute to the name of the module and simply err if
        // it is already set to a different value.
        let type_object = T::lazy_type_object().get_or_init(self.inner.py());

        let __module__ = type_object.getattr("__module__")?.extract::<String>()?;
        if __module__ == "builtins" {
            type_object.setattr("__module__", self.inner.name()?)?;
            return Ok(self);
        }

        let __name__ = self.inner.name()?.extract::<String>()?;
        if __module__ == __name__ {
            Ok(self)
        } else {
            let err = format!(
                "Class {} is attached to module {} but its __module__ attribute is already set to {}",
                type_object.name()?,
                __name__,
                __module__
            );
            Err(PyErr::new::<PyImportError, _>(err))
        }
    }

    pub fn finish(self) -> Bound<'py, PyModule> {
        self.inner
    }
}
