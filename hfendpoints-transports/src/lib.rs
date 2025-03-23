pub enum TransportError {
    ValidationError(String),
}

/// Utility type to represent a Result with associated TransportError type
pub type TransportResult<T> = Result<T, TransportError>;

pub trait Transport {}
