///
pub enum InferError {}

///
pub type InferResult<T> = Result<T, InferError>;

pub trait InferService {
    type Request;
    type Response;

    ///
    ///
    /// # Arguments
    ///
    /// * `request`:
    ///
    /// returns: Result<Option<String>, InferError>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    fn infer(&mut self, request: Self::Request) -> InferResult<Option<String>>;
}
