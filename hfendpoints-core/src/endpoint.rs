use crate::Error;

/// Super-trait defining a type of endpoint
pub trait Endpoint<A> {
    /// Start the underlying endpoint and starts serving requests
    ///
    /// This method is effectively asynchronous, but to not mess-up in trait, we use the `impl Future<>` definition
    fn serve(&self, binding: A) -> impl Future<Output = Result<(), Error>> + Send;
}
