use crate::Error;

///
pub trait Endpoint<A> {
    fn serve(&self, binding: A) -> impl Future<Output=Result<(), Error>> + Send;
}
