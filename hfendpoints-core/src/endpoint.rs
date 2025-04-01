use std::thread::JoinHandle;

///
pub trait Endpoint {
    fn spawn_handler(&self) -> JoinHandle<()>;
}
