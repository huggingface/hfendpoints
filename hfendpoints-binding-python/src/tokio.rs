use std::sync::atomic::AtomicU16;
use std::sync::atomic::Ordering::SeqCst;
use tokio::runtime::Builder;


/// Create a new Tokio multithreaded runtime with predefined thread-name
pub fn create_multithreaded_runtime() -> Builder {
    let mut builder = Builder::new_multi_thread();
    builder.enable_all().thread_name_fn(|| {
        static THREAD_ID: AtomicU16 = AtomicU16::new(0);
        format!("hfendpoints-thread-{}", THREAD_ID.fetch_add(1, SeqCst))
    });

    builder
}