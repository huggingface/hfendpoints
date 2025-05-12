use crate::headers::X_REQUEST_ID_NAME;
use std::fmt::Debug;
use tokio::net::{TcpListener, ToSocketAddrs};
use tower::ServiceBuilder;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::instrument;
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_scalar::{Scalar, Servable};

mod api;
mod context;
pub mod environ;
pub mod error;
pub mod headers;
mod routes;

use crate::api::ApiDoc;
use crate::environ::Timeout;
use crate::routes::StatusRouter;
pub use context::Context;
pub use error::HttpError;
use hfendpoints_core::environ::TryFromEnv;
use hfendpoints_core::Error;

pub type HttpResult<T> = Result<T, HttpError>;
pub type RequestWithContext<I> = (I, Context);

const STATUS_TAG: &str = "Status";
const STATUS_DESC: &str = "Healthiness and monitoring of the endpoint";

pub const AUDIO_TAG: &str = "Audio";
pub const AUDIO_DESC: &str = "Learn how to turn audio into text or text into audio.";

pub const EMBEDDINGS_TAG: &str = "Embeddings";
pub const EMBEDDINGS_DESC: &str = "Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.";

#[instrument(skip(task_router))]
pub async fn serve_http<A, R>(interface: A, task_router: R) -> HttpResult<()>
where
    A: ToSocketAddrs + Debug,
    R: Into<OpenApiRouter>,
{
    // Retrieve the timeout duration from envvar
    let timeout = Timeout::try_from_env().map_err(|err| Error::Environment(err))?;

    // Default routes
    let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .merge(task_router.into())
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
                .layer(PropagateRequestIdLayer::new(X_REQUEST_ID_NAME.clone()))
                .layer::<TimeoutLayer>(timeout.into()),
        )
        .merge(StatusRouter::default().into())
        .split_for_parts();

    // Documentation route
    let router = router.merge(Scalar::with_url("/docs", api));

    let listener = TcpListener::bind(interface).await?;
    axum::serve(listener, router).await?;
    Ok(())
}
