use crate::STATUS_TAG;
use axum::http::StatusCode;
use tracing::instrument;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

#[utoipa::path(
    method(get, head),
    path = "/health",
    tag = STATUS_TAG,
    responses(
        (status = OK, description = "Success", body = str, content_type = "application/json")
    )
)]
#[instrument]
pub async fn health() -> StatusCode {
    StatusCode::OK
}

/// Provides all the routes to report status
#[derive(Default)]
pub struct StatusRouter;

/// Convert the underlying `StatusRouter` to one compatible with `utoipa_axum::router::OpenApiRouter`
impl From<StatusRouter> for OpenApiRouter {
    fn from(_: StatusRouter) -> Self {
        OpenApiRouter::new().routes(routes!(health))
    }
}
