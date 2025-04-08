use axum::http::{HeaderName, HeaderValue};
use headers::{Error, Header};
use std::borrow::Cow;
use std::ops::Deref;

static X_REQUEST_ID_NAME: HeaderName = HeaderName::from_static("x-request-id");


/// Holds the value of the x-request-id header used to
/// correlate request and execution within the server.
#[derive(Debug, Clone)]
pub struct RequestId(Cow<'static, str>);

impl Deref for RequestId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        match &self.0 {
            Cow::Borrowed(value) => *value,
            Cow::Owned(value) => value.as_str()
        }
    }
}

impl Header for RequestId {
    fn name() -> &'static HeaderName {
        &X_REQUEST_ID_NAME
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, Error>
    where
        Self: Sized,
        I: Iterator<Item=&'i HeaderValue>,
    {
        let value = values
            .next()
            .ok_or_else(headers::Error::invalid)?;

        Ok(RequestId(Cow::from(value.to_str().map_err(|err| {
            Error::invalid()
        })?.to_string())))
    }

    fn encode<E: Extend<HeaderValue>>(&self, values: &mut E) {
        let value = HeaderValue::from_str(&self.0).unwrap();
        values.extend(std::iter::once(value));
    }
}