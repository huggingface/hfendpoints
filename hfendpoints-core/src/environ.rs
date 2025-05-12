use std::borrow::Cow;
use thiserror::Error;

#[derive(Clone, Error, Debug)]
pub enum EnvironmentError {
    #[error(
        "Required environment variable {0} not found. Please define this variable {0}=... and relaunch the application."
    )]
    MissingEnvVar(String),

    #[error(
        "Found evironment variable {0} but validation failed: {1}. Please fix this variable and relaunch the application."
    )]
    InvalidEnvVar(Cow<'static, str>, String),
}

/// Super trait for all variables which can be inferred at runtime, from the environment variables
pub trait FromEnv {
    const ENV_VAR_NAME: &'static str;

    fn from_env() -> Self;
}

/// Super trait for all variables which can be inferred at runtime, from the environment variables
/// but whose extraction may fail
pub trait TryFromEnv {
    const ENV_VAR_NAME: &'static str;

    fn try_from_env() -> Result<Self, EnvironmentError>
    where
        Self: Sized;
}
