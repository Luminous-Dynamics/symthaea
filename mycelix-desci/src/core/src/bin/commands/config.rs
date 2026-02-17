use mycelix_desci_core::{Config, Result};

pub async fn show(config: Config) -> Result<()> {
    let toml = toml::to_string_pretty(&config).map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to serialize config: {}", e))
    })?;

    println!("{}", toml);

    Ok(())
}

pub async fn validate(config: Config) -> Result<()> {
    config.validate()?;
    println!("✓ Configuration is valid!");

    Ok(())
}
