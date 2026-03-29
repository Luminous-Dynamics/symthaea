// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Currency Support Module
//!
//! Provides currency management, exchange rate tracking, and currency conversion
//! for international business operations.

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

// ============================================================================
// Types
// ============================================================================

/// Currency definition
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Currency {
    pub code: String,
    pub name: String,
    pub symbol: String,
    pub decimal_places: i32,
    pub is_active: bool,
}

/// Exchange rate entry
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ExchangeRate {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub from_currency: String,
    pub to_currency: String,
    pub rate: Decimal,
    pub rate_date: NaiveDate,
    pub rate_type: String,
    pub source: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

/// Rate type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateType {
    Market,
    Custom,
    Budget,
}

impl RateType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RateType::Market => "MARKET",
            RateType::Custom => "CUSTOM",
            RateType::Budget => "BUDGET",
        }
    }
}

/// Tenant currency configuration
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TenantCurrencyConfig {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub base_currency: String,
    pub enabled_currencies: Vec<String>,
    pub rate_source: String,
    pub auto_update_rates: bool,
    pub last_rate_update: Option<DateTime<Utc>>,
    pub rounding_mode: String,
}

/// Currency amount with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneyAmount {
    pub amount: Decimal,
    pub currency: String,
    pub exchange_rate: Option<Decimal>,
    pub base_currency_amount: Option<Decimal>,
}

impl MoneyAmount {
    pub fn new(amount: Decimal, currency: &str) -> Self {
        Self {
            amount,
            currency: currency.to_string(),
            exchange_rate: None,
            base_currency_amount: None,
        }
    }

    pub fn with_conversion(
        amount: Decimal,
        currency: &str,
        exchange_rate: Decimal,
        base_currency_amount: Decimal,
    ) -> Self {
        Self {
            amount,
            currency: currency.to_string(),
            exchange_rate: Some(exchange_rate),
            base_currency_amount: Some(base_currency_amount),
        }
    }
}

// ============================================================================
// Service
// ============================================================================

/// Multi-Currency Service
#[derive(Clone)]
pub struct CurrencyService {
    pool: PgPool,
}

impl CurrencyService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get all active currencies
    pub async fn list_currencies(&self) -> Result<Vec<Currency>, CurrencyError> {
        let currencies = sqlx::query_as::<_, Currency>(
            "SELECT code, name, symbol, decimal_places, is_active
             FROM fin_currencies
             WHERE is_active = true
             ORDER BY code",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        Ok(currencies)
    }

    /// Get a specific currency
    pub async fn get_currency(&self, code: &str) -> Result<Currency, CurrencyError> {
        sqlx::query_as::<_, Currency>(
            "SELECT code, name, symbol, decimal_places, is_active
             FROM fin_currencies
             WHERE code = $1",
        )
        .bind(code)
        .fetch_optional(&self.pool)
        .await
        .map_err(CurrencyError::Database)?
        .ok_or_else(|| CurrencyError::NotFound(format!("Currency {} not found", code)))
    }

    /// Get tenant currency configuration
    pub async fn get_config(&self, tenant_id: Uuid) -> Result<TenantCurrencyConfig, CurrencyError> {
        sqlx::query_as::<_, TenantCurrencyConfig>(
            "SELECT id, tenant_id, base_currency, enabled_currencies,
                    rate_source, auto_update_rates, last_rate_update, rounding_mode
             FROM fin_tenant_currencies
             WHERE tenant_id = $1",
        )
        .bind(tenant_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(CurrencyError::Database)?
        .ok_or_else(|| CurrencyError::NotFound("Currency config not found".into()))
    }

    /// Initialize tenant currency configuration
    pub async fn init_config(
        &self,
        tenant_id: Uuid,
        base_currency: &str,
    ) -> Result<TenantCurrencyConfig, CurrencyError> {
        let id = Uuid::new_v4();

        sqlx::query(
            "INSERT INTO fin_tenant_currencies (id, tenant_id, base_currency, enabled_currencies)
             VALUES ($1, $2, $3, ARRAY[$3])
             ON CONFLICT (tenant_id) DO UPDATE SET base_currency = $3",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(base_currency)
        .execute(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        self.get_config(tenant_id).await
    }

    /// Enable a currency for the tenant
    pub async fn enable_currency(
        &self,
        tenant_id: Uuid,
        currency_code: &str,
    ) -> Result<(), CurrencyError> {
        // Verify currency exists
        self.get_currency(currency_code).await?;

        sqlx::query(
            "UPDATE fin_tenant_currencies
             SET enabled_currencies = array_append(
                 array_remove(enabled_currencies, $2), $2
             ),
             updated_at = NOW()
             WHERE tenant_id = $1",
        )
        .bind(tenant_id)
        .bind(currency_code)
        .execute(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        Ok(())
    }

    /// Disable a currency for the tenant
    pub async fn disable_currency(
        &self,
        tenant_id: Uuid,
        currency_code: &str,
    ) -> Result<(), CurrencyError> {
        // Check it's not the base currency
        let config = self.get_config(tenant_id).await?;
        if config.base_currency == currency_code {
            return Err(CurrencyError::InvalidOperation(
                "Cannot disable base currency".into(),
            ));
        }

        sqlx::query(
            "UPDATE fin_tenant_currencies
             SET enabled_currencies = array_remove(enabled_currencies, $2),
             updated_at = NOW()
             WHERE tenant_id = $1",
        )
        .bind(tenant_id)
        .bind(currency_code)
        .execute(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        Ok(())
    }

    /// Set exchange rate
    pub async fn set_rate(
        &self,
        tenant_id: Uuid,
        from_currency: &str,
        to_currency: &str,
        rate: Decimal,
        rate_date: NaiveDate,
        rate_type: RateType,
        source: Option<&str>,
        user_id: Option<Uuid>,
    ) -> Result<ExchangeRate, CurrencyError> {
        // Validate currencies
        self.get_currency(from_currency).await?;
        self.get_currency(to_currency).await?;

        if from_currency == to_currency {
            return Err(CurrencyError::InvalidOperation(
                "Cannot set rate for same currency".into(),
            ));
        }

        if rate <= Decimal::ZERO {
            return Err(CurrencyError::InvalidOperation(
                "Exchange rate must be positive".into(),
            ));
        }

        let id = Uuid::new_v4();

        sqlx::query(
            "INSERT INTO fin_exchange_rates
             (id, tenant_id, from_currency, to_currency, rate, rate_date, rate_type, source, created_by)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
             ON CONFLICT (tenant_id, from_currency, to_currency, rate_date, rate_type)
             DO UPDATE SET rate = $5, source = $8, created_by = $9",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(from_currency)
        .bind(to_currency)
        .bind(rate)
        .bind(rate_date)
        .bind(rate_type.as_str())
        .bind(source)
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        self.get_rate(tenant_id, from_currency, to_currency, Some(rate_date))
            .await
    }

    /// Get exchange rate
    pub async fn get_rate(
        &self,
        tenant_id: Uuid,
        from_currency: &str,
        to_currency: &str,
        date: Option<NaiveDate>,
    ) -> Result<ExchangeRate, CurrencyError> {
        let rate_date = date.unwrap_or_else(|| Utc::now().date_naive());

        // Same currency = 1.0
        if from_currency == to_currency {
            return Ok(ExchangeRate {
                id: Uuid::nil(),
                tenant_id,
                from_currency: from_currency.to_string(),
                to_currency: to_currency.to_string(),
                rate: Decimal::ONE,
                rate_date,
                rate_type: "MARKET".to_string(),
                source: Some("IDENTITY".to_string()),
                is_active: true,
                created_at: Utc::now(),
            });
        }

        // Try direct rate
        let direct: Option<ExchangeRate> = sqlx::query_as(
            "SELECT id, tenant_id, from_currency, to_currency, rate, rate_date,
                    rate_type, source, is_active, created_at
             FROM fin_exchange_rates
             WHERE tenant_id = $1
               AND from_currency = $2
               AND to_currency = $3
               AND rate_date <= $4
               AND is_active = true
             ORDER BY rate_date DESC
             LIMIT 1",
        )
        .bind(tenant_id)
        .bind(from_currency)
        .bind(to_currency)
        .bind(rate_date)
        .fetch_optional(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        if let Some(rate) = direct {
            return Ok(rate);
        }

        // Try inverse rate
        let inverse: Option<ExchangeRate> = sqlx::query_as(
            "SELECT id, tenant_id, from_currency, to_currency, rate, rate_date,
                    rate_type, source, is_active, created_at
             FROM fin_exchange_rates
             WHERE tenant_id = $1
               AND from_currency = $3
               AND to_currency = $2
               AND rate_date <= $4
               AND is_active = true
             ORDER BY rate_date DESC
             LIMIT 1",
        )
        .bind(tenant_id)
        .bind(from_currency)
        .bind(to_currency)
        .bind(rate_date)
        .fetch_optional(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        if let Some(mut rate) = inverse {
            rate.rate = Decimal::ONE / rate.rate;
            rate.from_currency = from_currency.to_string();
            rate.to_currency = to_currency.to_string();
            rate.source = Some("INVERSE".to_string());
            return Ok(rate);
        }

        // Try triangulation through USD (non-recursive - direct DB lookups)
        if from_currency != "USD" && to_currency != "USD" {
            let from_usd = self.get_direct_rate(tenant_id, from_currency, "USD", rate_date).await?;
            let usd_to = self.get_direct_rate(tenant_id, "USD", to_currency, rate_date).await?;

            if let (Some(f_rate), Some(t_rate)) = (from_usd, usd_to) {
                return Ok(ExchangeRate {
                    id: Uuid::nil(),
                    tenant_id,
                    from_currency: from_currency.to_string(),
                    to_currency: to_currency.to_string(),
                    rate: f_rate * t_rate,
                    rate_date,
                    rate_type: "MARKET".to_string(),
                    source: Some("TRIANGULATED".to_string()),
                    is_active: true,
                    created_at: Utc::now(),
                });
            }
        }

        Err(CurrencyError::NoRate(format!(
            "No exchange rate found for {} to {}",
            from_currency, to_currency
        )))
    }

    /// Get direct or inverse rate (non-recursive helper)
    async fn get_direct_rate(
        &self,
        tenant_id: Uuid,
        from_currency: &str,
        to_currency: &str,
        rate_date: NaiveDate,
    ) -> Result<Option<Decimal>, CurrencyError> {
        // Try direct
        let direct: Option<(Decimal,)> = sqlx::query_as(
            "SELECT rate FROM fin_exchange_rates
             WHERE tenant_id = $1
               AND from_currency = $2
               AND to_currency = $3
               AND rate_date <= $4
               AND is_active = true
             ORDER BY rate_date DESC
             LIMIT 1",
        )
        .bind(tenant_id)
        .bind(from_currency)
        .bind(to_currency)
        .bind(rate_date)
        .fetch_optional(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        if let Some((rate,)) = direct {
            return Ok(Some(rate));
        }

        // Try inverse
        let inverse: Option<(Decimal,)> = sqlx::query_as(
            "SELECT rate FROM fin_exchange_rates
             WHERE tenant_id = $1
               AND from_currency = $2
               AND to_currency = $3
               AND rate_date <= $4
               AND is_active = true
             ORDER BY rate_date DESC
             LIMIT 1",
        )
        .bind(tenant_id)
        .bind(to_currency)
        .bind(from_currency)
        .bind(rate_date)
        .fetch_optional(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        if let Some((rate,)) = inverse {
            return Ok(Some(Decimal::ONE / rate));
        }

        Ok(None)
    }

    /// Convert amount between currencies
    pub async fn convert(
        &self,
        tenant_id: Uuid,
        amount: Decimal,
        from_currency: &str,
        to_currency: &str,
        date: Option<NaiveDate>,
    ) -> Result<MoneyAmount, CurrencyError> {
        let rate_info = self.get_rate(tenant_id, from_currency, to_currency, date).await?;

        let target_currency = self.get_currency(to_currency).await?;
        let decimal_places = target_currency.decimal_places as u32;

        let converted = (amount * rate_info.rate)
            .round_dp_with_strategy(decimal_places, rust_decimal::RoundingStrategy::MidpointAwayFromZero);

        Ok(MoneyAmount::with_conversion(
            converted,
            to_currency,
            rate_info.rate,
            amount,
        ))
    }

    /// Convert to base currency
    pub async fn convert_to_base(
        &self,
        tenant_id: Uuid,
        amount: Decimal,
        from_currency: &str,
        date: Option<NaiveDate>,
    ) -> Result<MoneyAmount, CurrencyError> {
        let config = self.get_config(tenant_id).await?;
        self.convert(tenant_id, amount, from_currency, &config.base_currency, date)
            .await
    }

    /// List exchange rates for a date range
    pub async fn list_rates(
        &self,
        tenant_id: Uuid,
        from_date: NaiveDate,
        to_date: NaiveDate,
    ) -> Result<Vec<ExchangeRate>, CurrencyError> {
        let rates = sqlx::query_as::<_, ExchangeRate>(
            "SELECT id, tenant_id, from_currency, to_currency, rate, rate_date,
                    rate_type, source, is_active, created_at
             FROM fin_exchange_rates
             WHERE tenant_id = $1
               AND rate_date >= $2
               AND rate_date <= $3
               AND is_active = true
             ORDER BY rate_date DESC, from_currency, to_currency",
        )
        .bind(tenant_id)
        .bind(from_date)
        .bind(to_date)
        .fetch_all(&self.pool)
        .await
        .map_err(CurrencyError::Database)?;

        Ok(rates)
    }

    /// Format money for display
    pub fn format_money(&self, currency: &Currency, amount: Decimal) -> String {
        let formatted = format!("{:.prec$}", amount, prec = currency.decimal_places as usize);
        format!("{}{}", currency.symbol, formatted)
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Currency errors
#[derive(Debug, thiserror::Error)]
pub enum CurrencyError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("No exchange rate: {0}")]
    NoRate(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("API error: {0}")]
    ApiError(String),
}
