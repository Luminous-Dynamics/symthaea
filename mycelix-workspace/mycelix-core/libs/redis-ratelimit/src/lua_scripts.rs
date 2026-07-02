// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lua scripts for atomic rate limiting operations
//!
//! These scripts run atomically on Redis, ensuring consistency
//! across distributed service instances.

/// Sliding window counter Lua script
///
/// Uses two counters (current and previous window) with weighted combination.
/// More memory efficient than sliding window log.
pub const SLIDING_WINDOW_COUNTER: &str = r#"
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4]) or 1

-- Calculate window boundaries
local window_start = math.floor(now_ms / window_ms) * window_ms
local prev_window_start = window_start - window_ms

-- Keys for current and previous windows
local curr_key = key .. ":" .. window_start
local prev_key = key .. ":" .. prev_window_start

-- Get counts
local curr_count = tonumber(redis.call("GET", curr_key) or "0")
local prev_count = tonumber(redis.call("GET", prev_key) or "0")

-- Calculate weighted count using sliding window approximation
local elapsed_in_window = now_ms - window_start
local weight = (window_ms - elapsed_in_window) / window_ms
local weighted_count = curr_count + (prev_count * weight)

-- Check if we can proceed
local remaining = limit - weighted_count
local allowed = remaining >= cost

if allowed then
    -- Increment current window counter
    redis.call("INCRBY", curr_key, cost)
    -- Set expiry (2x window to cover overlap)
    redis.call("PEXPIRE", curr_key, window_ms * 2)
    remaining = remaining - cost
end

-- Calculate retry time if limited
local retry_after = 0
if not allowed then
    -- Time until enough capacity frees up
    local needed = cost - remaining
    local rate_per_ms = limit / window_ms
    retry_after = math.ceil(needed / rate_per_ms)
end

return {allowed and 1 or 0, math.floor(remaining), retry_after, math.floor(weighted_count + (allowed and cost or 0))}
"#;

/// Token bucket Lua script
///
/// Allows bursts up to capacity while maintaining average rate.
pub const TOKEN_BUCKET: &str = r#"
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4]) or 1

-- Get current state
local data = redis.call("HMGET", key, "tokens", "last_update")
local tokens = tonumber(data[1])
local last_update = tonumber(data[2])

-- Initialize if new
if tokens == nil then
    tokens = capacity
    last_update = now_ms
end

-- Calculate tokens to add based on elapsed time
local elapsed_ms = now_ms - last_update
local tokens_to_add = (elapsed_ms / 1000.0) * refill_rate
tokens = math.min(capacity, tokens + tokens_to_add)

-- Check if we can consume
local allowed = tokens >= cost
local remaining = tokens

if allowed then
    tokens = tokens - cost
    remaining = tokens
end

-- Update state
redis.call("HMSET", key, "tokens", tokens, "last_update", now_ms)
-- Expire after enough time to fully refill
local ttl_ms = math.ceil((capacity / refill_rate) * 1000 * 2)
redis.call("PEXPIRE", key, ttl_ms)

-- Calculate retry time
local retry_after = 0
if not allowed then
    local needed = cost - tokens
    retry_after = math.ceil((needed / refill_rate) * 1000)
end

return {allowed and 1 or 0, math.floor(remaining), retry_after, math.floor(capacity - remaining)}
"#;

/// Fixed window Lua script
///
/// Simple counter that resets at window boundaries.
pub const FIXED_WINDOW: &str = r#"
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4]) or 1

-- Calculate current window
local window_start = math.floor(now_ms / window_ms) * window_ms
local window_key = key .. ":" .. window_start

-- Get current count
local count = tonumber(redis.call("GET", window_key) or "0")

-- Check limit
local remaining = limit - count
local allowed = remaining >= cost

if allowed then
    redis.call("INCRBY", window_key, cost)
    redis.call("PEXPIRE", window_key, window_ms)
    remaining = remaining - cost
end

-- Calculate retry time (until window resets)
local retry_after = 0
if not allowed then
    local window_end = window_start + window_ms
    retry_after = window_end - now_ms
end

return {allowed and 1 or 0, math.floor(remaining), retry_after, count + (allowed and cost or 0)}
"#;

/// Leaky bucket Lua script
///
/// Constant output rate regardless of input pattern.
pub const LEAKY_BUCKET: &str = r#"
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local leak_rate = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4]) or 1

-- Get current state
local data = redis.call("HMGET", key, "level", "last_leak")
local level = tonumber(data[1]) or 0
local last_leak = tonumber(data[2]) or now_ms

-- Calculate leakage since last update
local elapsed_ms = now_ms - last_leak
local leaked = (elapsed_ms / 1000.0) * leak_rate
level = math.max(0, level - leaked)

-- Check if we can add to bucket
local space = capacity - level
local allowed = space >= cost

if allowed then
    level = level + cost
end

-- Update state
redis.call("HMSET", key, "level", level, "last_leak", now_ms)
-- Expire after bucket would be empty
local ttl_ms = math.ceil((capacity / leak_rate) * 1000 * 2)
redis.call("PEXPIRE", key, ttl_ms)

-- Calculate retry time
local retry_after = 0
if not allowed then
    local needed = cost - space
    retry_after = math.ceil((needed / leak_rate) * 1000)
end

local remaining = math.floor(capacity - level)
return {allowed and 1 or 0, remaining, retry_after, math.floor(level)}
"#;

/// Get rate limit info without consuming (read-only)
pub const GET_INFO: &str = r#"
local key = KEYS[1]
local algorithm = ARGV[1]
local limit = tonumber(ARGV[2])
local window_ms = tonumber(ARGV[3])
local now_ms = tonumber(ARGV[4])

if algorithm == "sliding_window" then
    local window_start = math.floor(now_ms / window_ms) * window_ms
    local prev_window_start = window_start - window_ms

    local curr_count = tonumber(redis.call("GET", key .. ":" .. window_start) or "0")
    local prev_count = tonumber(redis.call("GET", key .. ":" .. prev_window_start) or "0")

    local elapsed_in_window = now_ms - window_start
    local weight = (window_ms - elapsed_in_window) / window_ms
    local weighted_count = curr_count + (prev_count * weight)

    return {math.floor(limit - weighted_count), math.floor(weighted_count)}

elseif algorithm == "token_bucket" then
    local data = redis.call("HMGET", key, "tokens", "last_update")
    local tokens = tonumber(data[1]) or limit
    return {math.floor(tokens), math.floor(limit - tokens)}

elseif algorithm == "fixed_window" then
    local window_start = math.floor(now_ms / window_ms) * window_ms
    local count = tonumber(redis.call("GET", key .. ":" .. window_start) or "0")
    return {math.floor(limit - count), count}

elseif algorithm == "leaky_bucket" then
    local data = redis.call("HMGET", key, "level", "last_leak")
    local level = tonumber(data[1]) or 0
    return {math.floor(limit - level), math.floor(level)}
end

return {limit, 0}
"#;

/// Reset rate limit for a key
pub const RESET: &str = r#"
local key = KEYS[1]
local pattern = key .. ":*"

-- Delete main key
redis.call("DEL", key)

-- Delete any sub-keys (for window-based algorithms)
local keys = redis.call("KEYS", pattern)
for i, k in ipairs(keys) do
    redis.call("DEL", k)
end

return #keys + 1
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scripts_not_empty() {
        assert!(!SLIDING_WINDOW_COUNTER.is_empty());
        assert!(!TOKEN_BUCKET.is_empty());
        assert!(!FIXED_WINDOW.is_empty());
        assert!(!LEAKY_BUCKET.is_empty());
        assert!(!GET_INFO.is_empty());
        assert!(!RESET.is_empty());
    }

    #[test]
    fn test_scripts_contain_return() {
        // All scripts should return values
        assert!(SLIDING_WINDOW_COUNTER.contains("return"));
        assert!(TOKEN_BUCKET.contains("return"));
        assert!(FIXED_WINDOW.contains("return"));
        assert!(LEAKY_BUCKET.contains("return"));
    }
}
