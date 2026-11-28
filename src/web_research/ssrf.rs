//! Capability-constrained outbound destination policy for WebResearch.
//!
//! DNS answers are rejected as a whole if any address is non-public or special-use.
//! The exact validated SocketAddr set is returned to reqwest's connector, closing
//! the validation/connect re-resolution gap. IP-literal URLs are checked separately
//! because reqwest does not invoke the DNS resolver for those URLs.
//!
//! IPv4/IPv6 special-use prefixes are a conservative snapshot of the IANA
//! special-purpose registries last updated 2025-10-09. Every listed range is
//! denied, including some registry entries marked globally reachable, because
//! protocol-service anycast is not needed by public literature retrieval.
//! Update the list only with a reviewed registry snapshot and tests.

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use url::{Host, Url};

type ResolverError = Box<dyn Error + Send + Sync + 'static>;

const IPV4_SPECIAL_USE: &[(&str, u8)] = &[
    ("0.0.0.0", 8),
    ("0.0.0.0", 32),
    ("10.0.0.0", 8),
    ("100.64.0.0", 10),
    ("127.0.0.0", 8),
    ("169.254.0.0", 16),
    ("172.16.0.0", 12),
    ("192.0.0.0", 24),
    ("192.0.0.0", 29),
    ("192.0.0.8", 32),
    ("192.0.0.9", 32),
    ("192.0.0.10", 32),
    ("192.0.0.170", 32),
    ("192.0.0.171", 32),
    ("192.0.2.0", 24),
    ("192.31.196.0", 24),
    ("192.52.193.0", 24),
    ("192.88.99.0", 24),
    ("192.88.99.2", 32),
    ("192.168.0.0", 16),
    ("192.175.48.0", 24),
    ("198.18.0.0", 15),
    ("198.51.100.0", 24),
    ("203.0.113.0", 24),
    ("240.0.0.0", 4),
    ("255.255.255.255", 32),
];

const IPV6_SPECIAL_USE: &[(&str, u8)] = &[
    ("::1", 128),
    ("::", 128),
    ("::ffff:0:0", 96),
    ("64:ff9b::", 96),
    ("64:ff9b:1::", 48),
    ("100::", 64),
    ("100:0:0:1::", 64),
    ("2001::", 23),
    ("2001::", 32),
    ("2001:1::1", 128),
    ("2001:1::2", 128),
    ("2001:1::3", 128),
    ("2001:2::", 48),
    ("2001:3::", 32),
    ("2001:4:112::", 48),
    ("2001:10::", 28),
    ("2001:20::", 28),
    ("2001:30::", 28),
    ("2001:db8::", 32),
    ("2002::", 16),
    ("2620:4f:8000::", 48),
    ("3fff::", 20),
    ("5f00::", 16),
    ("fc00::", 7),
    ("fe80::", 10),
    // Deprecated site-local unicast is not an acceptable public destination.
    ("fec0::", 10),
    // IPv4-compatible IPv6 encodings are deprecated and must not bypass checks.
    ("::", 96),
];

const LOCAL_HOST_SUFFIXES: &[&str] = &[
    "localhost",
    "local",
    "internal",
    "home.arpa",
    "test",
    "example",
    "invalid",
    "lan",
    "localdomain",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UrlPolicyRejection {
    InvalidUrl(String),
    UnsupportedScheme(String),
    EmbeddedCredentials,
    DisallowedPort(u16),
    LocalOnlyHostname(String),
    ForbiddenIpLiteral(IpAddr),
}

impl fmt::Display for UrlPolicyRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUrl(reason) => write!(f, "URL_POLICY_REJECTED: invalid URL: {reason}"),
            Self::UnsupportedScheme(scheme) => {
                write!(f, "URL_POLICY_REJECTED: unsupported scheme {scheme}")
            }
            Self::EmbeddedCredentials => {
                f.write_str("URL_POLICY_REJECTED: embedded URL credentials are forbidden")
            }
            Self::DisallowedPort(port) => {
                write!(
                    f,
                    "URL_POLICY_REJECTED: destination port {port} is not allowed"
                )
            }
            Self::LocalOnlyHostname(host) => {
                write!(
                    f,
                    "URL_POLICY_REJECTED: local-only hostname {host} is forbidden"
                )
            }
            Self::ForbiddenIpLiteral(ip) => {
                write!(
                    f,
                    "URL_POLICY_REJECTED: non-public IP literal {ip} is forbidden"
                )
            }
        }
    }
}

impl Error for UrlPolicyRejection {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedirectPolicyRejection {
    pub status: u16,
}

impl fmt::Display for RedirectPolicyRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "REDIRECT_REJECTED: HTTP {} redirect was not followed",
            self.status
        )
    }
}

impl Error for RedirectPolicyRejection {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DnsPolicyRejection {
    LocalOnlyHostname(String),
    EmptyAnswerSet(String),
    NonPublicAnswerSet {
        hostname: String,
        denied_addresses: Vec<IpAddr>,
    },
}

impl fmt::Display for DnsPolicyRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalOnlyHostname(host) => {
                write!(
                    f,
                    "DNS_POLICY_REJECTED: local-only hostname {host} is forbidden"
                )
            }
            Self::EmptyAnswerSet(host) => {
                write!(
                    f,
                    "DNS_POLICY_REJECTED: {host} returned no destination addresses"
                )
            }
            Self::NonPublicAnswerSet {
                hostname,
                denied_addresses,
            } => write!(
                f,
                "DNS_POLICY_REJECTED: {hostname} returned a mixed or non-public address set: {denied_addresses:?}"
            ),
        }
    }
}

impl Error for DnsPolicyRejection {}

/// URL-level checks catch IP literals (which do not pass through DNS), credentials,
/// local-only hostnames, unsupported schemes and non-default destination ports.
pub fn validate_request_url(input: &str) -> Result<Url, UrlPolicyRejection> {
    let url =
        Url::parse(input).map_err(|error| UrlPolicyRejection::InvalidUrl(error.to_string()))?;
    let expected_port = match url.scheme() {
        "http" => 80,
        "https" => 443,
        other => return Err(UrlPolicyRejection::UnsupportedScheme(other.to_string())),
    };

    let authority = input
        .split_once("://")
        .map(|(_, rest)| rest.split(['/', '?', '#']).next().unwrap_or_default())
        .unwrap_or_default();
    if !url.username().is_empty() || url.password().is_some() || authority.contains('@') {
        return Err(UrlPolicyRejection::EmbeddedCredentials);
    }

    if let Some(port) = url.port()
        && port != expected_port
    {
        return Err(UrlPolicyRejection::DisallowedPort(port));
    }

    match url.host() {
        Some(Host::Ipv4(ip)) if !is_public_destination(IpAddr::V4(ip)) => {
            return Err(UrlPolicyRejection::ForbiddenIpLiteral(IpAddr::V4(ip)));
        }
        Some(Host::Ipv6(ip)) if !is_public_destination(IpAddr::V6(ip)) => {
            return Err(UrlPolicyRejection::ForbiddenIpLiteral(IpAddr::V6(ip)));
        }
        Some(Host::Domain(host)) if is_local_only_hostname(host) => {
            return Err(UrlPolicyRejection::LocalOnlyHostname(host.to_string()));
        }
        Some(_) => {}
        None => {
            return Err(UrlPolicyRejection::InvalidUrl(
                "URL has no host".to_string(),
            ));
        }
    }

    Ok(url)
}

fn is_local_only_hostname(hostname: &str) -> bool {
    let normalized = hostname.trim_end_matches('.').to_ascii_lowercase();
    LOCAL_HOST_SUFFIXES
        .iter()
        .any(|suffix| normalized == *suffix || normalized.ends_with(&format!(".{suffix}")))
}

/// Conservative destination allow policy.
///
/// IPv6 is allowlisted to global-unicast 2000::/3 after excluding all IANA
/// special-use entries and multicast/loopback/unspecified addresses. IPv4
/// excludes private, loopback, link-local, multicast, unspecified, broadcast
/// and the complete pinned IANA special-purpose snapshot above.
pub fn is_public_destination(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(ip) => {
            if ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip == Ipv4Addr::BROADCAST
            {
                return false;
            }
            !IPV4_SPECIAL_USE.iter().any(|(network, prefix)| {
                network
                    .parse::<Ipv4Addr>()
                    .is_ok_and(|base| ipv4_in_prefix(ip, base, *prefix))
            })
        }
        IpAddr::V6(ip) => {
            let value = u128::from(ip);
            let global_unicast = ipv6_in_prefix(ip, Ipv6Addr::from(0x2000_u128 << 112), 3);
            if !global_unicast || ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
                return false;
            }
            !IPV6_SPECIAL_USE.iter().any(|(network, prefix)| {
                network
                    .parse::<Ipv6Addr>()
                    .is_ok_and(|base| ipv6_in_prefix(ip, base, *prefix))
            }) && value != 0
        }
    }
}

fn ipv4_in_prefix(address: Ipv4Addr, network: Ipv4Addr, prefix: u8) -> bool {
    if prefix == 0 {
        return true;
    }
    let mask = u32::MAX << (32 - u32::from(prefix));
    (u32::from(address) & mask) == (u32::from(network) & mask)
}

fn ipv6_in_prefix(address: Ipv6Addr, network: Ipv6Addr, prefix: u8) -> bool {
    if prefix == 0 {
        return true;
    }
    let mask = u128::MAX << (128 - u32::from(prefix));
    (u128::from(address) & mask) == (u128::from(network) & mask)
}

pub type LookupFuture = Pin<Box<dyn Future<Output = io::Result<Vec<SocketAddr>>> + Send + 'static>>;

/// Injectable raw hostname lookup. Implementations must return the complete A/AAAA
/// answer set for one lookup attempt, not a pre-filtered subset.
pub trait HostLookup: Send + Sync {
    fn lookup(&self, hostname: String) -> LookupFuture;
}

#[derive(Debug, Default)]
pub struct SystemHostLookup;

impl HostLookup for SystemHostLookup {
    fn lookup(&self, hostname: String) -> LookupFuture {
        Box::pin(async move {
            let answers = tokio::net::lookup_host((hostname.as_str(), 0)).await?;
            Ok(answers.collect())
        })
    }
}

#[derive(Clone)]
pub struct GuardedDnsResolver {
    lookup: Arc<dyn HostLookup>,
}

impl fmt::Debug for GuardedDnsResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuardedDnsResolver").finish_non_exhaustive()
    }
}

impl GuardedDnsResolver {
    pub fn system() -> Self {
        Self::with_lookup(SystemHostLookup)
    }

    pub fn with_lookup<L>(lookup: L) -> Self
    where
        L: HostLookup + 'static,
    {
        Self {
            lookup: Arc::new(lookup),
        }
    }
}

impl Resolve for GuardedDnsResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let hostname = name.as_str().trim_end_matches('.').to_ascii_lowercase();
        let lookup = self.lookup.clone();
        Box::pin(async move {
            if is_local_only_hostname(&hostname) {
                return Err(
                    Box::new(DnsPolicyRejection::LocalOnlyHostname(hostname)) as ResolverError
                );
            }

            let addresses = lookup.lookup(hostname.clone()).await?;
            if addresses.is_empty() {
                return Err(Box::new(DnsPolicyRejection::EmptyAnswerSet(hostname)) as ResolverError);
            }

            let denied_addresses = addresses
                .iter()
                .map(SocketAddr::ip)
                .filter(|address| !is_public_destination(*address))
                .collect::<Vec<_>>();
            if !denied_addresses.is_empty() {
                return Err(Box::new(DnsPolicyRejection::NonPublicAnswerSet {
                    hostname,
                    denied_addresses,
                }) as ResolverError);
            }

            let pinned: Addrs = Box::new(addresses.into_iter());
            Ok(pinned)
        })
    }
}

/// Classify typed URL-policy failures carried through anyhow context.
pub fn is_url_policy_rejection_anyhow(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<UrlPolicyRejection>().is_some())
}

/// Classify typed DNS-policy failures carried through anyhow context.
pub fn is_dns_policy_rejection_anyhow(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<DnsPolicyRejection>().is_some())
}

/// Refuse every 3xx response consistently. The client is configured with
/// redirect Policy::none(), so the response is the terminal result and no
/// Location target has been contacted.
pub fn reject_redirect_response(response: reqwest::Response) -> anyhow::Result<reqwest::Response> {
    let status = response.status();
    if status.is_redirection() {
        return Err(anyhow::Error::new(RedirectPolicyRejection {
            status: status.as_u16(),
        }));
    }
    Ok(response)
}

/// Classify typed redirect rejections carried through anyhow context.
pub fn is_redirect_policy_rejection_anyhow(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<RedirectPolicyRejection>().is_some())
}

/// Used to classify policy failures in the acquisition receipt without turning
/// ordinary DNS timeouts or resolver failures into policy rejections.
pub fn is_dns_policy_rejection(error: &reqwest::Error) -> bool {
    let mut source = error.source();
    while let Some(current) = source {
        if current.downcast_ref::<DnsPolicyRejection>().is_some() {
            return true;
        }
        source = current.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::str::FromStr;
    use std::sync::Mutex;

    #[derive(Default)]
    struct StubLookup {
        answers: Mutex<VecDeque<Vec<SocketAddr>>>,
    }

    impl StubLookup {
        fn with_answers(answers: Vec<Vec<SocketAddr>>) -> Self {
            Self {
                answers: Mutex::new(answers.into()),
            }
        }
    }

    impl HostLookup for StubLookup {
        fn lookup(&self, _hostname: String) -> LookupFuture {
            let result = self
                .answers
                .lock()
                .expect("stub lock")
                .pop_front()
                .unwrap_or_default();
            Box::pin(async move { Ok(result) })
        }
    }

    async fn resolve(
        resolver: &GuardedDnsResolver,
        hostname: &str,
    ) -> Result<Vec<SocketAddr>, String> {
        let name = Name::from_str(hostname).expect("valid test hostname");
        resolver
            .resolve(name)
            .await
            .map(|addresses| addresses.collect())
            .map_err(|error| error.to_string())
    }

    fn socket(address: &str) -> SocketAddr {
        format!("{address}:0")
            .parse()
            .expect("valid socket address")
    }

    fn last_ipv4(network: Ipv4Addr, prefix: u8) -> Ipv4Addr {
        let mask = if prefix == 0 {
            0
        } else {
            u32::MAX << (32 - u32::from(prefix))
        };
        Ipv4Addr::from(u32::from(network) | !mask)
    }

    fn last_ipv6(network: Ipv6Addr, prefix: u8) -> Ipv6Addr {
        let mask = if prefix == 0 {
            0
        } else {
            u128::MAX << (128 - u32::from(prefix))
        };
        Ipv6Addr::from(u128::from(network) | !mask)
    }

    #[test]
    fn allows_global_unicast_v4_and_v6() {
        assert!(is_public_destination("8.8.8.8".parse().unwrap()));
        assert!(is_public_destination(
            "2606:4700:4700::1111".parse().unwrap()
        ));
    }

    #[test]
    fn rejects_private_and_special_use_v4() {
        for (network, prefix) in IPV4_SPECIAL_USE {
            let address = network.parse::<Ipv4Addr>().unwrap();
            let last = last_ipv4(address, *prefix);
            assert!(
                !is_public_destination(IpAddr::V4(address)),
                "start {network}/{prefix}"
            );
            assert!(
                !is_public_destination(IpAddr::V4(last)),
                "end {network}/{prefix}: {last}"
            );
        }
    }

    #[test]
    fn rejects_special_use_v6_and_non_global_ranges() {
        for (network, prefix) in IPV6_SPECIAL_USE {
            let address = network.parse::<Ipv6Addr>().unwrap();
            let last = last_ipv6(address, *prefix);
            assert!(
                !is_public_destination(IpAddr::V6(address)),
                "start {network}/{prefix}"
            );
            assert!(
                !is_public_destination(IpAddr::V6(last)),
                "end {network}/{prefix}: {last}"
            );
        }
        assert!(!is_public_destination("ff02::1".parse().unwrap()));
        assert!(!is_public_destination("fe80::1".parse().unwrap()));
        assert!(!is_public_destination("fc00::1".parse().unwrap()));
        assert!(!is_public_destination("::ffff:8.8.8.8".parse().unwrap()));
    }

    #[tokio::test]
    async fn pins_exact_validated_public_dns_answer_set() {
        let expected = vec![socket("8.8.8.8"), socket("1.1.1.1")];
        let resolver =
            GuardedDnsResolver::with_lookup(StubLookup::with_answers(vec![expected.clone()]));
        let actual = resolve(&resolver, "public.example").await.unwrap();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn rejects_entire_mixed_public_private_answer_set() {
        let answers = vec![socket("8.8.8.8"), socket("10.1.2.3")];
        let resolver = GuardedDnsResolver::with_lookup(StubLookup::with_answers(vec![answers]));
        let error = resolve(&resolver, "mixed.example").await.unwrap_err();
        assert!(error.contains("DNS_POLICY_REJECTED"));
    }

    #[tokio::test]
    async fn rejects_entire_mixed_ipv4_ipv6_answer_set() {
        let answers = vec![
            socket("8.8.8.8"),
            socket("2606:4700:4700::1111"),
            socket("fc00::1"),
        ];
        let resolver = GuardedDnsResolver::with_lookup(StubLookup::with_answers(vec![answers]));
        let error = resolve(&resolver, "mixed-v4-v6.example").await.unwrap_err();
        assert!(error.contains("DNS_POLICY_REJECTED"));
        assert!(error.contains("fc00::1"));
    }

    #[tokio::test]
    async fn rejects_empty_answer_set() {
        let resolver = GuardedDnsResolver::with_lookup(StubLookup::with_answers(vec![Vec::new()]));
        assert!(
            resolve(&resolver, "empty.example")
                .await
                .unwrap_err()
                .contains("DNS_POLICY_REJECTED")
        );
    }

    #[tokio::test]
    async fn validates_each_resolution_attempt_against_dns_rebinding() {
        let answers = vec![vec![socket("8.8.8.8")], vec![socket("127.0.0.1")]];
        let resolver = GuardedDnsResolver::with_lookup(StubLookup::with_answers(answers));
        assert_eq!(
            resolve(&resolver, "rebind.example").await.unwrap(),
            vec![socket("8.8.8.8")]
        );
        assert!(
            resolve(&resolver, "rebind.example")
                .await
                .unwrap_err()
                .contains("DNS_POLICY_REJECTED")
        );
    }

    #[test]
    fn all_http_redirect_statuses_are_typed_rejections() {
        for status in [
            reqwest::StatusCode::MOVED_PERMANENTLY,
            reqwest::StatusCode::FOUND,
            reqwest::StatusCode::SEE_OTHER,
            reqwest::StatusCode::TEMPORARY_REDIRECT,
            reqwest::StatusCode::PERMANENT_REDIRECT,
        ] {
            let error = anyhow::Error::new(RedirectPolicyRejection {
                status: status.as_u16(),
            });
            assert!(is_redirect_policy_rejection_anyhow(&error));
        }
    }

    #[test]
    fn typed_url_policy_errors_survive_anyhow_context() {
        let error = anyhow::Error::new(UrlPolicyRejection::ForbiddenIpLiteral(
            "127.0.0.1".parse().unwrap(),
        ))
        .context("page target rejected");
        assert!(is_url_policy_rejection_anyhow(&error));
    }

    #[test]
    fn typed_dns_policy_errors_survive_anyhow_context() {
        let error = anyhow::Error::new(DnsPolicyRejection::EmptyAnswerSet(
            "empty.example".to_string(),
        ))
        .context("search provider DNS failed");
        assert!(is_dns_policy_rejection_anyhow(&error));
    }

    #[test]
    fn validates_http_and_https_public_urls() {
        assert!(validate_request_url("https://example.com/path").is_ok());
        assert!(validate_request_url("http://example.com/path").is_ok());
        assert!(validate_request_url("https://8.8.8.8/").is_ok());
    }

    #[test]
    fn rejects_non_public_ip_literal_urls() {
        for url in [
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://[fc00::1]/",
            "http://[::ffff:127.0.0.1]/",
        ] {
            assert!(validate_request_url(url).is_err(), "{url}");
        }
    }

    #[test]
    fn rejects_non_http_schemes_credentials_local_names_and_non_default_ports() {
        for url in [
            "file:///etc/passwd",
            "ftp://example.com/",
            "https://user:secret@example.com/",
            "https://metadata.google.internal/",
            "http://service.local/",
            "http://example.com:8080/",
            "https://example.com:8443/",
        ] {
            assert!(validate_request_url(url).is_err(), "{url}");
        }
    }


    #[tokio::test]
    async fn reqwest_transport_preserves_dns_policy_rejection_source() {
        // This exercises the real reqwest resolver adapter. A private answer must
        // fail during resolution, before reqwest can open a connection.
        let resolver = GuardedDnsResolver::with_lookup(StubLookup::with_answers(vec![vec![
            socket("127.0.0.1"),
        ]]));
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .dns_resolver2(resolver)
            .build()
            .expect("guarded test client");

        let error = client
            .get("https://resolver-policy.example/")
            .send()
            .await
            .expect_err("private DNS answer must be rejected before connecting");
        assert!(
            is_dns_policy_rejection(&error),
            "reqwest must preserve the typed DNS policy cause: {error:?}"
        );
    }
}
