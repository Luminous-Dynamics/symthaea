// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! RFC 5322 message decode + DSN (RFC 3464) extraction.
//!
//! Thin wrapper over `mail-parser` that gives the rest of the gateway a
//! stable view: either `IncomingMail { to, from, subject, body, attachments }`
//! or `IncomingDsn { original_msg_id, status, diagnostic }`.

use mail_parser::{MessageParser, MimeHeaders};

/// An incoming external mail after RFC 5322 decode.
#[derive(Debug, Clone)]
pub struct IncomingMail {
    pub from: String,
    pub rcpt_to: Vec<String>,
    pub subject: String,
    pub message_id: Option<String>,
    pub body_plain: Option<String>,
    pub body_html: Option<String>,
    pub attachments: Vec<Attachment>,
    /// Timestamp from the Date: header, if present.
    pub date: Option<i64>,
    /// Raw bytes — retained for the re-encrypt + persist path.
    pub raw: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Attachment {
    pub filename: Option<String>,
    pub mime_type: Option<String>,
    pub content: Vec<u8>,
}

/// A Delivery Status Notification (RFC 3464) decoded to the fields we care
/// about for outbound-bounce correlation.
#[derive(Debug, Clone)]
pub struct IncomingDsn {
    /// The VERP-extracted msg_id. Populated by the bounce handler after
    /// decoding `To:`.
    pub original_msg_id: Option<String>,
    /// Final-Recipient field from the message/delivery-status part.
    pub final_recipient: Option<String>,
    /// Status field (RFC 3463 code, e.g. "5.1.1").
    pub status: Option<String>,
    /// Diagnostic-Code field (human-readable explanation).
    pub diagnostic: Option<String>,
}

/// Parse an arbitrary incoming message. If it's a normal mail, returns
/// Mail(_). If it's a DSN, returns Dsn(_). Returns Err on malformed input.
pub fn parse_incoming(raw: &[u8]) -> crate::GatewayResult<ParsedMessage> {
    let msg = MessageParser::default()
        .parse(raw)
        .ok_or_else(|| crate::GatewayError::Parse("mail-parser returned None".into()))?;

    // DSN detection — RFC 3464 mandates `Content-Type: multipart/report;
    // report-type=delivery-status`.
    let is_dsn = msg
        .content_type()
        .and_then(|ct| ct.attribute("report-type"))
        .map(|v| v.eq_ignore_ascii_case("delivery-status"))
        .unwrap_or(false);

    if is_dsn {
        return Ok(ParsedMessage::Dsn(parse_dsn(&msg)));
    }

    let from = msg
        .from()
        .and_then(|f| f.first())
        .and_then(|a| a.address())
        .unwrap_or_default()
        .to_string();

    let rcpt_to: Vec<String> = msg
        .to()
        .map(|hdr| {
            hdr.iter()
                .filter_map(|a| a.address().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let subject = msg.subject().unwrap_or("").to_string();
    let message_id = msg.message_id().map(str::to_string);
    let body_plain = msg.body_text(0).map(|s| s.to_string());
    let body_html = msg.body_html(0).map(|s| s.to_string());
    let date = msg.date().map(|d| d.to_timestamp());

    let mut attachments = Vec::new();
    let att_count: u32 = msg.attachment_count().try_into().unwrap_or(u32::MAX);
    for idx in 0..att_count {
        if let Some(att) = msg.attachment(idx) {
            attachments.push(Attachment {
                filename: att.attachment_name().map(str::to_string),
                mime_type: att
                    .content_type()
                    .map(|ct| format!("{}/{}", ct.ctype(), ct.subtype().unwrap_or(""))),
                content: att.contents().to_vec(),
            });
        }
    }

    Ok(ParsedMessage::Mail(IncomingMail {
        from,
        rcpt_to,
        subject,
        message_id,
        body_plain,
        body_html,
        attachments,
        date,
        raw: raw.to_vec(),
    }))
}

fn parse_dsn(msg: &mail_parser::Message) -> IncomingDsn {
    // RFC 3464 structure: the multipart/report contains a
    // message/delivery-status part. We walk parts looking for one whose
    // content-type indicates delivery-status; its body is a sequence of
    // RFC 822-style field blocks separated by blank lines.
    let mut final_recipient = None;
    let mut status = None;
    let mut diagnostic = None;

    for part in msg.parts.iter() {
        let is_delivery_status = part
            .content_type()
            .map(|ct| {
                ct.ctype().eq_ignore_ascii_case("message")
                    && ct
                        .subtype()
                        .unwrap_or("")
                        .eq_ignore_ascii_case("delivery-status")
            })
            .unwrap_or(false);

        if !is_delivery_status {
            continue;
        }

        let body = part.contents();
        let text = std::str::from_utf8(body).unwrap_or("");
        for line in text.lines() {
            let (key, value) = match line.split_once(':') {
                Some(pair) => pair,
                None => continue,
            };
            let k = key.trim().to_ascii_lowercase();
            let v = value.trim().to_string();
            match k.as_str() {
                "final-recipient" => final_recipient = Some(v),
                "status" => status = Some(v),
                "diagnostic-code" => diagnostic = Some(v),
                _ => {}
            }
        }
    }

    IncomingDsn {
        original_msg_id: None, // VERP-decoded later by bounce handler
        final_recipient,
        status,
        diagnostic,
    }
}

#[derive(Debug, Clone)]
pub enum ParsedMessage {
    Mail(IncomingMail),
    Dsn(IncomingDsn),
}
