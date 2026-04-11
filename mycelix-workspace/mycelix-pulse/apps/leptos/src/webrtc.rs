// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WebRTC adapter for P2P video/audio calls.
//! Uses JS eval for complex RTCPeerConnection operations.

use wasm_bindgen_futures::JsFuture;

/// Create an SDP offer for a new call.
pub async fn create_offer() -> Result<(String, String), String> {
    // Returns (sdp_offer, peer_connection_id)
    let js = r#"(async function() {
        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        window.__mycelix_pc = pc;
        const dc = pc.createDataChannel('mycelix-data');
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        return offer.sdp;
    })()"#;

    let promise = js_sys::eval(js).map_err(|e| format!("{e:?}"))?;
    let sdp = JsFuture::from(js_sys::Promise::from(promise)).await
        .map_err(|e| format!("{e:?}"))?;

    Ok((sdp.as_string().unwrap_or_default(), "pc-0".into()))
}

/// Accept an SDP offer and create an answer.
pub async fn create_answer(remote_sdp: &str) -> Result<String, String> {
    let js = format!(
        r#"(async function() {{
            const pc = new RTCPeerConnection({{
                iceServers: [{{ urls: 'stun:stun.l.google.com:19302' }}]
            }});
            window.__mycelix_pc_remote = pc;
            await pc.setRemoteDescription(new RTCSessionDescription({{ type: 'offer', sdp: `{remote_sdp}` }}));
            const answer = await pc.createAnswer();
            await pc.setLocalDescription(answer);
            return answer.sdp;
        }})()"#,
    );

    let promise = js_sys::eval(&js).map_err(|e| format!("{e:?}"))?;
    let sdp = JsFuture::from(js_sys::Promise::from(promise)).await
        .map_err(|e| format!("{e:?}"))?;

    Ok(sdp.as_string().unwrap_or_default())
}

/// Get user camera/microphone.
pub async fn get_user_media(video: bool, audio: bool) -> Result<(), String> {
    let js = format!(
        r#"(async function() {{
            const stream = await navigator.mediaDevices.getUserMedia({{ video: {video}, audio: {audio} }});
            window.__mycelix_stream = stream;
            return 'ok';
        }})()"#,
    );

    let promise = js_sys::eval(&js).map_err(|e| format!("{e:?}"))?;
    JsFuture::from(js_sys::Promise::from(promise)).await
        .map_err(|e| format!("{e:?}"))?;

    Ok(())
}

/// Get screen share stream.
pub async fn get_display_media() -> Result<(), String> {
    let js = r#"(async function() {
        const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
        window.__mycelix_screen = stream;
        return 'ok';
    })()"#;

    let promise = js_sys::eval(js).map_err(|e| format!("{e:?}"))?;
    JsFuture::from(js_sys::Promise::from(promise)).await
        .map_err(|e| format!("{e:?}"))?;

    Ok(())
}

/// Stop all media tracks.
pub fn stop_media() {
    let _ = js_sys::eval(
        "if(window.__mycelix_stream){window.__mycelix_stream.getTracks().forEach(t=>t.stop())};\
         if(window.__mycelix_screen){window.__mycelix_screen.getTracks().forEach(t=>t.stop())};\
         if(window.__mycelix_pc){window.__mycelix_pc.close()};\
         if(window.__mycelix_pc_remote){window.__mycelix_pc_remote.close()}"
    );
}
