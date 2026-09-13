// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    let pid = std::process::id();
    let sid = unsafe { libc::getsid(0) };
    let pgrp = unsafe { libc::getpgrp() };
    let no_new_privs = unsafe { libc::prctl(libc::PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) };
    if sid < 0 || pgrp < 0 || no_new_privs < 0 {
        return Err(io::Error::last_os_error());
    }

    let status = fs::read_to_string("/proc/self/status")?;
    let umask = status
        .lines()
        .find_map(|line| line.strip_prefix("Umask:\t"))
        .ok_or_else(|| io::Error::other("/proc/self/status missing Umask"))?;

    let mut stdout = io::stdout().lock();
    writeln!(stdout, "pid={pid}")?;
    writeln!(stdout, "sid={sid}")?;
    writeln!(stdout, "pgrp={pgrp}")?;
    writeln!(stdout, "no_new_privs={no_new_privs}")?;
    writeln!(stdout, "umask={umask}")?;
    print_limit(&mut stdout, "as", libc::RLIMIT_AS)?;
    print_limit(&mut stdout, "cpu", libc::RLIMIT_CPU)?;
    print_limit(&mut stdout, "fsize", libc::RLIMIT_FSIZE)?;
    print_limit(&mut stdout, "nofile", libc::RLIMIT_NOFILE)?;
    print_limit(&mut stdout, "nproc", libc::RLIMIT_NPROC)?;
    print_limit(&mut stdout, "core", libc::RLIMIT_CORE)?;
    writeln!(stdout, "exec_ready")?;
    stdout.flush()?;

    let mut release = [0u8; 1];
    io::stdin().read_exact(&mut release)?;
    if release[0] != 0x5a {
        return Err(io::Error::other("invalid rlimit target release byte"));
    }
    Ok(())
}

fn print_limit(
    writer: &mut impl Write,
    name: &str,
    resource: libc::__rlimit_resource_t,
) -> io::Result<()> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::getrlimit(resource, &mut limit) } != 0 {
        return Err(io::Error::last_os_error());
    }
    writeln!(writer, "rlimit_{name}_cur={}", limit.rlim_cur)?;
    writeln!(writer, "rlimit_{name}_max={}", limit.rlim_max)?;
    Ok(())
}
