{ pkgs, probe }:

pkgs.testers.runNixOSTest {
  name = "symthaea-agency-executor-config";

  nodes.machine = { pkgs, ... }: {
    system.stateVersion = "26.05";

    systemd.services.symthaea-system-broker = {
      description = "Symthaea Agency Kernel configuration qualification probe";
      wantedBy = [ "multi-user.target" ];
      path = [ pkgs.systemd pkgs.coreutils ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = "${probe} symthaea-system-broker.service";

        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = true;
        MemoryDenyWriteExecute = true;
        LockPersonality = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
      };
    };
  };

  testScript = ''
    machine.start()
    machine.wait_for_unit("multi-user.target")
    machine.wait_for_unit("symthaea-system-broker.service")

    first = machine.succeed(
        "journalctl -b -u symthaea-system-broker.service --no-pager -o cat"
    )
    assert "configuration_digest=" in first, first

    fragment = machine.succeed(
        "systemctl show symthaea-system-broker.service --property=FragmentPath --value"
    ).strip()
    resolved = machine.succeed(f"readlink -f {fragment}").strip()
    assert resolved.startswith("/nix/store/"), resolved

    machine.succeed(
        "mkdir -p /run/systemd/system/symthaea-system-broker.service.d && "
        "printf '[Service]\\nNoNewPrivileges=no\\n' > "
        "/run/systemd/system/symthaea-system-broker.service.d/99-weaken.conf && "
        "systemctl daemon-reload && "
        "systemctl stop symthaea-system-broker.service"
    )

    machine.fail("systemctl start symthaea-system-broker.service")
    weakened = machine.succeed(
        "journalctl -b -u symthaea-system-broker.service --no-pager -o cat"
    )
    assert "hardening policy is not satisfied" in weakened, weakened

    current_nnp = machine.succeed(
        "systemctl show symthaea-system-broker.service --property=NoNewPrivileges --value"
    ).strip()
    assert current_nnp == "no", current_nnp
  '';
}
