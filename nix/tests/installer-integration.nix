{ pkgs }:

let
  nixosTest = pkgs.testers.nixosTest;
in
nixosTest {
  name = "installer-integration";

  nodes = {
    installer = { lib, ... }: {
      services.sshd.enable = true;
      users.users.root.password = "password";
      users.mutableUsers = true;
      
      environment.systemPackages = [ pkgs.python3 ];
    };

    client = { pkgs, ... }: {
      environment.systemPackages = [ pkgs.curl pkgs.openssh ];
    };
  };

  testScript = ''
    installer.start()
    client.start()

    installer.wait_for_unit("sshd.service")
    
    # Verify the installer is listening for connections
    client.wait_until_succeeds("ssh -o StrictHostKeyChecking=no root@installer 'echo hello'")
    
    # Basic check to ensure python is available for the relay scripts
    installer.succeed("python3 --version")
    
    print("Installer integration check passed")
  '';
}
