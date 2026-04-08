{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    platformio
    python3
    python3Packages.pyserial
  ];

  shellHook = ''
    echo "--------------------------------------------"
    echo " LT Fault Detection - Dev Environment"
    echo " Firmware:  cd firmware && pio run"
    echo " Monitor:   cd firmware && pio device monitor"
    echo " Data:      python software/capture_data.py"
    echo "--------------------------------------------"
  '';
}
