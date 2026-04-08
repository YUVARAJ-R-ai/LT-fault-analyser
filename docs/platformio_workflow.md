# PlatformIO Setup for NixOS: Project Workflow

## Why PlatformIO on NixOS?
The `platformio` package in Nixpkgs is wrapped in a `buildFHSUserEnv`. This creates a virtual Linux filesystem, allowing PlatformIO's downloaded toolchains (like compilers for ESP8266/ESP32) to work seamlessly on NixOS without manual patching.

## Step 1: System-Level NixOS Configuration

Add the `udev` rule to allow non-sudo USB access:

```nix
# In your NixOS configuration file
services.udev.packages = with pkgs; [ platformio-core ];
```

Then apply and reboot:
```bash
sudo nixos-rebuild switch
```

## Step 2: Per-Project `shell.nix`

Create a `shell.nix` in the project root:

```nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    platformio
  ];
  shellHook = ''
    echo "PlatformIO Dev Environment Ready"
  '';
}
```

Enter with: `nix-shell`

## Step 3: PlatformIO Project Setup

Initialize project for NodeMCU (ESP8266):
```bash
pio project init --board nodemcuv2
```

### `platformio.ini` for this project:

```ini
[env:nodemcuv2]
platform = espressif8266
board = nodemcuv2
framework = arduino
monitor_speed = 250000

lib_deps =
    adafruit/Adafruit ADS1X15 @ ^2.5.0
    plerup/EspSoftwareSerial @ ^6.16.1
```

## Step 4: Build, Flash, Monitor

```bash
pio run                     # Compile
pio run --target upload     # Flash to device
pio device monitor          # Open Serial Monitor
```

## Pinout: NodeMCU <> ADS1115 (I2C)

| ADS1115 Pin | NodeMCU Pin |
| :---------- | :---------- |
| **SCL**     | **D1**      |
| **SDA**     | **D2**      |
| **VCC**     | **3V3**     |
| **GND**     | **GND**     |

## Troubleshooting

### Device Not Detected (`/dev/ttyUSB*` missing)
```bash
sudo systemctl stop ModemManager.service
sudo systemctl disable ModemManager.service
```
Replug the device after stopping.

### Upload Succeeds but Device Misbehaves
- Run `pio device monitor` immediately after upload and check logs.
- Verify wiring meticulously (SCL/SDA often get swapped).
